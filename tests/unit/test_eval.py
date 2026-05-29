"""Unit tests for the simulator, baselines/conditions, runner orchestration, and report.

All LLM calls are mocked — no network, no API key. Covers the parsing/aggregation
paths and asserts the user simulator carries zero flowchart knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage

from agent2model.eval.baselines import (
    BASELINE_NAMES,
    CompiledCondition,
    ConditionContext,
    InContextCondition,
    LangGraphCondition,
    SameModelOrchCondition,
    make_condition,
    run_condition,
)
from agent2model.eval.judge import Judge, JudgeConfig, JudgeVerdict
from agent2model.eval.report import write_json_report, write_pdf_report
from agent2model.eval.runner import (
    ConditionResult,
    EvalConfig,
    EvalRunner,
    compare_conditions,
    estimate_eval_cost,
    summarize_condition,
)
from agent2model.eval.simulator import UserSimulator, simulator_system_prompt
from agent2model.exceptions import EvalError
from agent2model.generation.formatter import Turn
from agent2model.ir.loader import load_flowchart_from_string
from agent2model.ir.schema import Flowchart
from agent2model.ir.validator import validate

SAMPLE_YAML = """
name: support
description: Help a customer.
start: greet
nodes:
  greet:
    role: agent
    prompt: Greet the customer.
    next: [ask]
  ask:
    role: agent
    prompt: Ask what they need.
    next: [respond]
  respond:
    role: user
    prompt: Reply as the customer.
    next: [decide]
  decide:
    role: decision
    next:
      - to: resolve
        when: satisfied
      - to: ask
        when: needs more
  resolve:
    role: agent
    prompt: Wrap up.
    next: [done]
  done:
    terminal: success
scenario_variables:
  user_styles: [calm, angry]
  destination: [Japan, Italy]
"""


@pytest.fixture
def flowchart() -> Flowchart:
    fc = load_flowchart_from_string(SAMPLE_YAML)
    validate(fc)
    return fc


# --------------------------------------------------------------------------- #
# Mocks                                                                        #
# --------------------------------------------------------------------------- #

_VERDICT_JSON = (
    '{"user_posed_challenge": true, "criteria": {'
    '"Task Success": {"score": 4}, "Information Accuracy": {"score": 5}, '
    '"Consistency": {"score": 5}, "Graceful Handling": {"score": 4}, '
    '"Naturalness": {"score": 4}}}'
)


class _ScriptedMessages:
    """Returns canned replies; a simulator reply ends the convo after a few turns."""

    def __init__(self, replies: list[str], *, judge_reply: str | None = None) -> None:
        self._replies = replies
        self._judge_reply = judge_reply
        self.calls: list[dict[str, Any]] = []
        self._i = 0

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        system = kwargs.get("system")
        text_system = system if isinstance(system, str) else (system[0]["text"] if system else "")
        if self._judge_reply is not None and "evaluator" in text_system.lower():
            reply = self._judge_reply
        else:
            reply = self._replies[self._i % len(self._replies)]
            self._i += 1
        return Message(
            id="m",
            type="message",
            role="assistant",
            model=kwargs["model"],
            content=[TextBlock(type="text", text=reply)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=40),
        )


class _Client:
    def __init__(self, replies: list[str], *, judge_reply: str | None = None) -> None:
        self.messages = _ScriptedMessages(replies, judge_reply=judge_reply)


class _StaticResponder:
    """Agent responder that returns a fixed reply (no network)."""

    def __init__(self, reply: str = "Agent reply.") -> None:
        self.reply = reply
        self.count = 0

    async def respond(self, transcript: list[Turn]) -> str:
        self.count += 1
        return self.reply


# --------------------------------------------------------------------------- #
# Simulator — NO flowchart knowledge                                           #
# --------------------------------------------------------------------------- #


def test_simulator_prompt_has_no_flowchart_knowledge() -> None:
    prompt = simulator_system_prompt(
        {"destination": "Japan", "style": "skeptical"}, personality="skeptical"
    )
    import re

    low = prompt.lower()
    # No flowchart structure or node ids may leak into the simulator's prompt.
    assert "flowchart" not in low
    assert "when:" not in low
    assert "next:" not in low
    for node_id in ["greet", "resolve", "decide", "respond"]:
        assert re.search(rf"\b{node_id}\b", low) is None
    # The only mention of "procedure" is the explicit denial of any knowledge of it.
    assert "no knowledge of any internal procedure" in low
    assert "Japan" in prompt


def test_simulator_carries_personality_and_facts() -> None:
    prompt = simulator_system_prompt({"destination": "Italy"}, personality="impatient")
    assert "impatient" in prompt
    assert "Italy" in prompt


async def test_simulator_next_message_flips_roles_and_detects_done() -> None:
    client = _Client(["I need help [[END]]"])
    sim = UserSimulator({"destination": "Japan"}, client=client)
    text, done = await sim.next_message(
        [Turn(role="user", content="hi"), Turn(role="assistant", content="how can I help?")]
    )
    assert done is True
    assert "[[END]]" not in text
    sent = client.messages.calls[0]["messages"]
    # The agent's turn ("assistant") is presented to the simulator as "user".
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"] == "how can I help?"


async def test_simulator_primes_opening_turn() -> None:
    client = _Client(["Hello, I have a question."])
    sim = UserSimulator({}, client=client)
    text, done = await sim.next_message([])
    assert text
    assert done is False
    # With an empty transcript the simulator is prompted to open.
    assert client.messages.calls[0]["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------- #
# run_condition / conversation loop                                            #
# --------------------------------------------------------------------------- #


async def test_run_condition_alternates_until_done() -> None:
    # Simulator says one thing then signals done on its 2nd turn.
    client = _Client(["First customer message", "All set thanks [[END]]"])
    sim = UserSimulator({}, client=client)
    ctx = ConditionContext({}, sim, max_turns=12)
    responder = _StaticResponder()
    convo = await run_condition(responder, ctx)
    roles = [t.role for t in convo.turns]
    # user, assistant, user(done) -> stops.
    assert roles == ["user", "assistant", "user"]
    assert responder.count == 1


async def test_run_condition_respects_max_turns() -> None:
    # Simulator never says done; the cap stops the loop.
    client = _Client(["keep going"])
    sim = UserSimulator({}, client=client)
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await run_condition(_StaticResponder(), ctx)
    assert len(convo.turns) <= 4


# --------------------------------------------------------------------------- #
# Conditions                                                                   #
# --------------------------------------------------------------------------- #


async def test_in_context_condition_serializes_flowchart(flowchart: Flowchart) -> None:
    client = _Client(["customer msg", "thanks [[END]]"])
    cond = InContextCondition(flowchart, client=client)
    # The in-context system prompt must actually carry the procedure.
    assert "PROCEDURE" in cond._system
    assert "greet" in cond._system
    sim = UserSimulator({"destination": "Japan"}, client=_Client(["hi", "bye [[END]]"]))
    ctx = ConditionContext({"destination": "Japan"}, sim, max_turns=6)
    convo = await cond.run_scenario(ctx)
    assert convo.turns


async def test_same_model_orch_uses_given_model(flowchart: Flowchart) -> None:
    client = _Client(["agent reply"])
    cond = SameModelOrchCondition(flowchart, model="qwen2.5-3b", client=client)
    sim = UserSimulator({}, client=_Client(["hi", "done [[END]]"]))
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await cond.run_scenario(ctx)
    assert convo.turns
    # The agent calls used the base model id.
    assert any(c["model"] == "qwen2.5-3b" for c in client.messages.calls)


async def test_langgraph_condition_with_injected_responder(flowchart: Flowchart) -> None:
    # Inject a responder so the langgraph package is not required for the test.
    cond = LangGraphCondition(flowchart, responder=_StaticResponder("orchestrated"))
    sim = UserSimulator({}, client=_Client(["hi", "ok [[END]]"]))
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await cond.run_scenario(ctx)
    assert any(t.content == "orchestrated" for t in convo.turns)


async def test_compiled_condition_with_injected_responder() -> None:
    cond = CompiledCondition(
        model="compiled", served_url="http://x", responder=_StaticResponder("from weights")
    )
    sim = UserSimulator({}, client=_Client(["hi", "bye [[END]]"]))
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await cond.run_scenario(ctx)
    assert any(t.content == "from weights" for t in convo.turns)


# A minimal fake OpenAI-compatible async client for the compiled condition.
class _FakeOpenAIChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> Any:
        return self

    async def create(self, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.calls.append(kwargs)
        choice = SimpleNamespace(message=SimpleNamespace(content=self.reply))
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20)
        return SimpleNamespace(choices=[choice], usage=usage)


async def test_compiled_condition_talks_to_openai_compatible_endpoint() -> None:
    fake = _FakeOpenAIChat("compiled-model reply")
    cond = CompiledCondition(model="compiled", served_url="http://x/v1", client=fake)
    sim = UserSimulator({}, client=_Client(["hi", "thanks [[END]]"]))
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await cond.run_scenario(ctx)
    assert any(t.content == "compiled-model reply" for t in convo.turns)
    # No flowchart system prompt is given to the compiled model — it self-orchestrates.
    sent_messages = fake.calls[0]["messages"]
    assert all(m["role"] != "system" for m in sent_messages)
    assert cond.cost.api_calls >= 1


async def test_langgraph_condition_real_graph(flowchart: Flowchart) -> None:
    # Exercises the real LangGraph orchestration (langgraph is a dev dependency).
    # The frontier model is mocked via the injected Anthropic client.
    cond_client = _Client(["graph-orchestrated reply"])
    cond = LangGraphCondition(flowchart, client=cond_client)
    sim = UserSimulator({}, client=_Client(["hi", "done [[END]]"]))
    ctx = ConditionContext({}, sim, max_turns=4)
    convo = await cond.run_scenario(ctx)
    assert any(t.content == "graph-orchestrated reply" for t in convo.turns)


def test_make_condition_known_names(flowchart: Flowchart) -> None:
    for name in BASELINE_NAMES:
        cond = make_condition(name, flowchart)
        assert cond.name == name


def test_make_condition_compiled_requires_url(flowchart: Flowchart) -> None:
    with pytest.raises(EvalError, match="served-url"):
        make_condition("compiled", flowchart)
    cond = make_condition("compiled", flowchart, served_url="http://localhost:8000/v1")
    assert cond.name == "compiled"


def test_make_condition_unknown_raises(flowchart: Flowchart) -> None:
    with pytest.raises(EvalError, match="Unknown condition"):
        make_condition("bogus", flowchart)


# --------------------------------------------------------------------------- #
# Aggregation (pure)                                                           #
# --------------------------------------------------------------------------- #


def _verdict(**scores: int) -> JudgeVerdict:
    base = {
        "Task Success": 4,
        "Information Accuracy": 4,
        "Consistency": 4,
        "Graceful Handling": 4,
        "Naturalness": 4,
    }
    base.update(scores)
    return JudgeVerdict(scores=base, user_posed_challenge=True)  # type: ignore[arg-type]


def test_summarize_condition_aggregates() -> None:
    verdicts = [
        _verdict(**{"Task Success": 5}),
        _verdict(**{"Task Success": 2}),
        _verdict(**{"Task Success": 4}),
        _verdict(**{"Task Success": 3}),
    ]
    result = summarize_condition(
        "in_context", verdicts, cost_usd=0.5, wall_clock_s=[1.0, 2.0, 3.0, 4.0]
    )
    assert result.condition == "in_context"
    assert result.n_conversations == 4
    # Task Success <= 3 for scores 2 and 3 -> 2 of 4.
    assert result.failure_rate == pytest.approx(0.5)
    assert result.avg_wall_clock_s == pytest.approx(2.5)
    ts = next(cs for cs in result.criterion_stats if cs.criterion == "Task Success")
    assert ts.ci_low <= ts.mean <= ts.ci_high


def test_compare_conditions_runs_holm() -> None:
    strong = [_verdict(**{"Task Success": 5, "Naturalness": 5}) for _ in range(10)]
    weak = [_verdict(**{"Task Success": 2, "Naturalness": 2}) for _ in range(10)]
    a = summarize_condition("compiled", strong, cost_usd=0.1, wall_clock_s=[1.0] * 10)
    b = summarize_condition("in_context", weak, cost_usd=1.0, wall_clock_s=[1.0] * 10)
    comp = compare_conditions(a, b, paired=True)
    assert comp.condition_a == "compiled"
    assert set(comp.pvalues) == set(a.scores)
    assert comp.significant["Task Success"] is True


def test_estimate_eval_cost_positive() -> None:
    assert estimate_eval_cost(EvalConfig(n=10), 2) > 0


# --------------------------------------------------------------------------- #
# Runner orchestration (mocked judge + simulator + condition)                  #
# --------------------------------------------------------------------------- #


async def test_runner_end_to_end_mocked(flowchart: Flowchart) -> None:
    judge_client = _Client(["unused"], judge_reply=_VERDICT_JSON)
    judge = Judge(JudgeConfig(), client=judge_client)
    # A single in-context condition driven by a scripted agent client.
    cond_client = _Client(["agent says hi"])
    condition = InContextCondition(flowchart, client=cond_client)

    config = EvalConfig(n=3, budget_usd=100, max_concurrent=2, max_turns=4)
    runner = EvalRunner(
        flowchart,
        [condition],
        config,
        judge=judge,
        simulator_client=_Client(["customer", "done [[END]]"]),
    )
    result = await runner.run()
    assert result.flowchart_name == "support"
    assert result.n == 3
    assert len(result.conditions) == 1
    cr = result.conditions[0]
    assert cr.n_conversations == 3
    assert cr.condition == "in_context"
    # Judge returned Task Success 4 for all -> failure rate 0.
    assert cr.failure_rate == 0.0
    assert result.total_cost_usd > 0


async def test_runner_compares_all_pairs_without_compiled(flowchart: Flowchart) -> None:
    judge = Judge(JudgeConfig(), client=_Client(["x"], judge_reply=_VERDICT_JSON))
    c1 = InContextCondition(flowchart, client=_Client(["a"]))
    c2 = SameModelOrchCondition(flowchart, model="qwen2.5-3b", client=_Client(["b"]))
    config = EvalConfig(n=2, budget_usd=100, max_concurrent=2, max_turns=4)
    runner = EvalRunner(
        flowchart,
        [c1, c2],
        config,
        judge=judge,
        simulator_client=_Client(["c", "done [[END]]"]),
    )
    result = await runner.run()
    # No compiled condition -> one unordered pair (in_context vs same_model_orch).
    assert len(result.comparisons) == 1
    pair = result.comparisons[0]
    assert {pair.condition_a, pair.condition_b} == {"in_context", "same_model_orch"}


async def test_runner_budget_guard(flowchart: Flowchart) -> None:
    from agent2model.exceptions import EvalBudgetExceeded

    # Huge token usage with a tiny budget -> must raise.
    class _BigMessages(_ScriptedMessages):
        async def create(self, **kwargs: Any) -> Message:
            msg = await super().create(**kwargs)
            msg.usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
            return msg

    judge_client = _Client(["x"], judge_reply=_VERDICT_JSON)
    judge_client.messages = _BigMessages(["x"], judge_reply=_VERDICT_JSON)
    judge = Judge(JudgeConfig(), client=judge_client)
    cond_client = _Client(["hi"])
    cond_client.messages = _BigMessages(["hi"])
    condition = InContextCondition(flowchart, client=cond_client)
    sim_client = _Client(["c", "done [[END]]"])
    sim_client.messages = _BigMessages(["c", "done [[END]]"])

    config = EvalConfig(n=5, budget_usd=0.000001, max_concurrent=1, max_turns=4)
    runner = EvalRunner(flowchart, [condition], config, judge=judge, simulator_client=sim_client)
    with pytest.raises(EvalBudgetExceeded):
        await runner.run()


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #


def _run_result() -> Any:
    from agent2model.eval.runner import EvalRunResult

    a = summarize_condition(
        "compiled", [_verdict() for _ in range(5)], cost_usd=0.005, wall_clock_s=[1.0] * 5
    )
    b = summarize_condition(
        "in_context", [_verdict() for _ in range(5)], cost_usd=0.5, wall_clock_s=[2.0] * 5
    )
    comp = compare_conditions(a, b, paired=True)
    return EvalRunResult(
        flowchart_name="support",
        n=5,
        conditions=[a, b],
        comparisons=[comp],
        total_cost_usd=0.505,
    )


def test_write_json_report(tmp_path: Path) -> None:
    result = _run_result()
    path = write_json_report(result, tmp_path / "eval_report.json")
    data = json.loads(path.read_text())
    assert data["flowchart_name"] == "support"
    assert len(data["conditions"]) == 2
    assert data["conditions"][0]["criterion_stats"][0]["criterion"] == "Task Success"


def test_write_pdf_report_handles_missing_matplotlib(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        with pytest.raises(EvalError, match="matplotlib"):
            write_pdf_report(_run_result(), tmp_path / "r.pdf")
    else:
        out = write_pdf_report(_run_result(), tmp_path / "r.pdf")
        assert out.exists()
        assert out.stat().st_size > 0


def test_condition_result_roundtrips() -> None:
    cr = summarize_condition(
        "x", [_verdict() for _ in range(3)], cost_usd=0.1, wall_clock_s=[1.0] * 3
    )
    dumped = cr.model_dump(mode="json")
    restored = ConditionResult.model_validate(dumped)
    assert restored.condition == "x"
