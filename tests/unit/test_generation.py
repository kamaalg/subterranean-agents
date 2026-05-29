"""Unit tests for Phase 2 synthetic data generation.

All Anthropic calls are mocked — these tests run with no network and no API key.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage

from agent2model.exceptions import FlowchartValidationError, GenerationBudgetExceeded
from agent2model.generation.formatter import Conversation, Turn, write_dataset
from agent2model.generation.generator import (
    DEFAULT_MODEL,
    ConversationGenerator,
    CostTracker,
    GenerationConfig,
    build_system_prompt,
    estimate_cost,
)
from agent2model.generation.scenarios import sample_scenario
from agent2model.generation.traversal import TraversalConfig, sample_path
from agent2model.ir.loader import load_flowchart_from_string
from agent2model.ir.schema import Flowchart
from agent2model.ir.validator import validate

# A small valid flowchart exercising agent, user, and decision roles plus a cycle.
SAMPLE_YAML = """
name: support
description: Help a customer.
start: greet
nodes:
  greet:
    role: agent
    prompt: Greet the {user_styles} customer.
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
  budget_range: [100, 200]
"""


@pytest.fixture
def sample_flowchart() -> Flowchart:
    fc = load_flowchart_from_string(SAMPLE_YAML)
    validate(fc)
    return fc


# --------------------------------------------------------------------------- #
# Anthropic mock                                                               #
# --------------------------------------------------------------------------- #


class _FakeMessages:
    """Stand-in for ``client.messages`` with a controllable usage footprint."""

    def __init__(self, input_tokens: int, output_tokens: int, cache_read: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read = cache_read
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return Message(
            id="msg_test",
            type="message",
            role="assistant",
            model=kwargs["model"],
            content=[TextBlock(type="text", text="generated turn")],
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_read_input_tokens=self.cache_read,
            ),
        )


class _FakeClient:
    def __init__(
        self, input_tokens: int = 100, output_tokens: int = 50, cache_read: int = 0
    ) -> None:
        self.messages = _FakeMessages(input_tokens, output_tokens, cache_read)


# --------------------------------------------------------------------------- #
# Traversal                                                                    #
# --------------------------------------------------------------------------- #


def test_sample_path_reaches_terminal(sample_flowchart: Flowchart) -> None:
    for seed in range(20):
        path = sample_path(sample_flowchart, random.Random(seed))
        assert path[0] == sample_flowchart.start
        assert sample_flowchart.nodes[path[-1]].is_terminal


def test_sample_path_is_seed_deterministic(sample_flowchart: Flowchart) -> None:
    p1 = sample_path(sample_flowchart, random.Random(42))
    p2 = sample_path(sample_flowchart, random.Random(42))
    assert p1 == p2


def test_sample_path_different_seeds_can_differ(sample_flowchart: Flowchart) -> None:
    paths = {tuple(sample_path(sample_flowchart, random.Random(s))) for s in range(30)}
    # The decision node + cycle should yield more than one distinct path.
    assert len(paths) > 1


def test_edge_weights_bias_selection(sample_flowchart: Flowchart) -> None:
    # Force the decision node almost always to resolve.
    cfg = TraversalConfig(edge_weights={"decide->resolve": 1000.0, "decide->ask": 0.001})
    lengths = [len(sample_path(sample_flowchart, random.Random(s), config=cfg)) for s in range(20)]
    # With resolve strongly favoured, paths should be short (no extra ask loops).
    assert max(lengths) <= 6


def test_sample_path_unbounded_cycle_raises() -> None:
    # Pure cycle with no escape: validate() would reject it, but the sampler
    # must also fail loudly rather than hang if handed an unvalidated graph.
    fc = Flowchart.model_validate(
        {
            "name": "trap",
            "start": "a",
            "nodes": {
                "a": {"role": "agent", "next": ["a"]},
                "done": {"terminal": "success"},
            },
        }
    )
    with pytest.raises(FlowchartValidationError, match="max_steps"):
        sample_path(fc, random.Random(0), config=TraversalConfig(max_steps=5))


def test_sample_path_dead_end_raises() -> None:
    fc = Flowchart.model_validate(
        {
            "name": "deadend",
            "start": "a",
            "nodes": {
                "a": {"role": "agent", "next": []},
                "done": {"terminal": "success"},
            },
        }
    )
    with pytest.raises(FlowchartValidationError, match="no outgoing edges"):
        sample_path(fc, random.Random(0))


def test_traversal_max_steps_terminal_bias_breaks_cycle() -> None:
    fc = load_flowchart_from_string("""
        name: loopy
        start: a
        nodes:
          a:
            role: decision
            next:
              - to: a
                when: loop
              - to: done
                when: escape
          done:
            terminal: success
        """)
    validate(fc)
    cfg = TraversalConfig(max_steps=200)
    path = sample_path(fc, random.Random(1), config=cfg)
    assert path[-1] == "done"


# --------------------------------------------------------------------------- #
# Scenarios                                                                    #
# --------------------------------------------------------------------------- #


def test_scenario_respects_pools(sample_flowchart: Flowchart) -> None:
    for seed in range(20):
        s = sample_scenario(sample_flowchart, random.Random(seed))
        assert s["user_styles"] in ["calm", "angry"]
        assert 100 <= s["budget_range"] <= 200
        assert isinstance(s["budget_range"], int)


def test_scenario_is_seedable(sample_flowchart: Flowchart) -> None:
    assert sample_scenario(sample_flowchart, random.Random(7)) == sample_scenario(
        sample_flowchart, random.Random(7)
    )


def test_scenario_float_range() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a: {role: agent, prompt: hi, next: [done]}
          done: {terminal: success}
        scenario_variables:
          rate: [0.0, 1.0]
          fixed: hello
        """)
    s = sample_scenario(fc, random.Random(3))
    assert 0.0 <= s["rate"] <= 1.0
    assert isinstance(s["rate"], float)
    assert s["fixed"] == "hello"


def test_scenario_range_reversed_bounds() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a: {role: agent, prompt: hi, next: [done]}
          done: {terminal: success}
        scenario_variables:
          backwards: [200, 100]
        """)
    s = sample_scenario(fc, random.Random(0))
    assert 100 <= s["backwards"] <= 200


def test_scenario_empty_pool_raises() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a: {role: agent, prompt: hi, next: [done]}
          done: {terminal: success}
        scenario_variables:
          empty: []
        """)
    with pytest.raises(ValueError, match="empty pool"):
        sample_scenario(fc, random.Random(0))


def test_scenario_empty_when_no_variables() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a: {role: agent, prompt: hi, next: [done]}
          done: {terminal: success}
        """)
    assert sample_scenario(fc, random.Random(0)) == {}


# --------------------------------------------------------------------------- #
# Cost tracking                                                                #
# --------------------------------------------------------------------------- #


def test_cost_tracker_pricing() -> None:
    ct = CostTracker(model="claude-sonnet-4-5")
    ct.add_usage(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert ct.cost_usd == pytest.approx(18.0)  # 3 + 15
    assert ct.api_calls == 1


def test_cost_tracker_cache_discount() -> None:
    ct = CostTracker(model="claude-sonnet-4-5")
    ct.add_usage(
        Usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
        )
    )
    # cache read 3*0.1 + cache write 3*1.25 = 0.3 + 3.75
    assert ct.cost_usd == pytest.approx(4.05)


def test_estimate_cost_positive() -> None:
    assert estimate_cost(GenerationConfig(n=100, budget_usd=10)) > 0


# --------------------------------------------------------------------------- #
# Generator                                                                    #
# --------------------------------------------------------------------------- #


def test_generate_one_deterministic(sample_flowchart: Flowchart) -> None:
    cfg = GenerationConfig(n=1, budget_usd=100, seed=5)
    gen = ConversationGenerator(sample_flowchart, cfg, client=_FakeClient())

    async def _go() -> tuple[list[str], list[str]]:
        c1 = await gen.generate_one(0)
        c2 = await ConversationGenerator(sample_flowchart, cfg, client=_FakeClient()).generate_one(
            0
        )
        return [t.role for t in c1.turns], [t.role for t in c2.turns]

    import asyncio

    roles1, roles2 = asyncio.run(_go())
    assert roles1 == roles2
    # No flowchart node ids should leak into roles; only user/assistant.
    assert set(roles1) <= {"user", "assistant"}


def test_generate_one_tolerates_unknown_placeholder(tmp_path: Path) -> None:
    # A prompt referencing a variable not in the scenario pools must not crash.
    fc = load_flowchart_from_string("""
        name: ph
        start: a
        nodes:
          a: {role: agent, prompt: "Hello {missing_var}!", next: [done]}
          done: {terminal: success}
        """)
    validate(fc)
    cfg = GenerationConfig(n=1, budget_usd=100)
    gen = ConversationGenerator(fc, cfg, client=_FakeClient())

    import asyncio

    conv = asyncio.run(gen.generate_one(0))
    assert conv.turns[0].role == "assistant"


def test_generator_run_writes_cost_and_dataset(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    cfg = GenerationConfig(n=4, budget_usd=100, seed=1, max_concurrent=2)
    gen = ConversationGenerator(sample_flowchart, cfg, client=_FakeClient())

    import asyncio

    convos = asyncio.run(gen.run(tmp_path))
    assert len(convos) == 4

    cost_json = json.loads((tmp_path / "cost.json").read_text())
    assert cost_json["model"] == DEFAULT_MODEL
    assert cost_json["api_calls"] > 0
    assert cost_json["cost_usd"] > 0

    state = json.loads((tmp_path / "generation_state.json").read_text())
    assert len(state["conversations"]) == 4

    written = write_dataset(convos, tmp_path / "dataset.jsonl")
    assert written == 4


def test_budget_hard_stop(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    # Each call costs a lot; budget is tiny -> must raise.
    client = _FakeClient(input_tokens=1_000_000, output_tokens=1_000_000)
    cfg = GenerationConfig(n=10, budget_usd=0.000001, max_concurrent=1)
    gen = ConversationGenerator(sample_flowchart, cfg, client=client)

    import asyncio

    with pytest.raises(GenerationBudgetExceeded):
        asyncio.run(gen.run(tmp_path))

    # Cost file still written on the hard stop.
    assert (tmp_path / "cost.json").exists()


def test_resume_skips_completed(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    cfg = GenerationConfig(n=6, budget_usd=100, max_concurrent=2)
    client1 = _FakeClient()
    gen1 = ConversationGenerator(sample_flowchart, cfg, client=client1)

    import asyncio

    asyncio.run(gen1.run(tmp_path))
    first_calls = len(client1.messages.calls)
    assert first_calls > 0

    # Second run with the same build dir should make no new API calls.
    client2 = _FakeClient()
    gen2 = ConversationGenerator(sample_flowchart, cfg, client=client2)
    convos = asyncio.run(gen2.run(tmp_path))
    assert len(client2.messages.calls) == 0
    assert len(convos) == 6


def test_resume_restores_cost(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    cfg = GenerationConfig(n=3, budget_usd=100)
    asyncio_run_first = ConversationGenerator(sample_flowchart, cfg, client=_FakeClient())

    import asyncio

    asyncio.run(asyncio_run_first.run(tmp_path))
    first_cost = json.loads((tmp_path / "cost.json").read_text())["cost_usd"]

    gen2 = ConversationGenerator(sample_flowchart, cfg, client=_FakeClient())
    asyncio.run(gen2.run(tmp_path))
    # No new calls -> restored cost is unchanged, not reset to zero.
    assert gen2.cost.cost_usd == pytest.approx(first_cost)


def test_partial_resume_only_generates_remaining(
    sample_flowchart: Flowchart, tmp_path: Path
) -> None:
    # Pre-seed a checkpoint with index 0 already complete.
    state = {
        "conversations": {
            "0": Conversation(turns=[Turn(role="user", content="hi")]).model_dump(mode="json")
        },
        "cost": CostTracker(model=DEFAULT_MODEL).to_report(),
    }
    (tmp_path / "generation_state.json").write_text(json.dumps(state))

    cfg = GenerationConfig(n=3, budget_usd=100, max_concurrent=1)
    client = _FakeClient()
    gen = ConversationGenerator(sample_flowchart, cfg, client=client)

    import asyncio

    convos = asyncio.run(gen.run(tmp_path))
    assert len(convos) == 3
    # Index 0 reused (its content was the canned "hi"); indices 1 and 2 generated.
    assert convos[0].turns[0].content == "hi"


# --------------------------------------------------------------------------- #
# Formatter — no flowchart leakage                                             #
# --------------------------------------------------------------------------- #


def test_dataset_is_valid_chat_template(tmp_path: Path) -> None:
    convos = [
        Conversation(
            turns=[
                Turn(role="user", content="I need help"),
                Turn(role="assistant", content="Sure!"),
            ]
        )
    ]
    path = tmp_path / "dataset.jsonl"
    assert write_dataset(convos, path) == 1
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"messages"}
    for msg in record["messages"]:
        assert set(msg) == {"role", "content"}
        assert msg["role"] in {"user", "assistant"}


def test_no_flowchart_leakage(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    cfg = GenerationConfig(n=5, budget_usd=100)
    gen = ConversationGenerator(sample_flowchart, cfg, client=_FakeClient())

    import asyncio

    convos = asyncio.run(gen.run(tmp_path))
    path = tmp_path / "dataset.jsonl"
    write_dataset(convos, path)
    raw = path.read_text()
    # Node ids, roles, and condition keywords must never appear in the dataset.
    for token in ["greet", "decide", "decision", "terminal", "when", "next:", "respond"]:
        assert token not in raw


def test_empty_conversations_skipped(tmp_path: Path) -> None:
    convos = [Conversation(turns=[]), Conversation(turns=[Turn(role="user", content="hi")])]
    assert write_dataset(convos, tmp_path / "d.jsonl") == 1


def test_build_system_prompt_mentions_no_node_internals(sample_flowchart: Flowchart) -> None:
    sp = build_system_prompt(sample_flowchart)
    assert "support" in sp
    # The agent system prompt frames the role but does not dump node ids.
    assert "decide" not in sp
    assert "greet" not in sp


# --------------------------------------------------------------------------- #
# Integration (skipped without an API key)                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_tiny_real_generation(sample_flowchart: Flowchart, tmp_path: Path) -> None:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY unset; skipping real-API integration test")

    import asyncio

    cfg = GenerationConfig(n=2, budget_usd=1.0, max_concurrent=2)
    gen = ConversationGenerator(sample_flowchart, cfg)
    convos = asyncio.run(gen.run(tmp_path))
    assert convos
    assert (tmp_path / "cost.json").exists()
