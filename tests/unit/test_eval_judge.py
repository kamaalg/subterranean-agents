"""Unit tests for the rubric, the judge JSON parser, and the judge call path.

All LLM calls are mocked — no network, no API key.
"""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage

from subterranean.eval.judge import (
    Judge,
    JudgeConfig,
    build_judge_prompt,
    parse_judge_json,
)
from subterranean.eval.rubric import (
    GRACEFUL_HANDLING_CAP,
    RUBRIC,
    apply_graceful_handling_cap,
)
from subterranean.exceptions import EvalError
from subterranean.generation.formatter import Turn

# --------------------------------------------------------------------------- #
# Rubric                                                                       #
# --------------------------------------------------------------------------- #


def test_rubric_has_five_criteria_in_order() -> None:
    assert RUBRIC.names() == (
        "Task Success",
        "Information Accuracy",
        "Consistency",
        "Graceful Handling",
        "Naturalness",
    )


def test_rubric_prompt_block_is_self_contained() -> None:
    block = RUBRIC.judge_prompt_block()
    for name in RUBRIC.names():
        assert name in block
    assert "1-5" in block
    # The cap rule must be present so the judge knows about it.
    assert str(GRACEFUL_HANDLING_CAP) in block
    assert "cap" in block.lower()


def test_rubric_anchors_are_verbatim() -> None:
    ts = RUBRIC.by_name("Task Success")
    assert ts.anchors[5] == "complete procedure with a clear terminal state"
    assert ts.anchors[3] == "middle stages done but the conversation fizzled"
    assert ts.anchors[1] == "no meaningful progress"


def test_only_graceful_handling_is_capped() -> None:
    capped = [c.name for c in RUBRIC.criteria if c.capped_without_challenge]
    assert capped == ["Graceful Handling"]


def test_rubric_by_name_unknown_raises() -> None:
    with pytest.raises(KeyError):
        RUBRIC.by_name("Nope")


# --------------------------------------------------------------------------- #
# Graceful-Handling cap                                                        #
# --------------------------------------------------------------------------- #


def test_cap_applies_without_challenge() -> None:
    assert apply_graceful_handling_cap(5, user_posed_challenge=False) == 3
    assert apply_graceful_handling_cap(4, user_posed_challenge=False) == 3


def test_cap_no_effect_with_challenge() -> None:
    assert apply_graceful_handling_cap(5, user_posed_challenge=True) == 5


def test_cap_leaves_low_scores_untouched() -> None:
    assert apply_graceful_handling_cap(2, user_posed_challenge=False) == 2


# --------------------------------------------------------------------------- #
# parse_judge_json                                                             #
# --------------------------------------------------------------------------- #

_WELL_FORMED = """
{"user_posed_challenge": true, "criteria": {
  "Task Success": {"score": 4, "justification": "reached terminal"},
  "Information Accuracy": {"score": 5, "justification": "all details kept"},
  "Consistency": {"score": 5, "justification": "no contradictions"},
  "Graceful Handling": {"score": 5, "justification": "adapted well"},
  "Naturalness": {"score": 4, "justification": "reads human"}
}}
"""


def test_parse_well_formed() -> None:
    v = parse_judge_json(_WELL_FORMED)
    assert v.scores["Task Success"] == 4
    assert v.scores["Graceful Handling"] == 5  # challenge posed -> not capped
    assert v.justifications["Information Accuracy"] == "all details kept"
    assert v.user_posed_challenge is True


def test_parse_applies_cap_when_no_challenge() -> None:
    text = (
        '{"user_posed_challenge": false, "criteria": {'
        '"Task Success": {"score": 5}, "Information Accuracy": {"score": 5}, '
        '"Consistency": {"score": 5}, "Graceful Handling": {"score": 5}, '
        '"Naturalness": {"score": 5}}}'
    )
    v = parse_judge_json(text)
    assert v.scores["Graceful Handling"] == 3
    assert v.scores["Naturalness"] == 5


def test_parse_tolerates_markdown_fence_and_prose() -> None:
    text = "Here is my assessment:\n```json\n" + _WELL_FORMED.strip() + "\n```\nDone."
    v = parse_judge_json(text)
    assert v.scores["Consistency"] == 5


def test_parse_extracts_object_from_surrounding_prose() -> None:
    text = "Reasoning... " + _WELL_FORMED.strip() + " ...end."
    v = parse_judge_json(text)
    assert v.scores["Task Success"] == 4


def test_parse_clamps_out_of_range_scores() -> None:
    text = (
        '{"criteria": {"Task Success": {"score": 9}, '
        '"Information Accuracy": {"score": 0}, "Consistency": {"score": 3}, '
        '"Graceful Handling": {"score": 3}, "Naturalness": {"score": 3}}}'
    )
    v = parse_judge_json(text)
    assert v.scores["Task Success"] == 5
    assert v.scores["Information Accuracy"] == 1


def test_parse_accepts_bare_numeric_scores() -> None:
    text = (
        '{"criteria": {"Task Success": 4, "Information Accuracy": 5, '
        '"Consistency": 4, "Graceful Handling": 3, "Naturalness": 5}}'
    )
    v = parse_judge_json(text)
    assert v.scores["Task Success"] == 4


def test_parse_rounds_float_scores() -> None:
    text = (
        '{"criteria": {"Task Success": 4.4, "Information Accuracy": 4.6, '
        '"Consistency": 4, "Graceful Handling": 3, "Naturalness": 5}}'
    )
    v = parse_judge_json(text)
    assert v.scores["Task Success"] == 4
    assert v.scores["Information Accuracy"] == 5


def test_parse_malformed_json_raises() -> None:
    with pytest.raises(EvalError, match="malformed JSON"):
        parse_judge_json('{"criteria": {"Task Success": {"score": 4,}}}')


def test_parse_no_json_raises() -> None:
    with pytest.raises(EvalError, match="no JSON object"):
        parse_judge_json("the agent did fine, 4 out of 5")


def test_parse_unbalanced_object_raises() -> None:
    # An opening brace that never closes (string contents respected).
    with pytest.raises(EvalError, match="unbalanced"):
        parse_judge_json('{"criteria": {"Task Success": "he said \\"hi')


def test_judge_openai_backend_missing_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(EvalError, match="openai"):
        Judge(JudgeConfig(backend="openai", model="gpt-4o"))


def test_parse_missing_criterion_raises() -> None:
    with pytest.raises(EvalError, match="missing criterion"):
        parse_judge_json('{"criteria": {"Task Success": {"score": 4}}}')


def test_parse_non_numeric_score_raises() -> None:
    text = (
        '{"criteria": {"Task Success": {"score": "great"}, '
        '"Information Accuracy": {"score": 5}, "Consistency": {"score": 4}, '
        '"Graceful Handling": {"score": 3}, "Naturalness": {"score": 5}}}'
    )
    with pytest.raises(EvalError, match="non-numeric"):
        parse_judge_json(text)


# --------------------------------------------------------------------------- #
# build_judge_prompt                                                           #
# --------------------------------------------------------------------------- #


def test_judge_prompt_includes_scenario_and_transcript() -> None:
    transcript = [
        Turn(role="user", content="I want to fly to Japan"),
        Turn(role="assistant", content="Great, when?"),
    ]
    prompt = build_judge_prompt(transcript, {"destination": "Japan"}, procedure_description="book")
    assert "Japan" in prompt
    assert "Customer: I want to fly to Japan" in prompt
    assert "Agent: Great, when?" in prompt
    assert "book" in prompt
    for name in RUBRIC.names():
        assert name in prompt


# --------------------------------------------------------------------------- #
# Judge call path (mocked Anthropic)                                           #
# --------------------------------------------------------------------------- #


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return Message(
            id="msg",
            type="message",
            role="assistant",
            model=kwargs["model"],
            content=[TextBlock(type="text", text=self.reply)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=200, output_tokens=80),
        )


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


async def test_judge_score_parses_and_tracks_cost() -> None:
    client = _FakeClient(_WELL_FORMED)
    judge = Judge(JudgeConfig(), client=client)
    verdict = await judge.score(
        [Turn(role="user", content="hi"), Turn(role="assistant", content="hello")],
        {"destination": "Japan"},
    )
    assert verdict.scores["Task Success"] == 4
    assert judge.cost.api_calls == 1
    assert judge.cost.cost_usd > 0
    # The judge system prompt carries the rubric and is cached.
    sent = client.messages.calls[0]
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Task Success" in sent["system"][0]["text"]
