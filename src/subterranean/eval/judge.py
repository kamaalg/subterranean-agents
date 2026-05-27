"""LLM-as-judge scoring a conversation against the 5-criterion rubric.

The judge is given a completed conversation plus the scenario context and returns
a per-criterion integer score (``1``..``5``) with a brief justification, as
structured JSON. The default judge is Claude (``claude-sonnet-4-5``) via
:class:`~anthropic.AsyncAnthropic`; a GPT-4 option is configurable but Anthropic
is the default and the only backend that must work locally.

Two rubric rules are enforced *after* parsing, not left to the model:

- the **Graceful-Handling cap** (see
  :func:`subterranean.eval.rubric.apply_graceful_handling_cap`), and
- score clamping into ``[1, 5]`` so a stray out-of-range value can never poison
  the statistics.

JSON parsing is deliberately robust: the judge may wrap its object in prose or a
markdown fence, so we extract the first balanced ``{...}`` block before parsing.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from pydantic import BaseModel, ConfigDict, Field

from subterranean.eval.rubric import (
    MAX_SCORE,
    MIN_SCORE,
    RUBRIC,
    CriterionName,
    Rubric,
    apply_graceful_handling_cap,
)
from subterranean.exceptions import EvalError
from subterranean.generation.generator import DEFAULT_MODEL, CostTracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types import Message

    from subterranean.generation.formatter import Turn
    from subterranean.generation.scenarios import Scenario

__all__ = ["Judge", "JudgeConfig", "JudgeVerdict", "parse_judge_json"]

JudgeBackend = Literal["anthropic", "openai"]
"""Which provider backs the judge. ``anthropic`` (Claude) is the default."""


class JudgeVerdict(BaseModel):
    """A judge's per-criterion scores plus justifications for one conversation.

    Attributes:
        scores: Final integer score (``1``..``5``) per criterion name, after the
            Graceful-Handling cap and range clamping.
        justifications: One-line justification per criterion (best-effort; may be
            empty if the judge omitted it).
        user_posed_challenge: Whether the judge determined the user posed any
            challenge — drives the Graceful-Handling cap.
    """

    model_config = ConfigDict(extra="forbid")

    scores: dict[CriterionName, int]
    justifications: dict[CriterionName, str] = Field(default_factory=dict)
    user_posed_challenge: bool = False


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge.

    Attributes:
        backend: ``anthropic`` (default, Claude) or ``openai`` (GPT-4).
        model: Model id for the chosen backend. Defaults to
            :data:`~subterranean.generation.generator.DEFAULT_MODEL` for
            Anthropic.
        max_tokens: ``max_tokens`` per judge call.
    """

    model_config = ConfigDict(extra="forbid")

    backend: JudgeBackend = "anthropic"
    model: str = DEFAULT_MODEL
    max_tokens: int = Field(default=1024, gt=0)


def _extract_json_object(text: str) -> str:
    """Extract the first balanced ``{...}`` object from arbitrary judge text.

    Tolerates markdown fences and surrounding prose by scanning for the first
    ``{`` and returning through its matching ``}`` (respecting strings/escapes).

    Args:
        text: Raw judge output.

    Returns:
        The substring spanning the first balanced JSON object.

    Raises:
        EvalError: If no balanced object can be found.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise EvalError(f"judge returned no JSON object: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise EvalError(f"judge returned an unbalanced JSON object: {text[:200]!r}")


def parse_judge_json(text: str, *, rubric: Rubric = RUBRIC) -> JudgeVerdict:
    """Parse a judge's raw output into a validated :class:`JudgeVerdict`.

    The expected shape is::

        {
          "user_posed_challenge": true,
          "criteria": {
            "Task Success": {"score": 4, "justification": "..."},
            ...
          }
        }

    Scores are clamped into ``[1, 5]`` and the Graceful-Handling cap is applied,
    so the returned verdict is always rubric-legal regardless of model misbehaviour.

    Args:
        text: Raw judge output (may include prose or a markdown fence).
        rubric: Rubric defining the expected criterion names.

    Returns:
        The validated, rule-corrected verdict.

    Raises:
        EvalError: If the JSON cannot be located/parsed, or a criterion score is
            missing or non-numeric.

    Example:
        >>> v = parse_judge_json('{"criteria": {"Task Success": {"score": 4}, '
        ...     '"Information Accuracy": {"score": 5}, "Consistency": {"score": 5}, '
        ...     '"Graceful Handling": {"score": 5}, "Naturalness": {"score": 4}}}')
        >>> v.scores["Graceful Handling"]  # capped: no challenge flag -> 3
        3
    """
    try:
        data: dict[str, Any] = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise EvalError(f"judge returned malformed JSON: {exc}") from exc

    criteria = data.get("criteria", data)
    if not isinstance(criteria, dict):
        raise EvalError("judge JSON has no 'criteria' object")
    user_posed_challenge = bool(data.get("user_posed_challenge", False))

    scores: dict[CriterionName, int] = {}
    justifications: dict[CriterionName, str] = {}
    for name in rubric.names():
        entry = criteria.get(name)
        if entry is None:
            raise EvalError(f"judge JSON is missing criterion '{name}'")
        raw: Any = entry.get("score") if isinstance(entry, dict) else entry
        try:
            value = round(float(raw))
        except (TypeError, ValueError) as exc:
            raise EvalError(f"criterion '{name}' has a non-numeric score {raw!r}") from exc
        value = max(MIN_SCORE, min(MAX_SCORE, value))
        if rubric.by_name(name).capped_without_challenge:
            value = apply_graceful_handling_cap(value, user_posed_challenge=user_posed_challenge)
        scores[name] = value
        if isinstance(entry, dict) and entry.get("justification"):
            justifications[name] = str(entry["justification"])
    return JudgeVerdict(
        scores=scores,
        justifications=justifications,
        user_posed_challenge=user_posed_challenge,
    )


def _transcript_text(transcript: Sequence[Turn]) -> str:
    """Render a transcript as ``Customer:``/``Agent:`` lines for the judge."""
    label = {"user": "Customer", "assistant": "Agent"}
    return "\n".join(f"{label[t.role]}: {t.content}" for t in transcript)


def build_judge_prompt(
    transcript: Sequence[Turn],
    scenario: Scenario,
    *,
    rubric: Rubric = RUBRIC,
    procedure_description: str = "",
) -> str:
    """Build the user-message prompt for one judging call.

    Args:
        transcript: The completed conversation to score.
        scenario: The scenario context the user was grounded in (gives the judge
            ground truth to check Information Accuracy against).
        rubric: The rubric being applied.
        procedure_description: Optional one-line description of the intended
            procedure, so the judge can assess Task Success.

    Returns:
        The judge user-message text.
    """
    facts = "\n".join(f"- {k}: {v}" for k, v in scenario.items()) or "(none)"
    schema_names = ", ".join(f'"{n}"' for n in rubric.names())
    return (
        (f"Intended procedure: {procedure_description}\n\n" if procedure_description else "")
        + "Ground-truth scenario the customer was given (use this to check that the "
        f"agent used and retained information correctly):\n{facts}\n\n"
        f"Conversation to evaluate:\n{_transcript_text(transcript)}\n\n"
        "Score the AGENT's performance. First decide whether the customer posed any "
        "challenge, change, ambiguity, or edge case. Then score each criterion. "
        "Respond with ONLY a JSON object of the form:\n"
        '{"user_posed_challenge": <true|false>, "criteria": {'
        + ", ".join(f'{n}: {{"score": <1-5>, "justification": "..."}}' for n in (schema_names,))
        + "}}\n"
        "Include all of these criteria keys: " + schema_names + "."
    )


class Judge:
    """LLM-as-judge that scores conversations against the rubric.

    Example:
        >>> judge = Judge(JudgeConfig())  # doctest: +SKIP
        >>> verdict = await judge.score(transcript, scenario)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        *,
        rubric: Rubric = RUBRIC,
        client: Any | None = None,
        cost: CostTracker | None = None,
    ) -> None:
        """Initialise the judge.

        Args:
            config: Judge configuration. Defaults to the Anthropic Claude judge.
            rubric: The rubric to apply.
            client: Optional pre-built client (an ``AsyncAnthropic`` for the
                ``anthropic`` backend, or an async OpenAI client for ``openai``).
                Tests inject a mock; when omitted the right client is created.
            cost: Optional shared cost tracker.

        Raises:
            EvalError: If the ``openai`` backend is selected but the ``openai``
                package is not installed (only when a client must be created).
        """
        self.config = config or JudgeConfig()
        self.rubric = rubric
        self.cost = cost if cost is not None else CostTracker(model=self.config.model)
        self.system_prompt = (
            "You are a meticulous, impartial evaluator of customer-support conversations. "
            "You judge ONLY the agent's performance against a fixed rubric.\n\n"
            + rubric.judge_prompt_block()
        )
        self._client = client if client is not None else self._make_client()

    def _make_client(self) -> Any:
        if self.config.backend == "anthropic":
            return AsyncAnthropic()
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EvalError(
                "The 'openai' judge backend needs the optional openai package. "
                "Install it with `pip install -e '.[openai]'`, or use the default "
                "anthropic backend."
            ) from exc
        return AsyncOpenAI()

    async def score(
        self,
        transcript: Sequence[Turn],
        scenario: Scenario,
        *,
        procedure_description: str = "",
    ) -> JudgeVerdict:
        """Score one conversation and return the rule-corrected verdict.

        Args:
            transcript: The completed conversation.
            scenario: The scenario the user was grounded in.
            procedure_description: Optional procedure description for Task Success.

        Returns:
            The parsed, capped, clamped :class:`JudgeVerdict`.
        """
        prompt = build_judge_prompt(
            transcript,
            scenario,
            rubric=self.rubric,
            procedure_description=procedure_description,
        )
        if self.config.backend == "anthropic":
            raw = await self._call_anthropic(prompt)
        else:
            raw = await self._call_openai(prompt)
        return parse_judge_json(raw, rubric=self.rubric)

    async def _call_anthropic(self, prompt: str) -> str:
        message: Message = await self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        self.cost.add_usage(message.usage)
        return "".join(b.text for b in message.content if isinstance(b, TextBlock)).strip()

    async def _call_openai(self, prompt: str) -> str:  # pragma: no cover - optional backend
        response = await self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.cost.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.cost.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.cost.api_calls += 1
        return response.choices[0].message.content or ""
