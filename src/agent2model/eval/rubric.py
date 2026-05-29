"""The paper's 5-criterion evaluation rubric (Dennis et al. 2026, §3).

This module encodes the rubric as a typed Pydantic structure so it can serve two
masters at once:

- as a **prompt-ready string** handed to the LLM judge
  (:func:`Rubric.judge_prompt_block`), and
- as **data** for the report (criterion names, score scale, the rule that
  Graceful Handling is capped at 3 when the user posed no challenges).

The behavioural anchors are ported verbatim from the plan file's Phase 6 section
(which captured them from the paper's §3); they are not paraphrased from memory.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CRITERIA",
    "GRACEFUL_HANDLING",
    "MAX_SCORE",
    "MIN_SCORE",
    "RUBRIC",
    "Criterion",
    "CriterionName",
    "Rubric",
    "apply_graceful_handling_cap",
]

MIN_SCORE = 1
"""Lowest score on every criterion."""

MAX_SCORE = 5
"""Highest score on every criterion."""

GRACEFUL_HANDLING = "Graceful Handling"
"""Name of the one criterion subject to the no-challenges cap."""

GRACEFUL_HANDLING_CAP = 3
"""Maximum Graceful Handling score when the user posed no challenges.

A conversation with a perfectly cooperative user gives the agent no opportunity
to demonstrate graceful handling, so the paper caps the achievable score at 3.
"""

CriterionName = Literal[
    "Task Success",
    "Information Accuracy",
    "Consistency",
    "Graceful Handling",
    "Naturalness",
]
"""The five rubric criterion names, in the paper's reporting order."""


class Criterion(BaseModel):
    """One scored rubric criterion with its behavioural anchors.

    Attributes:
        name: The criterion's display name (one of :data:`CriterionName`).
        question: The single evaluative question the judge answers.
        anchors: Behavioural-anchor text keyed by score level (``1``..``5``). Not
            every level is spelled out in the paper; the anchors that are given
            are stored verbatim and the judge interpolates between them.
        capped_without_challenge: When True, the criterion's score is capped at
            :data:`GRACEFUL_HANDLING_CAP` if the user posed no challenges.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: CriterionName
    question: str
    anchors: dict[int, str]
    capped_without_challenge: bool = False

    def anchor_text(self) -> str:
        """Render this criterion's anchors as ``5 = ...`` lines for a prompt."""
        return "\n".join(f"  {level} = {self.anchors[level]}" for level in sorted(self.anchors))


# The five criteria, anchors verbatim from the plan's Phase 6 section.
CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        name="Task Success",
        question=(
            "Did the agent execute the procedure correctly through to an appropriate "
            "terminal state, with consistent and accurate handling at each decision point?"
        ),
        anchors={
            5: "complete procedure with a clear terminal state",
            3: "middle stages done but the conversation fizzled",
            1: "no meaningful progress",
        },
    ),
    Criterion(
        name="Information Accuracy",
        question="Did the agent correctly use and retain all user-provided information?",
        anchors={
            5: "every detail correctly reflected",
            1: "fabricated or ignored input",
        },
    ),
    Criterion(
        name="Consistency",
        question="Did the agent maintain coherent state across the conversation?",
        anchors={
            5: "no contradictions or repeated questions",
            1: "contradicts itself repeatedly",
        },
    ),
    Criterion(
        name="Graceful Handling",
        question="How well did the agent handle changes, ambiguity, and edge cases?",
        anchors={
            5: "smoothly adapts",
            1: "any deviation breaks the flow",
        },
        capped_without_challenge=True,
    ),
    Criterion(
        name="Naturalness",
        question="Does the conversation read like talking to a skilled human agent?",
        anchors={
            5: "indistinguishable from a human",
            1: "mechanical, scripted",
        },
    ),
)


class Rubric(BaseModel):
    """The full 5-criterion rubric, usable as both prompt text and report data.

    Example:
        >>> RUBRIC.names()
        ('Task Success', 'Information Accuracy', 'Consistency', 'Graceful Handling', \
'Naturalness')
        >>> "1-5" in RUBRIC.judge_prompt_block()
        True
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria: tuple[Criterion, ...] = Field(default=CRITERIA)
    min_score: int = MIN_SCORE
    max_score: int = MAX_SCORE

    def names(self) -> tuple[CriterionName, ...]:
        """Return the criterion names in the paper's reporting order."""
        return tuple(c.name for c in self.criteria)

    def by_name(self, name: str) -> Criterion:
        """Look up a criterion by name.

        Args:
            name: A criterion name.

        Returns:
            The matching :class:`Criterion`.

        Raises:
            KeyError: If no criterion has that name.
        """
        for criterion in self.criteria:
            if criterion.name == name:
                return criterion
        raise KeyError(name)

    def judge_prompt_block(self) -> str:
        """Render the rubric as a self-contained block for the judge's prompt.

        The block lists each criterion's question and behavioural anchors and
        states the score scale and the Graceful-Handling cap rule, so the judge
        has the full rubric in-context.

        Returns:
            A multi-line string ready to embed in the judge system prompt.
        """
        lines = [
            f"Score each criterion on an integer scale of {self.min_score}-{self.max_score} "
            f"({self.min_score} = worst, {self.max_score} = best).",
            "",
        ]
        for idx, criterion in enumerate(self.criteria, start=1):
            lines.append(f"{idx}. {criterion.name}: {criterion.question}")
            lines.append(criterion.anchor_text())
            if criterion.capped_without_challenge:
                lines.append(
                    f"  NOTE: cap this score at {GRACEFUL_HANDLING_CAP} if the user posed "
                    "no challenges, changes, ambiguity, or edge cases — a cooperative user "
                    "gives no opportunity to demonstrate graceful handling."
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


RUBRIC = Rubric()
"""The canonical rubric instance used across the eval harness."""


def apply_graceful_handling_cap(score: int, *, user_posed_challenge: bool) -> int:
    """Apply the Graceful-Handling cap to a raw judge score.

    Args:
        score: The judge's raw Graceful Handling score (``1``..``5``).
        user_posed_challenge: Whether the user posed any challenge, change,
            ambiguity, or edge case in the conversation.

    Returns:
        ``score`` unchanged when the user posed a challenge; otherwise the score
        clamped to at most :data:`GRACEFUL_HANDLING_CAP`.

    Example:
        >>> apply_graceful_handling_cap(5, user_posed_challenge=False)
        3
        >>> apply_graceful_handling_cap(5, user_posed_challenge=True)
        5
        >>> apply_graceful_handling_cap(2, user_posed_challenge=False)
        2
    """
    if user_posed_challenge:
        return score
    return min(score, GRACEFUL_HANDLING_CAP)
