"""Typed exceptions for the agent2model library.

Errors are raised as exceptions, never returned as values. Each public failure
mode has a dedicated type so callers (and the CLI) can present actionable messages.
"""

from __future__ import annotations


class agent2modelError(Exception):
    """Base class for all agent2model errors."""


class FlowchartValidationError(agent2modelError):
    """Raised when a flowchart violates an IR invariant.

    The message is intended to be shown directly to a user, so it should name the
    offending node/edge and explain the broken invariant in plain language.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or [message]
        super().__init__(message)


class GenerationBudgetExceeded(agent2modelError):
    """Raised when synthetic data generation would exceed the user's ``--budget``.

    Attributes:
        spent_usd: Amount already spent when the cap was hit.
        budget_usd: The configured hard cap.
    """

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Generation budget exceeded: spent ${spent_usd:.2f} of ${budget_usd:.2f} cap."
        )


class TrainingDivergedError(agent2modelError):
    """Raised when fine-tuning diverges (NaN/Inf loss or runaway gradient)."""


class ServingError(agent2modelError):
    """Raised when a compiled model cannot be served.

    Covers both "nothing servable was found in the build directory" and "the
    optional serving stack (vLLM) is not installed on this host". The message is
    written to be shown directly to a user, so it should explain what was missing
    and what to do next.
    """


class EvalBudgetExceeded(agent2modelError):
    """Raised when an evaluation run would exceed the user's ``--budget``.

    The eval harness makes many LLM calls (the user simulator, the
    model-under-test for each condition, and the judge), so it carries the same
    USD hard-stop semantics as data generation.

    Attributes:
        spent_usd: Amount already spent when the cap was hit.
        budget_usd: The configured hard cap.
    """

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Evaluation budget exceeded: spent ${spent_usd:.2f} of ${budget_usd:.2f} cap."
        )


class EvalError(agent2modelError):
    """Raised for non-budget evaluation failures.

    Covers an unknown baseline name, a missing optional dependency needed to
    render the PDF report, or a condition that requires an external service
    (a served model / a LangGraph install) that is unavailable. The message is
    written to be shown directly to a user.
    """
