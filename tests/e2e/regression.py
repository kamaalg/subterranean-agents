"""Pure regression-gate helpers for the e2e reproduction tier.

These functions are deliberately free of I/O, network, and GPU so the >5%
regression logic can be unit-tested without running the full pipeline. They
compare measured per-criterion means against the paper targets in
``benchmarks/targets.json``.

The gate is one-sided: a criterion only *fails* when it drops more than the
tolerance below its target. Scoring above the paper is never a regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOLERANCE = 0.05
"""Default relative tolerance: a criterion may not drop >5% below its target."""


def within_tolerance(observed: float, target: float, tol: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether an observed score is within tolerance of its target.

    The check is one-sided on the downside: ``observed`` passes if it is at most
    ``tol`` (as a fraction of ``target``) below ``target``. Meeting or exceeding
    the target always passes.

    Args:
        observed: The measured criterion mean.
        target: The paper's target criterion mean (assumed positive).
        tol: Maximum allowed relative drop below ``target`` (default 0.05 = 5%).

    Returns:
        ``True`` if ``observed >= target * (1 - tol)``, else ``False``.

    Example:
        >>> within_tolerance(4.0, 4.11, tol=0.05)
        True
        >>> within_tolerance(3.5, 4.11, tol=0.05)
        False
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target!r}")
    return observed >= target * (1.0 - tol)


@dataclass(frozen=True)
class Regression:
    """A single criterion that regressed beyond tolerance.

    Attributes:
        criterion: The criterion name.
        observed: The measured mean.
        target: The paper target mean.
        tol: The tolerance applied.
    """

    criterion: str
    observed: float
    target: float
    tol: float

    @property
    def drop_pct(self) -> float:
        """Percentage drop below the target (positive number)."""
        return (self.target - self.observed) / self.target * 100.0

    def __str__(self) -> str:
        return (
            f"{self.criterion}: observed {self.observed:.3f} vs target "
            f"{self.target:.3f} (down {self.drop_pct:.1f}%, > {self.tol:.0%} gate)"
        )


def find_regressions(
    observed: dict[str, float],
    targets: dict[str, float],
    tol: float = DEFAULT_TOLERANCE,
) -> list[Regression]:
    """Return every criterion whose observed mean regressed beyond tolerance.

    Only criteria present in ``targets`` are checked; a target with no observed
    value is reported as a regression (a missing measurement cannot clear the
    gate).

    Args:
        observed: Measured criterion means keyed by criterion name.
        targets: Paper target means keyed by criterion name.
        tol: Maximum allowed relative drop (default 0.05).

    Returns:
        A list of :class:`Regression` records, empty if all criteria pass.
    """
    regressions: list[Regression] = []
    for criterion, target in targets.items():
        if criterion not in observed:
            regressions.append(Regression(criterion, float("nan"), target, tol))
            continue
        value = observed[criterion]
        if not within_tolerance(value, target, tol):
            regressions.append(Regression(criterion, value, target, tol))
    return regressions


def assert_no_regression(
    observed: dict[str, float],
    targets: dict[str, float],
    tol: float = DEFAULT_TOLERANCE,
) -> None:
    """Raise ``AssertionError`` if any criterion regressed beyond tolerance.

    Args:
        observed: Measured criterion means keyed by criterion name.
        targets: Paper target means keyed by criterion name.
        tol: Maximum allowed relative drop (default 0.05).

    Raises:
        AssertionError: If one or more criteria dropped more than ``tol`` below
            target. The message lists every offending criterion.

    Example:
        >>> assert_no_regression({"Task Success": 4.2}, {"Task Success": 4.11})
    """
    regressions = find_regressions(observed, targets, tol)
    if regressions:
        lines = "\n  - ".join(str(r) for r in regressions)
        raise AssertionError(f"Regression beyond {tol:.0%} tolerance:\n  - {lines}")


def load_targets(example: str, targets_path: Path) -> dict[str, float]:
    """Load the per-criterion target means for one example from ``targets.json``.

    Args:
        example: Example key (e.g. ``"travel_booking"``).
        targets_path: Path to ``benchmarks/targets.json``.

    Returns:
        Mapping of criterion name to target mean.

    Raises:
        KeyError: If ``example`` is not present in the targets file.
    """
    data = json.loads(targets_path.read_text(encoding="utf-8"))
    examples = data["examples"]
    if example not in examples:
        raise KeyError(f"No targets for example {example!r}; have {sorted(examples)}")
    criteria = examples[example]["criteria"]
    return {name: float(value) for name, value in criteria.items()}


def observed_from_report(
    report: dict[str, object], condition: str = "compiled"
) -> dict[str, float]:
    """Extract per-criterion means for one condition from an eval-report dict.

    Parses the structure written by
    :func:`subterranean.eval.report.write_json_report` (an
    :class:`~subterranean.eval.runner.EvalRunResult` dumped to JSON).

    Args:
        report: The parsed ``eval_report.json`` contents.
        condition: Condition name to read (the compiled model is ``"compiled"``).

    Returns:
        Mapping of criterion name to measured mean.

    Raises:
        KeyError: If ``condition`` is not present in the report.
    """
    conditions = report["conditions"]
    assert isinstance(conditions, list)
    for cond in conditions:
        assert isinstance(cond, dict)
        if cond.get("condition") == condition:
            stats = cond["criterion_stats"]
            assert isinstance(stats, list)
            return {str(cs["criterion"]): float(cs["mean"]) for cs in stats}
    available = [c.get("condition") for c in conditions if isinstance(c, dict)]
    raise KeyError(f"Condition {condition!r} not in report; have {available}")
