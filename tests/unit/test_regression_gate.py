"""Unit tests for the e2e regression gate's pure helpers.

These exercise the >5% regression logic without any GPU/model/network, so the
release gate's math is always covered even though the full e2e tier cannot run
on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.regression import (
    assert_no_regression,
    find_regressions,
    load_targets,
    observed_from_report,
    within_tolerance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS_PATH = REPO_ROOT / "benchmarks" / "targets.json"


# --- within_tolerance -------------------------------------------------------


def test_within_tolerance_exact_target_passes() -> None:
    assert within_tolerance(4.11, 4.11)


def test_within_tolerance_above_target_passes() -> None:
    assert within_tolerance(4.9, 4.11)


def test_within_tolerance_just_inside_5pct_passes() -> None:
    # 5% below 4.0 is exactly 3.8; the boundary is inclusive.
    assert within_tolerance(3.8, 4.0, tol=0.05)


def test_within_tolerance_just_outside_5pct_fails() -> None:
    assert not within_tolerance(3.79, 4.0, tol=0.05)


def test_within_tolerance_rejects_nonpositive_target() -> None:
    with pytest.raises(ValueError, match="positive"):
        within_tolerance(1.0, 0.0)


# --- find_regressions / assert_no_regression --------------------------------


def test_no_regression_when_all_pass() -> None:
    observed = {"Task Success": 4.2, "Naturalness": 4.0}
    targets = {"Task Success": 4.11, "Naturalness": 4.12}
    assert find_regressions(observed, targets) == []
    assert_no_regression(observed, targets)  # must not raise


def test_detects_single_regression() -> None:
    observed = {"Task Success": 3.0, "Naturalness": 4.1}
    targets = {"Task Success": 4.11, "Naturalness": 4.12}
    regressions = find_regressions(observed, targets)
    assert [r.criterion for r in regressions] == ["Task Success"]
    assert regressions[0].drop_pct > 5.0


def test_missing_observation_is_a_regression() -> None:
    regressions = find_regressions({}, {"Task Success": 4.11})
    assert len(regressions) == 1
    assert regressions[0].criterion == "Task Success"


def test_assert_no_regression_raises_with_listing() -> None:
    observed = {"Task Success": 2.0}
    targets = {"Task Success": 4.11}
    with pytest.raises(AssertionError, match="Task Success"):
        assert_no_regression(observed, targets)


def test_extra_observed_criteria_are_ignored() -> None:
    observed = {"Task Success": 4.2, "Unrelated": 0.0}
    targets = {"Task Success": 4.11}
    assert_no_regression(observed, targets)  # must not raise


# --- targets.json + report parsing ------------------------------------------


@pytest.mark.parametrize("example", ["travel_booking", "zoom_support", "insurance_claims"])
def test_load_targets_has_five_criteria(example: str) -> None:
    targets = load_targets(example, TARGETS_PATH)
    assert set(targets) == {
        "Task Success",
        "Information Accuracy",
        "Consistency",
        "Graceful Handling",
        "Naturalness",
    }
    assert all(1.0 <= v <= 5.0 for v in targets.values())


def test_load_targets_unknown_example_raises() -> None:
    with pytest.raises(KeyError, match="No targets"):
        load_targets("nope", TARGETS_PATH)


def test_paper_targets_pass_their_own_gate() -> None:
    # Sanity: the published numbers trivially satisfy the gate against themselves.
    targets = load_targets("travel_booking", TARGETS_PATH)
    assert_no_regression(dict(targets), targets)


def test_observed_from_report_extracts_condition_means() -> None:
    report = {
        "conditions": [
            {
                "condition": "in_context",
                "criterion_stats": [{"criterion": "Task Success", "mean": 4.6}],
            },
            {
                "condition": "compiled",
                "criterion_stats": [
                    {"criterion": "Task Success", "mean": 4.2},
                    {"criterion": "Naturalness", "mean": 4.1},
                ],
            },
        ]
    }
    observed = observed_from_report(report, condition="compiled")
    assert observed == {"Task Success": 4.2, "Naturalness": 4.1}


def test_observed_from_report_missing_condition_raises() -> None:
    report = {"conditions": [{"condition": "in_context", "criterion_stats": []}]}
    with pytest.raises(KeyError, match="compiled"):
        observed_from_report(report, condition="compiled")
