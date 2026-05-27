"""Unit tests for the pure statistics functions in the eval runner.

These are network-free and deterministic; they validate the statistical
machinery the eval report depends on against SciPy ground truth and hand-worked
examples.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from subterranean.eval.runner import (
    bootstrap_ci,
    failure_rate,
    holm_bonferroni,
    paired_test,
    unpaired_test,
)

# --------------------------------------------------------------------------- #
# bootstrap_ci                                                                 #
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_deterministic_under_seed() -> None:
    samples = [4, 5, 4, 3, 5, 4, 4, 5]
    a = bootstrap_ci(samples, seed=123)
    b = bootstrap_ci(samples, seed=123)
    assert a == b


def test_bootstrap_ci_brackets_the_mean() -> None:
    samples = [4, 4, 5, 5, 4, 3, 5, 4, 4, 5]
    mean = float(np.mean(samples))
    lo, hi = bootstrap_ci(samples, seed=0)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_wider_for_more_variance() -> None:
    tight = bootstrap_ci([4, 4, 4, 4, 4, 4, 4, 4], seed=0)
    spread = bootstrap_ci([1, 5, 1, 5, 1, 5, 1, 5], seed=0)
    assert (tight[1] - tight[0]) < (spread[1] - spread[0])


def test_bootstrap_ci_single_sample_is_degenerate() -> None:
    assert bootstrap_ci([4.0], seed=0) == (4.0, 4.0)


def test_bootstrap_ci_empty_is_nan() -> None:
    lo, hi = bootstrap_ci([], seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_resample_count_respected() -> None:
    # A different resample count changes the estimate but stays valid/bracketing.
    samples = [3, 4, 5, 4, 3, 5, 4]
    lo, hi = bootstrap_ci(samples, resamples=2000, seed=7)
    assert lo <= float(np.mean(samples)) <= hi


def test_bootstrap_ci_approximates_normal_for_large_n() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(4.0, 0.5, size=500).tolist()
    lo, hi = bootstrap_ci(data, seed=1)
    # The true mean (4.0) should sit inside a 95% interval.
    assert lo <= 4.0 <= hi


# --------------------------------------------------------------------------- #
# failure_rate                                                                 #
# --------------------------------------------------------------------------- #


def test_failure_rate_counts_le_three() -> None:
    # 3 and below are failures: scores 3, 2, 1 -> 3 of 5.
    assert failure_rate([5, 4, 3, 2, 1]) == pytest.approx(0.6)


def test_failure_rate_boundary_three_is_failure() -> None:
    assert failure_rate([3, 3, 3]) == 1.0
    assert failure_rate([4, 4, 4]) == 0.0


def test_failure_rate_empty_is_zero() -> None:
    assert failure_rate([]) == 0.0


def test_failure_rate_custom_threshold() -> None:
    assert failure_rate([5, 4, 3, 2], threshold=2) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Holm-Bonferroni                                                              #
# --------------------------------------------------------------------------- #


def test_holm_bonferroni_hand_worked_example() -> None:
    # m=3, alpha=0.05. Sorted: 0.01 (<=0.0167 ok), 0.03 (<=0.025? no -> stop).
    # Step-down: 0.04 is also not rejected. Original order [0.01, 0.04, 0.03].
    assert holm_bonferroni([0.01, 0.04, 0.03]) == [True, False, False]


def test_holm_bonferroni_all_reject() -> None:
    # Sorted 0.001 (<=0.0125), 0.002 (<=0.0167), 0.003 (<=0.025), 0.004 (<=0.05).
    assert holm_bonferroni([0.001, 0.002, 0.003, 0.004]) == [True, True, True, True]


def test_holm_bonferroni_step_down_stops_after_first_failure() -> None:
    # m=5, alpha=0.05. Smallest is 0.2 > 0.01 -> nothing rejected.
    assert holm_bonferroni([0.2, 0.3, 0.4, 0.5, 0.6]) == [False] * 5


def test_holm_bonferroni_preserves_input_order() -> None:
    # The single tiny p-value is at index 2; only it is rejected.
    result = holm_bonferroni([0.9, 0.9, 0.0001, 0.9, 0.9])
    assert result == [False, False, True, False, False]


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_matches_manual_thresholds() -> None:
    # Validate the step-down thresholds explicitly: alpha/(m-k).
    pvals = [0.012, 0.02, 0.04, 0.5, 0.6]
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    expected = [False] * m
    rejecting = True
    for rank, idx in enumerate(order):
        thr = 0.05 / (m - rank)
        if rejecting and pvals[idx] <= thr:
            expected[idx] = True
        else:
            rejecting = False
    assert holm_bonferroni(pvals) == expected


# --------------------------------------------------------------------------- #
# Wilcoxon / Mann-Whitney wrappers                                             #
# --------------------------------------------------------------------------- #


def test_paired_test_matches_scipy() -> None:
    a = [4, 5, 4, 3, 5, 4, 2, 5]
    b = [3, 4, 4, 2, 4, 3, 1, 4]
    expected = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    assert paired_test(a, b) == pytest.approx(expected)


def test_paired_test_identical_returns_one() -> None:
    # All differences zero -> no evidence of difference -> p = 1.0.
    assert paired_test([4, 4, 4], [4, 4, 4]) == 1.0


def test_paired_test_detects_clear_difference() -> None:
    a = [5] * 10
    b = [2] * 9 + [3]
    assert paired_test(a, b) < 0.05


def test_paired_test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_test([1, 2, 3], [1, 2])


def test_unpaired_test_matches_scipy() -> None:
    a = [5, 5, 4, 5, 4, 5]
    b = [3, 2, 3, 4, 3, 2, 1]
    expected = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    assert unpaired_test(a, b) == pytest.approx(expected)


def test_unpaired_test_detects_difference() -> None:
    assert unpaired_test([5, 5, 5, 5, 5], [1, 1, 1, 1, 1]) < 0.05
