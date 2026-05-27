"""Evaluation runner: orchestrate n scenarios per condition and compute statistics.

The runner samples ``n`` scenarios (one user simulator each), runs every
condition against them concurrently with a Rich progress bar and a USD budget
guard, judges each conversation against the rubric, and aggregates the exact
statistics the spec demands.

All statistics live in **pure functions** — :func:`bootstrap_ci`,
:func:`paired_test`, :func:`unpaired_test`, :func:`holm_bonferroni`,
:func:`failure_rate` — that take plain arrays/lists of scores and call
SciPy/NumPy (never a hand-rolled implementation). They are unit-tested with no
API calls. The async orchestration (:class:`EvalRunner`) leans on them.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from scipy import stats

from subterranean.eval.judge import Judge, JudgeConfig, JudgeVerdict
from subterranean.eval.rubric import RUBRIC, CriterionName, Rubric
from subterranean.eval.simulator import UserSimulator
from subterranean.exceptions import EvalBudgetExceeded
from subterranean.generation.generator import DEFAULT_MODEL, CostTracker
from subterranean.generation.scenarios import sample_scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic import AsyncAnthropic

    from subterranean.eval.baselines import Condition
    from subterranean.generation.formatter import Conversation
    from subterranean.generation.scenarios import Scenario
    from subterranean.ir.schema import Flowchart

__all__ = [
    "ConditionResult",
    "CriterionStats",
    "EvalRunResult",
    "EvalRunner",
    "PairwiseComparison",
    "bootstrap_ci",
    "failure_rate",
    "holm_bonferroni",
    "paired_test",
    "summarize_condition",
    "unpaired_test",
]

TASK_SUCCESS: CriterionName = "Task Success"
FAILURE_THRESHOLD = 3
"""A conversation is a *failure* when Task Success is at or below this score."""


# --------------------------------------------------------------------------- #
# Pure statistics — unit-tested without any API calls                          #
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Resamples ``samples`` with replacement ``resamples`` times, takes the mean of
    each resample, and returns the central ``confidence`` percentile interval.
    Deterministic for a fixed ``seed``.

    Args:
        samples: Observed scores.
        confidence: Central mass of the interval (default 0.95).
        resamples: Number of bootstrap resamples (default 10,000, per the spec).
        seed: RNG seed for reproducibility.

    Returns:
        A ``(lo, hi)`` tuple bracketing the mean. For a single sample both bounds
        equal that value; for an empty input both are NaN.

    Example:
        >>> lo, hi = bootstrap_ci([4, 4, 5, 5, 4], seed=0)
        >>> lo <= 4.4 <= hi
        True
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def failure_rate(
    task_success_scores: Sequence[float], *, threshold: int = FAILURE_THRESHOLD
) -> float:
    """Fraction of conversations whose Task Success score is at or below ``threshold``.

    Args:
        task_success_scores: Task Success scores, one per conversation.
        threshold: Inclusive failure cutoff (default 3 — i.e. ``<= 3`` fails).

    Returns:
        The failure rate in ``[0, 1]``; ``0.0`` for an empty input.

    Example:
        >>> failure_rate([5, 4, 3, 2, 5])
        0.4
    """
    arr = np.asarray(task_success_scores, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr <= threshold))


def paired_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank test p-value for paired samples.

    Used when the two conditions were run on the *same* scenarios (the usual
    case), so each conversation has a matched pair of scores.

    Args:
        a: Scores from condition A.
        b: Scores from condition B (same length as ``a``).

    Returns:
        The two-sided p-value. Returns ``1.0`` when all paired differences are
        zero (Wilcoxon is undefined there — there is no evidence of a difference).

    Raises:
        ValueError: If the inputs differ in length.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired_test requires equal-length samples")
    if np.allclose(arr_a, arr_b):
        return 1.0
    result = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox", alternative="two-sided")
    return float(result.pvalue)


def unpaired_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U test p-value for unpaired samples.

    Used when the two conditions were run on *different* scenarios.

    Args:
        a: Scores from condition A.
        b: Scores from condition B (lengths may differ).

    Returns:
        The two-sided p-value.
    """
    result = stats.mannwhitneyu(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        alternative="two-sided",
    )
    return float(result.pvalue)


def holm_bonferroni(pvalues: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction across a family of tests.

    Sorts the p-values ascending; the ``k``-th smallest (0-indexed) is compared
    against ``alpha / (m - k)``. The first failure stops all subsequent
    rejections (step-down). Returns the reject/keep decision in the **original**
    order.

    Args:
        pvalues: The family of raw p-values (e.g. one per criterion).
        alpha: Family-wise error rate (default 0.05).

    Returns:
        A list of booleans, ``True`` where the null is rejected (significant),
        aligned to the input order.

    Example:
        >>> holm_bonferroni([0.01, 0.04, 0.03])
        [True, False, True]
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    still_rejecting = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if still_rejecting and pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            still_rejecting = False
    return reject


# --------------------------------------------------------------------------- #
# Result data models                                                           #
# --------------------------------------------------------------------------- #


class CriterionStats(BaseModel):
    """Aggregated statistics for one criterion within one condition.

    Attributes:
        criterion: The criterion name.
        mean: Mean score across conversations.
        ci_low: Lower 95% bootstrap CI bound.
        ci_high: Upper 95% bootstrap CI bound.
        n: Number of scores aggregated.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: CriterionName
    mean: float
    ci_low: float
    ci_high: float
    n: int


class ConditionResult(BaseModel):
    """All results for one condition across the scenario suite.

    Attributes:
        condition: The condition name (e.g. ``in_context``).
        n_conversations: Number of conversations completed.
        scores: Raw per-criterion score lists (criterion name -> list of scores).
        criterion_stats: Aggregated stats per criterion.
        failure_rate: Fraction of conversations with Task Success <= 3.
        cost_usd: Total API cost attributed to this condition (USD).
        avg_wall_clock_s: Mean wall-clock seconds per conversation.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    n_conversations: int
    scores: dict[CriterionName, list[int]]
    criterion_stats: list[CriterionStats]
    failure_rate: float
    cost_usd: float
    avg_wall_clock_s: float


class PairwiseComparison(BaseModel):
    """One pairwise comparison between two conditions across criteria.

    Attributes:
        condition_a: First condition name.
        condition_b: Second condition name.
        paired: Whether a paired (Wilcoxon) test was used.
        pvalues: Raw per-criterion p-values.
        significant: Holm-Bonferroni-corrected significance per criterion.
    """

    model_config = ConfigDict(extra="forbid")

    condition_a: str
    condition_b: str
    paired: bool
    pvalues: dict[CriterionName, float]
    significant: dict[CriterionName, bool]


class EvalRunResult(BaseModel):
    """The complete result of an evaluation run, ready for the report/JSON.

    Attributes:
        flowchart_name: Name of the evaluated procedure.
        n: Scenarios per condition requested.
        conditions: Per-condition aggregated results.
        comparisons: Pairwise comparisons (each baseline vs. compiled when present,
            else all unordered pairs).
        total_cost_usd: Total API cost across all conditions and the judge.
        judge_model: Model id used for judging.
    """

    model_config = ConfigDict(extra="forbid")

    flowchart_name: str
    n: int
    conditions: list[ConditionResult]
    comparisons: list[PairwiseComparison]
    total_cost_usd: float
    judge_model: str = DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Aggregation (pure)                                                           #
# --------------------------------------------------------------------------- #


def summarize_condition(
    condition: str,
    verdicts: Sequence[JudgeVerdict],
    *,
    cost_usd: float,
    wall_clock_s: Sequence[float],
    rubric: Rubric = RUBRIC,
    seed: int = 0,
) -> ConditionResult:
    """Aggregate per-conversation verdicts into a :class:`ConditionResult`.

    Pure: no API calls. Computes per-criterion mean + 95% bootstrap CI, the
    failure rate from Task Success, and mean wall-clock.

    Args:
        condition: Condition name.
        verdicts: Per-conversation judge verdicts.
        cost_usd: Total cost attributed to this condition.
        wall_clock_s: Per-conversation wall-clock seconds.
        rubric: Rubric defining the criteria.
        seed: Bootstrap seed.

    Returns:
        The aggregated :class:`ConditionResult`.
    """
    scores: dict[CriterionName, list[int]] = {
        name: [v.scores[name] for v in verdicts] for name in rubric.names()
    }
    crit_stats: list[CriterionStats] = []
    for name in rubric.names():
        values = scores[name]
        mean = float(np.mean(values)) if values else float("nan")
        lo, hi = bootstrap_ci(values, seed=seed)
        crit_stats.append(
            CriterionStats(criterion=name, mean=mean, ci_low=lo, ci_high=hi, n=len(values))
        )
    return ConditionResult(
        condition=condition,
        n_conversations=len(verdicts),
        scores=scores,
        criterion_stats=crit_stats,
        failure_rate=failure_rate(scores[TASK_SUCCESS]),
        cost_usd=cost_usd,
        avg_wall_clock_s=float(np.mean(wall_clock_s)) if len(wall_clock_s) else 0.0,
    )


def compare_conditions(
    result_a: ConditionResult,
    result_b: ConditionResult,
    *,
    paired: bool,
    rubric: Rubric = RUBRIC,
    alpha: float = 0.05,
) -> PairwiseComparison:
    """Run the per-criterion significance tests between two conditions.

    Uses Wilcoxon (paired) or Mann-Whitney U (unpaired) per criterion, then
    Holm-Bonferroni-corrects the 5 p-values together.

    Args:
        result_a: First condition's results.
        result_b: Second condition's results.
        paired: Whether the conditions share scenarios (paired test) or not.
        rubric: Rubric defining the criteria.
        alpha: Family-wise error rate.

    Returns:
        The :class:`PairwiseComparison`.
    """
    names = rubric.names()
    pvalues: dict[CriterionName, float] = {}
    for name in names:
        a = result_a.scores[name]
        b = result_b.scores[name]
        pvalues[name] = paired_test(a, b) if paired else unpaired_test(a, b)
    corrected = holm_bonferroni([pvalues[n] for n in names], alpha=alpha)
    significant = dict(zip(names, corrected, strict=True))
    return PairwiseComparison(
        condition_a=result_a.condition,
        condition_b=result_b.condition,
        paired=paired,
        pvalues=pvalues,
        significant=significant,
    )


# --------------------------------------------------------------------------- #
# Async orchestration                                                          #
# --------------------------------------------------------------------------- #


class EvalConfig(BaseModel):
    """Configuration for an evaluation run.

    Attributes:
        n: Scenarios per condition.
        budget_usd: Hard USD cap across all LLM calls (simulator + agents + judge).
        seed: Base RNG seed; scenario ``i`` derives a deterministic sub-seed.
        max_concurrent: Maximum concurrent scenario evaluations.
        max_turns: Hard cap on conversation length.
        judge: Judge configuration.
    """

    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=200, gt=0)
    budget_usd: float = Field(default=50.0, gt=0.0)
    seed: int = 0
    max_concurrent: int = Field(default=10, gt=0)
    max_turns: int = Field(default=12, gt=0)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


def estimate_eval_cost(
    config: EvalConfig, n_conditions: int, *, turns_per_convo: float = 6.0
) -> float:
    """Roughly estimate an eval run's cost in USD before it starts.

    Each conversation costs: simulator turns + agent turns (per condition) + one
    judge call. This is a coarse up-front figure for the CLI, not a precise one.

    Args:
        config: The run configuration.
        n_conditions: Number of conditions being evaluated.
        turns_per_convo: Assumed agent turns per conversation.

    Returns:
        Estimated total cost in USD.
    """
    from subterranean.generation.generator import _FALLBACK_PRICING, _PRICING

    rates = _PRICING.get(config.judge.model, _FALLBACK_PRICING)
    per_call_in, per_call_out = 700, 200
    # simulator + agent calls per convo, plus one judge call per convo.
    calls = config.n * n_conditions * (2 * turns_per_convo + 1)
    return float(
        calls * per_call_in * rates["input"] / 1_000_000
        + calls * per_call_out * rates["output"] / 1_000_000
    )


class EvalRunner:
    """Runs ``n`` scenarios per condition, judges them, and aggregates statistics.

    The runner samples scenarios once (so every condition is evaluated on the
    *same* scenarios — enabling paired tests), runs each condition concurrently
    against a fresh user simulator, judges every conversation, and folds all costs
    into a shared :class:`CostTracker` guarded by the budget.

    Example:
        >>> runner = EvalRunner(fc, conditions, config)  # doctest: +SKIP
        >>> result = await runner.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        flowchart: Flowchart,
        conditions: Sequence[Condition],
        config: EvalConfig,
        *,
        judge: Judge | None = None,
        simulator_client: AsyncAnthropic | None = None,
        simulator_model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialise the runner.

        Args:
            flowchart: The compiled flowchart (for scenario sampling + procedure
                description handed to the judge).
            conditions: The conditions to evaluate.
            config: Run configuration.
            judge: Optional pre-built judge (tests inject a mock-backed one).
            simulator_client: Optional shared Anthropic client for simulators.
            simulator_model: Model id for the user simulators.
        """
        self.flowchart = flowchart
        self.conditions = list(conditions)
        self.config = config
        self.judge = judge or Judge(config.judge)
        self.simulator_client = simulator_client
        self.simulator_model = simulator_model
        self.cost = CostTracker(model=config.judge.model)

    def _sample_scenarios(self) -> list[Scenario]:
        import random as _random

        # Derive an independent scenario per index, reproducibly from the seed.
        scenarios: list[Scenario] = []
        for i in range(self.config.n):
            sub = _random.Random(f"{self.config.seed}:{i}")
            scenarios.append(sample_scenario(self.flowchart, sub))
        return scenarios

    def _check_budget(self) -> None:
        if self.cost.cost_usd > self.config.budget_usd:
            raise EvalBudgetExceeded(self.cost.cost_usd, self.config.budget_usd)

    async def _run_one(
        self, condition: Condition, scenario: Scenario
    ) -> tuple[Conversation, JudgeVerdict, float]:
        from subterranean.eval.baselines import ConditionContext

        start = time.perf_counter()
        simulator = UserSimulator(
            scenario,
            model=self.simulator_model,
            client=self.simulator_client,
            cost=self.cost,
        )
        context = ConditionContext(scenario, simulator, max_turns=self.config.max_turns)
        conversation = await condition.run_scenario(context)
        verdict = await self.judge.score(
            conversation.turns,
            scenario,
            procedure_description=self.flowchart.description,
        )
        elapsed = time.perf_counter() - start
        self._check_budget()
        return conversation, verdict, elapsed

    async def _run_condition_suite(
        self,
        condition: Condition,
        scenarios: Sequence[Scenario],
        semaphore: asyncio.Semaphore,
        progress: Progress,
        task: TaskID,
    ) -> tuple[list[JudgeVerdict], list[float]]:
        """Run + judge every scenario for one condition; collect verdicts/timings."""
        verdicts: list[JudgeVerdict] = []
        wall: list[float] = []

        async def worker(scenario: Scenario) -> None:
            async with semaphore:
                _, verdict, elapsed = await self._run_one(condition, scenario)
            verdicts.append(verdict)
            wall.append(elapsed)
            progress.advance(task)

        await asyncio.gather(*(worker(s) for s in scenarios))
        return verdicts, wall

    async def run(self) -> EvalRunResult:
        """Evaluate every condition over the scenario suite and aggregate.

        Returns:
            The full :class:`EvalRunResult` (per-condition stats + pairwise
            comparisons + total cost).

        Raises:
            EvalBudgetExceeded: If the USD budget is hit mid-run.
        """
        scenarios = self._sample_scenarios()
        # The judge shares the runner's cost tracker so the total is accurate.
        self.judge.cost = self.cost

        results: list[ConditionResult] = []
        total = len(self.conditions) * len(scenarios)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Evaluating conditions", total=total)

            for condition in self.conditions:
                cost_before = self.cost.cost_usd
                verdicts, wall = await self._run_condition_suite(
                    condition, scenarios, semaphore, progress, task
                )
                cost_after = self.cost.cost_usd
                results.append(
                    summarize_condition(
                        condition.name,
                        verdicts,
                        cost_usd=cost_after - cost_before,
                        wall_clock_s=wall,
                        seed=self.config.seed,
                    )
                )

        comparisons = self._build_comparisons(results)
        return EvalRunResult(
            flowchart_name=self.flowchart.name,
            n=self.config.n,
            conditions=results,
            comparisons=comparisons,
            total_cost_usd=self.cost.cost_usd,
            judge_model=self.config.judge.model,
        )

    def _build_comparisons(self, results: Sequence[ConditionResult]) -> list[PairwiseComparison]:
        """Compare the compiled condition against each baseline (paired)."""
        by_name = {r.condition: r for r in results}
        comparisons: list[PairwiseComparison] = []
        if "compiled" in by_name:
            base = by_name["compiled"]
            for other in results:
                if other.condition == "compiled":
                    continue
                comparisons.append(compare_conditions(base, other, paired=True))
        else:
            # No compiled model served: compare all unordered pairs.
            ordered = list(results)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    comparisons.append(compare_conditions(ordered[i], ordered[j], paired=True))
        return comparisons
