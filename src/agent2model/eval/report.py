"""Render an :class:`~agent2model.eval.runner.EvalRunResult` to PDF and JSON.

The PDF is README-quality: per-criterion grouped bar charts with 95%-CI error
bars (conditions side by side), a failure-rate chart, and a cost-per-conversation
breakdown. matplotlib is imported **lazily** so the rest of the eval package (and
the unit tests) work without the optional ``[report]`` extra installed; if it is
missing, :func:`write_pdf_report` raises a clear, actionable
:class:`~agent2model.exceptions.EvalError`.

The structured :class:`~agent2model.eval.runner.EvalRunResult` is also dumped as
JSON (:func:`write_json_report`) so the numbers are machine-readable and
comparable to the paper's Table 1-3 targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent2model.eval.rubric import RUBRIC, CriterionName
from agent2model.exceptions import EvalError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent2model.eval.runner import ConditionResult, EvalRunResult

__all__ = ["write_json_report", "write_pdf_report"]

_MISSING_MATPLOTLIB = (
    "PDF report generation needs the optional matplotlib dependency, which is not "
    "installed. Install it with `pip install -e '.[report]'`. The JSON report "
    "(eval_report.json) was still written."
)


def write_json_report(result: EvalRunResult, path: str | Path) -> Path:
    """Write the run result to JSON.

    Args:
        result: The completed evaluation result.
        path: Destination ``.json`` file.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def _require_matplotlib() -> Any:
    """Import matplotlib lazily, raising an actionable error if it is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend; no display needed.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EvalError(_MISSING_MATPLOTLIB) from exc
    return plt


def write_pdf_report(result: EvalRunResult, path: str | Path) -> Path:
    """Render the evaluation result to a multi-page PDF.

    Pages: (1) per-criterion grouped bars with 95% CI error bars, (2) failure
    rate per condition, (3) cost per conversation per condition.

    Args:
        result: The completed evaluation result.
        path: Destination ``.pdf`` file.

    Returns:
        The written path.

    Raises:
        EvalError: If matplotlib (the ``[report]`` extra) is not installed.

    Example:
        >>> write_pdf_report(result, "build/travel/eval_report.pdf")  # doctest: +SKIP
        PosixPath('build/travel/eval_report.pdf')
    """
    plt = _require_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = list(RUBRIC.names())
    conditions = result.conditions

    with PdfPages(out) as pdf:
        _criteria_page(plt, pdf, result, names, conditions)
        _failure_page(plt, pdf, conditions)
        _cost_page(plt, pdf, conditions)
    return out


def _criteria_page(
    plt: Any,
    pdf: Any,
    result: EvalRunResult,
    names: Sequence[CriterionName],
    conditions: Sequence[ConditionResult],
) -> None:
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    width = 0.8 / max(1, len(conditions))
    for i, cond in enumerate(conditions):
        stats_by_name = {cs.criterion: cs for cs in cond.criterion_stats}
        means = [stats_by_name[n].mean for n in names]
        lows = [stats_by_name[n].mean - stats_by_name[n].ci_low for n in names]
        highs = [stats_by_name[n].ci_high - stats_by_name[n].mean for n in names]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=[lows, highs],
            capsize=3,
            label=cond.condition,
        )
    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels(list(names), rotation=20, ha="right")
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean score (1-5), 95% bootstrap CI")
    ax.set_title(f"{result.flowchart_name}: per-criterion scores by condition (n={result.n})")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _failure_page(plt: Any, pdf: Any, conditions: Sequence[ConditionResult]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [c.condition for c in conditions]
    rates = [c.failure_rate * 100 for c in conditions]
    ax.bar(labels, rates, color="#c0392b")
    ax.set_ylabel("Failure rate (% Task Success <= 3)")
    ax.set_title("Failure rate by condition")
    for i, r in enumerate(rates):
        ax.text(i, r, f"{r:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _cost_page(plt: Any, pdf: Any, conditions: Sequence[ConditionResult]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [c.condition for c in conditions]
    per_convo = [(c.cost_usd / c.n_conversations if c.n_conversations else 0.0) for c in conditions]
    ax.bar(labels, per_convo, color="#2980b9")
    ax.set_ylabel("Cost per conversation (USD)")
    ax.set_title("Cost per conversation by condition")
    for i, v in enumerate(per_convo):
        ax.text(i, v, f"${v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
