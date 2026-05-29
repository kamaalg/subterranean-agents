"""End-to-end paper-reproduction tests (release gate).

Each test loads a trained model's ``eval_report.json``, extracts the compiled
model's per-criterion means, and asserts no criterion regressed more than 5%
below the paper target in ``benchmarks/targets.json``.

These require a GPU-trained model and a real eval run, neither of which exists in
CI by default. They are therefore **skipped** unless both:

- ``AGENT2MODEL_E2E=1`` is set, and
- ``AGENT2MODEL_E2E_REPORTS`` points at a directory containing
  ``<example>/eval_report.json`` for each reproduced example.

The pure regression math is covered independently by
``tests/unit/test_regression_gate.py``, so the >5% gate logic is always tested.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.e2e.regression import (
    assert_no_regression,
    load_targets,
    observed_from_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS_PATH = REPO_ROOT / "benchmarks" / "targets.json"

EXAMPLES = ["travel_booking", "zoom_support", "insurance_claims"]

pytestmark = pytest.mark.e2e


def _reports_dir() -> Path | None:
    """Resolve the reports directory from env, or ``None`` if e2e is disabled."""
    if os.environ.get("AGENT2MODEL_E2E") != "1":
        return None
    raw = os.environ.get("AGENT2MODEL_E2E_REPORTS")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


@pytest.mark.parametrize("example", EXAMPLES)
def test_reproduction_within_tolerance(example: str) -> None:
    """The compiled model must score within 5% of the paper on every criterion."""
    reports_dir = _reports_dir()
    if reports_dir is None:
        pytest.skip(
            "e2e reproduction requires a GPU-trained model: set AGENT2MODEL_E2E=1 "
            "and AGENT2MODEL_E2E_REPORTS=<dir with <example>/eval_report.json>."
        )

    report_path = reports_dir / example / "eval_report.json"
    if not report_path.exists():
        pytest.skip(f"No eval report for {example} at {report_path}.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    observed = observed_from_report(report, condition="compiled")
    targets = load_targets(example, TARGETS_PATH)

    assert_no_regression(observed, targets, tol=0.05)
