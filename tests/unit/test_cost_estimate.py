"""Unit tests for :mod:`agent2model.cloud._costs`.

The estimator is deliberately rough — we assert on **ranges** for the three
paper recipes rather than exact values, since the underlying token assumptions
are tuned. The formatter and prompt gate are asserted on their structural
contract (every section present, ``--yes`` advertised, non-TTY guard).
"""

from __future__ import annotations

import pytest

from agent2model.cloud._costs import (
    CostEstimate,
    confirm_cost_or_exit,
    estimate_cost,
    format_cost_estimate,
)
from agent2model.cloud._recipes import get_recipe

# --------------------------------------------------------------------------- #
# estimate_cost — ranges, not exact values                                     #
# --------------------------------------------------------------------------- #


def test_estimate_cost_returns_cost_estimate() -> None:
    est = estimate_cost(get_recipe("travel"))
    assert isinstance(est, CostEstimate)


def test_estimate_cost_travel_in_sensible_range() -> None:
    est = estimate_cost(get_recipe("travel"))
    # Per the spec: travel is the cheap reproduction. Generation should land
    # in the same ballpark as the paper's $15-50 ceiling; the conservative
    # estimator can go a bit higher.
    assert 20.0 < est.gen_anthropic_usd < 80.0
    assert 1.0 < est.eval_anthropic_usd < 20.0
    # Training: 3B baseline is 3.5h on A100-80GB ($3.40/hr) = ~$11.90.
    # (A10G/A100-40GB are both too tight — see _recipes.GPU_3B.)
    assert 2.0 < est.train_gpu_hours < 6.0
    assert 6.0 < est.train_gpu_usd < 18.0
    assert est.serve_gpu_usd_per_hour == pytest.approx(1.10, abs=0.05)
    assert est.total_excl_serve_usd == pytest.approx(
        est.gen_anthropic_usd + est.eval_anthropic_usd + est.train_gpu_usd, rel=0.01
    )


def test_estimate_cost_zoom_costlier_than_travel() -> None:
    travel = estimate_cost(get_recipe("travel"))
    zoom = estimate_cost(get_recipe("zoom"))
    # 8B path uses 8x A100, more convos, longer transcripts -> clearly bigger.
    assert zoom.gen_anthropic_usd > travel.gen_anthropic_usd * 3
    assert zoom.train_gpu_usd > travel.train_gpu_usd
    assert zoom.serve_gpu_usd_per_hour > travel.serve_gpu_usd_per_hour
    assert zoom.total_excl_serve_usd > travel.total_excl_serve_usd


def test_estimate_cost_insurance_between_travel_and_zoom() -> None:
    travel = estimate_cost(get_recipe("travel"))
    zoom = estimate_cost(get_recipe("zoom"))
    insurance = estimate_cost(get_recipe("insurance"))
    # Insurance is the 55-node 8B run; 3000 convos vs zoom's 6000 -> sits in the
    # middle for generation cost. Training uses the same 8x A100 baseline so
    # train cost matches zoom's.
    assert travel.gen_anthropic_usd < insurance.gen_anthropic_usd < zoom.gen_anthropic_usd
    assert insurance.train_gpu_usd == zoom.train_gpu_usd


def test_estimate_cost_notes_include_assumptions() -> None:
    est = estimate_cost(get_recipe("travel"))
    blob = "\n".join(est.notes).lower()
    assert "rough" in blob or "estimate" in blob
    assert "modal" in blob
    assert "a10g" in blob and "a100" in blob


def test_cost_estimate_is_frozen() -> None:
    from pydantic import ValidationError

    est = estimate_cost(get_recipe("travel"))
    with pytest.raises(ValidationError):
        est.gen_anthropic_usd = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# format_cost_estimate                                                         #
# --------------------------------------------------------------------------- #


def test_format_cost_estimate_includes_breakdown_and_total() -> None:
    recipe = get_recipe("travel")
    est = estimate_cost(recipe)
    text = format_cost_estimate(est, recipe)
    assert "Modal GPU" in text
    assert "Anthropic API (generate" in text
    assert "Anthropic API (evaluate" in text
    assert "TOTAL" in text
    # Concrete numbers from the estimate must appear.
    assert f"{est.train_gpu_usd:.2f}" in text
    assert f"{est.gen_anthropic_usd:.2f}" in text


def test_format_cost_estimate_advertises_yes_flag() -> None:
    recipe = get_recipe("travel")
    text = format_cost_estimate(estimate_cost(recipe), recipe)
    assert "--yes" in text


def test_format_cost_estimate_shows_recipe_context() -> None:
    recipe = get_recipe("zoom")
    text = format_cost_estimate(estimate_cost(recipe), recipe)
    assert recipe.name in text
    assert recipe.size in text
    assert "8b" in text
    assert "A100-80GB:8" in text


# --------------------------------------------------------------------------- #
# confirm_cost_or_exit                                                         #
# --------------------------------------------------------------------------- #


def test_confirm_cost_or_exit_yes_skips_prompt() -> None:
    printed: list[str] = []
    est = confirm_cost_or_exit(
        get_recipe("travel"),
        yes=True,
        printer=lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
        confirmer=lambda _prompt: pytest.fail("must not prompt when --yes"),
    )
    assert isinstance(est, CostEstimate)
    assert any("TOTAL" in line for line in printed)


def test_confirm_cost_or_exit_interactive_accept() -> None:
    asked: list[str] = []
    est = confirm_cost_or_exit(
        get_recipe("travel"),
        yes=False,
        is_interactive=True,
        printer=lambda *a, **k: None,
        confirmer=lambda prompt: asked.append(prompt) or True,
    )
    assert isinstance(est, CostEstimate)
    assert asked == ["Continue? [y/N]: "]


def test_confirm_cost_or_exit_interactive_decline_aborts() -> None:
    with pytest.raises(RuntimeError, match="Aborted"):
        confirm_cost_or_exit(
            get_recipe("travel"),
            yes=False,
            is_interactive=True,
            printer=lambda *a, **k: None,
            confirmer=lambda _prompt: False,
        )


def test_confirm_cost_or_exit_noninteractive_without_yes_fails() -> None:
    with pytest.raises(RuntimeError, match="--yes"):
        confirm_cost_or_exit(
            get_recipe("travel"),
            yes=False,
            is_interactive=False,
            printer=lambda *a, **k: None,
        )
