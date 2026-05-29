"""Pure, modal-free cost estimator for a cloud pipeline run.

Mirrors the shape of :mod:`agent2model.cloud._recipes` — no Modal import, no
Anthropic call — so the same estimator can be exercised by unit tests and
called from the Modal ``local_entrypoint`` before any spend happens.

The estimator is intentionally *rough*: actual cost depends on real conversation
length, judge verbosity, GPU contention, and Modal autoscaling. The point is to
give the user a sane order-of-magnitude before they confirm a multi-hour run.
We reuse Anthropic per-million-token pricing from
:mod:`agent2model.generation.generator` so a single source of truth governs
prices.

GPU rates are baked in from Modal's published per-second prices (rounded to the
nearest cent per hour) as of arXiv:2605.22502v1's publication window. They live
in :data:`MODAL_GPU_USD_PER_HOUR` and are stamped onto every
:class:`CostEstimate` via its ``notes`` field so users can sanity-check.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent2model.cloud._recipes import Recipe
from agent2model.generation.generator import _FALLBACK_PRICING, _PRICING
from agent2model.training.config import ModelSize

__all__ = [
    "DEFAULT_GENERATION_MODEL",
    "MODAL_GPU_USD_PER_HOUR",
    "CostEstimate",
    "confirm_cost_or_exit",
    "estimate_cost",
    "format_cost_estimate",
]

#: Modal per-hour GPU rates in USD. Match Modal's published pricing. The 8x
#: A100 cell is the literal 8x multiplier; Modal bills per-second so this is the
#: nominal sticker price.
MODAL_GPU_USD_PER_HOUR: dict[str, float] = {
    "A10G": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 3.40,
    "A100-80GB:8": 24.0,
}

#: Default Anthropic model id used by the generator and judge — must match
#: :data:`agent2model.generation.generator.DEFAULT_MODEL` for the estimate to
#: line up with what the run actually bills.
DEFAULT_GENERATION_MODEL = "claude-sonnet-4-5"

# Training wall-clock baselines per the paper (§4) — a 3B run on a single A10G
# takes ~3.5h for ~2000 convos / 20 epochs; an 8B run on 8x A100 takes ~0.5h
# for ~6000 convos / 10 epochs. We scale linearly with epochs and convo count
# from those baselines, which is the right order of magnitude for SFT.
_BASELINE_3B: dict[str, float] = {
    "n_convos": 2000.0,
    "epochs": 20.0,
    "hours": 3.5,
}
_BASELINE_8B: dict[str, float] = {
    "n_convos": 6000.0,
    "epochs": 10.0,
    "hours": 0.5,
}

# Per-conversation token assumptions for generation. Travel-style workflows are
# short and chatty; insurance/zoom involve longer transcripts and more domain
# context. We split the difference per size — 3B path uses the lighter profile,
# 8B uses the heavier one. The actual values land in `notes` so a reader can see
# what we assumed.
_TOKENS_LIGHT = {"input_per_turn": 400, "output_per_turn": 130, "turns_per_convo": 7}
_TOKENS_HEAVY = {"input_per_turn": 700, "output_per_turn": 200, "turns_per_convo": 9}

# Per-eval-scenario token assumptions: the judge reads the whole transcript and
# the rubric, then emits a structured rationale. We model that as a single
# weighty call per scenario.
_EVAL_TOKENS_PER_SCENARIO = {"input": 4000, "output": 600}


class CostEstimate(BaseModel):
    """Rough USD cost breakdown for one end-to-end cloud run.

    Numbers are deliberately presented as *point estimates*; the spread is
    documented in :attr:`notes`. Every component is non-negative and the totals
    use simple sums of the components above them.

    Attributes:
        gen_anthropic_usd: Estimated Anthropic spend for synthetic data
            generation.
        eval_anthropic_usd: Estimated Anthropic spend for the eval harness.
        train_gpu_hours: Estimated training wall-clock in GPU-hours
            (multi-GPU runs are wall-clock, not GPU-hours summed; the cost
            below accounts for the GPU count by using the right per-hour rate).
        train_gpu_usd: ``train_gpu_hours * <rate for the recipe's size>``.
        serve_gpu_usd_per_hour: Informational hourly rate for serving the
            compiled model after training. Not added to the total — serving is
            autoscaled and an interactive decision.
        total_excl_serve_usd: Sum of generation + eval + training. Serving is
            excluded.
        notes: Caveats, ranges, and the literal assumptions used. Always
            non-empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_anthropic_usd: float = Field(ge=0.0)
    eval_anthropic_usd: float = Field(ge=0.0)
    train_gpu_hours: float = Field(ge=0.0)
    train_gpu_usd: float = Field(ge=0.0)
    serve_gpu_usd_per_hour: float = Field(ge=0.0)
    total_excl_serve_usd: float = Field(ge=0.0)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _rates_for(model: str) -> dict[str, float]:
    """USD-per-million-token rates for an Anthropic model id."""
    return _PRICING.get(model, _FALLBACK_PRICING)


def _gen_gpu_spec_for_size(size: ModelSize) -> str:
    """Return the Modal GPU spec for a training size — kept in sync with ``_recipes``."""
    return "A100-80GB" if size == "3b" else "A100-80GB:8"


def _train_hours_for(recipe: Recipe) -> float:
    """Scale the paper's wall-clock baselines linearly to this recipe's params.

    The scale is linear in ``n_convos * epochs`` (the total number of training
    examples seen). It is rough on purpose; users override with ``--yes`` once
    they have first-run telemetry.
    """
    baseline = _BASELINE_3B if recipe.size == "3b" else _BASELINE_8B
    baseline_work = float(baseline["n_convos"]) * float(baseline["epochs"])
    this_work = float(recipe.n_convos) * float(recipe.epochs)
    if baseline_work <= 0:
        return float(baseline["hours"])  # pragma: no cover - defensive
    return float(baseline["hours"]) * (this_work / baseline_work)


def _serve_rate_for(size: ModelSize) -> float:
    """Per-hour serving rate. 3B serves on a cheap A10G, 8B on a single A100."""
    return MODAL_GPU_USD_PER_HOUR["A10G"] if size == "3b" else MODAL_GPU_USD_PER_HOUR["A100-80GB"]


# --------------------------------------------------------------------------- #
# Public entrypoint                                                            #
# --------------------------------------------------------------------------- #


def estimate_cost(
    recipe: Recipe,
    *,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    judge_model: str | None = None,
) -> CostEstimate:
    """Compute a rough USD cost breakdown for running a recipe end to end.

    Args:
        recipe: The recipe whose cost to estimate.
        generation_model: Anthropic model id used for synthetic generation.
            Defaults to the production default; the rate comes from
            :data:`agent2model.generation.generator._PRICING`.
        judge_model: Anthropic model id used for the LLM judge during eval.
            Defaults to ``generation_model``.

    Returns:
        A :class:`CostEstimate`. ``serve_gpu_usd_per_hour`` is informational
        and is **not** added to ``total_excl_serve_usd``.

    Example:
        >>> from agent2model.cloud._recipes import get_recipe
        >>> est = estimate_cost(get_recipe("travel"))
        >>> est.total_excl_serve_usd > 0
        True
    """
    gen_profile = _TOKENS_LIGHT if recipe.size == "3b" else _TOKENS_HEAVY
    judge = judge_model or generation_model

    # --- Generation ---
    gen_rates = _rates_for(generation_model)
    gen_input_tokens = (
        recipe.n_convos * gen_profile["turns_per_convo"] * gen_profile["input_per_turn"]
    )
    gen_output_tokens = (
        recipe.n_convos * gen_profile["turns_per_convo"] * gen_profile["output_per_turn"]
    )
    gen_anthropic_usd = (
        gen_input_tokens * gen_rates["input"] / 1_000_000
        + gen_output_tokens * gen_rates["output"] / 1_000_000
    )

    # --- Evaluation ---
    eval_rates = _rates_for(judge)
    eval_input_tokens = recipe.eval_n * _EVAL_TOKENS_PER_SCENARIO["input"]
    eval_output_tokens = recipe.eval_n * _EVAL_TOKENS_PER_SCENARIO["output"]
    eval_anthropic_usd = (
        eval_input_tokens * eval_rates["input"] / 1_000_000
        + eval_output_tokens * eval_rates["output"] / 1_000_000
    )

    # --- Training (Modal GPU) ---
    train_hours = _train_hours_for(recipe)
    gpu_spec = _gen_gpu_spec_for_size(recipe.size)
    gpu_rate = MODAL_GPU_USD_PER_HOUR[gpu_spec]
    train_gpu_usd = train_hours * gpu_rate

    # --- Serving (informational only) ---
    serve_rate = _serve_rate_for(recipe.size)

    total = gen_anthropic_usd + eval_anthropic_usd + train_gpu_usd

    notes = [
        "Rough estimate, ~+/-2x; actual depends on conversation length and turn count.",
        f"Generation: assumes ~{gen_profile['input_per_turn']} input + "
        f"{gen_profile['output_per_turn']} output tokens per turn, "
        f"{gen_profile['turns_per_convo']} turns per convo, {generation_model}.",
        f"Eval: assumes ~{_EVAL_TOKENS_PER_SCENARIO['input']} input + "
        f"{_EVAL_TOKENS_PER_SCENARIO['output']} output tokens per scenario, {judge}.",
        f"Modal GPU rates: A10G ${MODAL_GPU_USD_PER_HOUR['A10G']:.2f}/hr, "
        f"A100-80GB ${MODAL_GPU_USD_PER_HOUR['A100-80GB']:.2f}/hr, "
        f"8xA100-80GB ${MODAL_GPU_USD_PER_HOUR['A100-80GB:8']:.2f}/hr.",
        f"Training: {recipe.size} on {gpu_spec}, scaled from paper baseline "
        f"(3B 2000x20=3.5h; 8B 6000x10=0.5h).",
        f"Serving rate {gpu_spec} ${serve_rate:.2f}/hr is informational; "
        "actual serve cost depends on uptime and autoscaling.",
    ]

    return CostEstimate(
        gen_anthropic_usd=round(gen_anthropic_usd, 4),
        eval_anthropic_usd=round(eval_anthropic_usd, 4),
        train_gpu_hours=round(train_hours, 4),
        train_gpu_usd=round(train_gpu_usd, 4),
        serve_gpu_usd_per_hour=round(serve_rate, 4),
        total_excl_serve_usd=round(total, 4),
        notes=notes,
    )


def format_cost_estimate(est: CostEstimate, recipe: Recipe) -> str:
    """Render a :class:`CostEstimate` as the confirm-prompt body.

    Pure formatter — no I/O, no colour. Kept separate from the prompt itself so
    tests can assert on the rendered text without driving stdin.

    Args:
        est: The estimate to render.
        recipe: The recipe being estimated (used for the contextual header).

    Returns:
        A multi-line string suitable for printing immediately above a yes/no
        prompt. Always ends with the ``--yes`` skip hint.

    Example:
        >>> from agent2model.cloud._recipes import get_recipe
        >>> text = format_cost_estimate(estimate_cost(get_recipe('travel')),
        ...                              get_recipe('travel'))
        >>> 'TOTAL' in text
        True
    """
    gpu_spec = _gen_gpu_spec_for_size(recipe.size)
    lines = [
        f"This run ({recipe.name}, {recipe.size}, {recipe.n_convos} convos, "
        f"{recipe.epochs} epochs, {recipe.eval_n} eval scenarios) is estimated to cost:",
        f"  Modal GPU (train_{recipe.size}, ~{est.train_gpu_hours:.2f}h on "
        f"{gpu_spec}):  ~${est.train_gpu_usd:.2f}",
        f"  Anthropic API (generate, {recipe.n_convos} convos): ~${est.gen_anthropic_usd:.2f}",
        f"  Anthropic API (evaluate, {recipe.eval_n} scenarios): ~${est.eval_anthropic_usd:.2f}",
        f"  TOTAL (excl. serve):              ~${est.total_excl_serve_usd:.2f}",
        f"  (Serving, if --serve-after: ~${est.serve_gpu_usd_per_hour:.2f}/hr while live.)",
        "Notes:",
    ]
    lines.extend(f"  - {note}" for note in est.notes)
    lines.append("  - `--yes` skips this prompt (for CI / non-interactive runs).")
    return "\n".join(lines)


def confirm_cost_or_exit(
    recipe: Recipe,
    *,
    yes: bool = False,
    is_interactive: bool | None = None,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    judge_model: str | None = None,
    printer: object = None,
    confirmer: object = None,
) -> CostEstimate:
    """Print a cost estimate and gate the run on a yes/no confirmation.

    Behaviour:

    * ``yes=True`` (the ``--yes`` flag) prints the estimate and returns
      immediately without prompting. Use this in CI.
    * Otherwise, when stdin is **not** a TTY (``is_interactive`` resolves to
      ``False``) the function raises :class:`RuntimeError` rather than
      hanging on a missing terminal.
    * In an interactive terminal the user is asked to confirm; ``no`` raises
      :class:`RuntimeError` with the standard "aborted" message.

    Args:
        recipe: The recipe being launched.
        yes: When True, skip the prompt.
        is_interactive: Override the TTY detection. When ``None``, falls back
            to :func:`sys.stdin.isatty`.
        generation_model: Anthropic model id for the generation cost estimate.
        judge_model: Anthropic model id for the eval cost estimate.
        printer: Optional override for ``print`` (tests).
        confirmer: Optional override for the yes/no input function. Receives
            the prompt string and must return a bool.

    Returns:
        The :class:`CostEstimate` that was rendered, so callers can record it.

    Raises:
        RuntimeError: If the user declines, or stdin is non-interactive and
            ``yes`` is False.

    Example:
        >>> from agent2model.cloud._recipes import get_recipe
        >>> est = confirm_cost_or_exit(get_recipe('travel'), yes=True,
        ...     printer=lambda *_a, **_k: None)
        >>> est.total_excl_serve_usd > 0
        True
    """
    import sys

    est = estimate_cost(recipe, generation_model=generation_model, judge_model=judge_model)
    text = format_cost_estimate(est, recipe)

    p = printer if callable(printer) else print
    p(text)

    if yes:
        return est

    interactive = is_interactive if is_interactive is not None else sys.stdin.isatty()
    if not interactive:
        raise RuntimeError(
            "Cost confirmation required but stdin is not a TTY; "
            "pass --yes to confirm non-interactively."
        )

    def _default_confirmer(prompt: str) -> bool:
        try:
            reply = input(prompt).strip().lower()
        except EOFError:
            return False
        return reply in {"y", "yes"}

    asker = confirmer if callable(confirmer) else _default_confirmer
    if not asker("Continue? [y/N]: "):
        raise RuntimeError("Aborted by user.")
    return est
