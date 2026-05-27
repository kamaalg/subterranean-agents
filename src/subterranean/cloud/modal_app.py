"""Modal deployment recipe for the full subterranean pipeline (Phase 7).

This is the primary cloud recipe: a user with **no local GPU** reproduces a paper
experiment end to end with a single command, e.g.::

    modal run -m subterranean.cloud.modal_app::reproduce_travel

Why this module requires ``modal`` to import
---------------------------------------------
A Modal app is defined declaratively: functions are decorated with
``@app.function(...)`` at module scope, which needs ``import modal`` at the top.
So **this module deliberately requires ``modal``** and is not importable without
the ``[cloud]`` extra. The rest of the package stays modal-free — ``import
subterranean`` and ``import subterranean.cloud`` work without modal, and all the
pure recipe logic lives in :mod:`subterranean.cloud._recipes` (no modal import),
which is what the unit tests exercise. ``cloud/__init__.py` does **not** import
this module, keeping the import contract intact.

Layout on Modal
---------------
* Three images: a CPU image (core + ``anthropic``) for the API-bound
  generate/evaluate steps, a ``[train]`` image for fine-tuning, and a ``[serve]``
  image (vLLM) for the inference endpoint.
* Two persisted volumes: ``BUILD_VOLUME`` for build artifacts (the flowchart IR,
  generated dataset, eval reports) and ``MODEL_VOLUME`` for fine-tuned weights.
* One secret: ``anthropic-secret`` providing ``ANTHROPIC_API_KEY`` to the
  API-bound functions.

Functions
---------
* :func:`generate_data` — CPU, calls :mod:`subterranean.generation`.
* :func:`train_3b` — single A10G/A100, the 3B path.
* :func:`train_8b` — 8x A100 with DeepSpeed ZeRO-3, the 8B path.
* :func:`evaluate` — CPU, API-bound, calls :mod:`subterranean.eval`.
* :func:`serve` — autoscaling vLLM endpoint wrapping :mod:`subterranean.serve`.

Entrypoints (``modal run -m subterranean.cloud.modal_app::<name>``):
``reproduce_travel`` (3B), ``reproduce_zoom`` (8B), ``reproduce_insurance`` (8B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from subterranean.cloud._recipes import (
    GPU_3B,
    GPU_8B,
    ExampleRecipe,
    build_training_config,
    get_recipe,
)

# --------------------------------------------------------------------------- #
# App, images, volumes, secrets                                                #
# --------------------------------------------------------------------------- #

APP_NAME = "subterranean"
app = modal.App(APP_NAME)

# The package is installed from its source tree (PyPI once published). Each image
# pulls the extras the step needs and nothing more, keeping cold-starts lean.
_PIP_PKG = "subterranean-agents"

#: CPU image for the API-bound generate/evaluate steps (core + anthropic only).
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[report]")

#: Training image: the heavy ML stack (torch/trl/deepspeed/bitsandbytes).
train_image = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[train]")

#: Serving image: vLLM (CUDA/Linux only).
serve_image = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[serve]")

#: Persisted build artifacts (flowchart IR, dataset.jsonl, eval reports).
BUILD_VOLUME = modal.Volume.from_name("subterranean-build", create_if_missing=True)
#: Persisted fine-tuned model weights.
MODEL_VOLUME = modal.Volume.from_name("subterranean-models", create_if_missing=True)

BUILD_ROOT = "/build"
MODEL_ROOT = "/models"
_VOLUMES = {BUILD_ROOT: BUILD_VOLUME, MODEL_ROOT: MODEL_VOLUME}

#: Anthropic API key, injected into the API-bound functions.
ANTHROPIC_SECRET = modal.Secret.from_name("anthropic-secret")

# Timeouts (seconds). Generation/eval are long API-bound jobs; the 3B run is the
# paper's ~3.5h; the 8B ZeRO-3 run is fast (~15-30 min) but gets head-room.
_HOUR = 60 * 60
GENERATE_TIMEOUT = 6 * _HOUR
TRAIN_3B_TIMEOUT = 6 * _HOUR
TRAIN_8B_TIMEOUT = 3 * _HOUR
EVALUATE_TIMEOUT = 4 * _HOUR


def _build_dir(example: str) -> str:
    """Return the on-volume build directory for an example."""
    return f"{BUILD_ROOT}/{example}"


def _model_dir(example: str) -> str:
    """Return the on-volume model output directory for an example."""
    return f"{MODEL_ROOT}/{example}"


# --------------------------------------------------------------------------- #
# Pipeline functions                                                           #
# --------------------------------------------------------------------------- #


# modal's decorators are typed as Any (see [tool.mypy.overrides] for modal.*), so
# strict mode flags the decorated functions as "untyped". That is expected for the
# Modal SDK surface; the wrapped function bodies are themselves fully annotated.
@app.function(  # type: ignore[untyped-decorator]
    image=cpu_image,
    volumes=_VOLUMES,
    secrets=[ANTHROPIC_SECRET],
    timeout=GENERATE_TIMEOUT,
    cpu=4.0,
)
def generate_data(
    example: str,
    *,
    n: int,
    budget_usd: float,
    model: str,
    seed: int = 0,
    max_concurrent: int = 10,
) -> str:
    """Generate synthetic training data on a Modal CPU worker.

    Reads ``<build>/flowchart.json`` from the build volume, runs the async
    :class:`~subterranean.generation.generator.ConversationGenerator`, and writes
    ``<build>/dataset.jsonl`` back to the volume. This step is API-bound (no GPU).

    Args:
        example: Reproduction example name (``travel`` / ``zoom`` / ``insurance``).
        n: Number of conversations to generate.
        budget_usd: Hard USD cap; generation stops if exceeded.
        model: Anthropic model id for generation.
        seed: Base RNG seed.
        max_concurrent: Maximum in-flight API calls.

    Returns:
        The path to the written ``dataset.jsonl`` on the build volume.
    """
    import asyncio

    from subterranean.cli import _load_compiled_flowchart
    from subterranean.generation.formatter import write_dataset
    from subterranean.generation.generator import ConversationGenerator, GenerationConfig
    from subterranean.logging import logger

    build = Path(_build_dir(example))
    flowchart = _load_compiled_flowchart(build)
    config = GenerationConfig(
        n=n, model=model, budget_usd=budget_usd, seed=seed, max_concurrent=max_concurrent
    )
    generator = ConversationGenerator(flowchart, config)
    conversations = asyncio.run(generator.run(build))

    dataset_path = build / "dataset.jsonl"
    written = write_dataset(conversations, dataset_path)
    logger.info(f"Generated {written} conversations to {dataset_path}.")
    BUILD_VOLUME.commit()
    return str(dataset_path)


def _run_training(example: str) -> str:
    """Shared body for the 3B/8B training functions.

    Loads the dataset from the build volume, builds the per-example
    :class:`~subterranean.training.config.TrainingConfig` (3B single-GPU or 8B
    ZeRO-3), trains via :func:`subterranean.training.trainer.train`, and persists
    the best checkpoint to the model volume.

    Args:
        example: Reproduction example name.

    Returns:
        The path to the best checkpoint on the model volume.
    """
    from subterranean.logging import logger
    from subterranean.training.trainer import train

    recipe = get_recipe(example)
    build = Path(_build_dir(example))
    output_dir = _model_dir(example)

    config = build_training_config(recipe, output_dir)
    logger.info(
        f"Training {example} ({config.size}): base={config.base_model}, "
        f"epochs={config.epochs}, gpus={config.num_gpus}."
    )
    best = train(config, build / "dataset.jsonl")
    MODEL_VOLUME.commit()
    logger.info(f"Best checkpoint: {best.path} (eval_loss={best.eval_loss}).")
    return best.path


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=train_image,
    gpu=GPU_3B,
    volumes=_VOLUMES,
    timeout=TRAIN_3B_TIMEOUT,
)
def train_3b(example: str) -> str:
    """Fine-tune the 3B path on a single A10G/A100.

    Args:
        example: Reproduction example name (a 3B example, e.g. ``travel``).

    Returns:
        The path to the best checkpoint on the model volume.
    """
    return _run_training(example)


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=train_image,
    gpu=GPU_8B,
    volumes=_VOLUMES,
    timeout=TRAIN_8B_TIMEOUT,
)
def train_8b(example: str) -> str:
    """Fine-tune the 8B path on 8x A100 80GB with DeepSpeed ZeRO-3.

    The ZeRO-3 config ships at
    :data:`subterranean.training.deepspeed.ZERO3_CONFIG_PATH`; the trainer wires
    it through the multi-GPU :meth:`TrainingConfig.for_8b` preset.

    Args:
        example: Reproduction example name (an 8B example, e.g. ``zoom``).

    Returns:
        The path to the best checkpoint on the model volume.
    """
    return _run_training(example)


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=cpu_image,
    volumes=_VOLUMES,
    secrets=[ANTHROPIC_SECRET],
    timeout=EVALUATE_TIMEOUT,
    cpu=4.0,
)
def evaluate(
    example: str,
    *,
    n: int,
    budget_usd: float,
    baselines: tuple[str, ...] = ("in_context",),
    served_url: str | None = None,
    judge_model: str | None = None,
    seed: int = 0,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """Run the evaluation harness, parallel across scenarios, on a CPU worker.

    Samples ``n`` scenarios, runs each condition concurrently against a
    flowchart-blind user simulator, judges on the 5-criterion rubric, and writes
    ``eval_report.json``/``eval_report.pdf`` to the build volume.

    Args:
        example: Reproduction example name.
        n: Scenarios per condition.
        budget_usd: Hard USD cap across all LLM calls.
        baselines: Baseline condition names to evaluate against.
        served_url: OpenAI-compatible base URL of a served compiled model; when
            given, the served ``compiled`` condition is added.
        judge_model: Anthropic model id for the judge (defaults to the harness's).
        seed: Base RNG seed.
        max_concurrent: Concurrent scenario evaluations.

    Returns:
        The serialised :class:`~subterranean.eval.runner.EvalRunResult`.
    """
    import asyncio

    from subterranean.cli import _load_compiled_flowchart
    from subterranean.eval.baselines import make_condition
    from subterranean.eval.judge import Judge, JudgeConfig
    from subterranean.eval.report import write_json_report, write_pdf_report
    from subterranean.eval.runner import EvalConfig, EvalRunner
    from subterranean.exceptions import EvalError
    from subterranean.generation.generator import DEFAULT_MODEL
    from subterranean.logging import logger

    build = Path(_build_dir(example))
    flowchart = _load_compiled_flowchart(build)

    names = list(baselines)
    if served_url:
        names.append("compiled")
    conditions = [make_condition(name, flowchart, served_url=served_url) for name in names]

    judge_cfg = JudgeConfig(model=judge_model or DEFAULT_MODEL)
    config = EvalConfig(
        n=n,
        budget_usd=budget_usd,
        seed=seed,
        max_concurrent=max_concurrent,
        judge=judge_cfg,
    )
    runner = EvalRunner(flowchart, conditions, config, judge=Judge(config.judge))
    result = asyncio.run(runner.run())

    write_json_report(result, build / "eval_report.json")
    try:
        write_pdf_report(result, build / "eval_report.pdf")
    except EvalError as exc:  # pragma: no cover - report extra optional on Modal
        logger.warning(str(exc))
    BUILD_VOLUME.commit()
    logger.info(f"Evaluated {example}: total cost ${result.total_cost_usd:.4f}.")
    return result.model_dump(mode="json")


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=serve_image,
    gpu="A100-80GB",
    volumes=_VOLUMES,
    timeout=_HOUR,
    scaledown_window=300,
    min_containers=0,
    max_containers=4,
)
@modal.concurrent(max_inputs=32)  # type: ignore[untyped-decorator]
@modal.web_server(port=8000, startup_timeout=600)  # type: ignore[untyped-decorator]
def serve(example: str) -> None:
    """Serve a compiled model behind an autoscaling, OpenAI-compatible vLLM endpoint.

    Resolves the best checkpoint for ``example`` on the model volume and launches
    vLLM's OpenAI-compatible server (``/v1/chat/completions``). Modal autoscaling
    spins containers up on demand and down to zero after the scaledown window.

    Args:
        example: Reproduction example name whose compiled model to serve.
    """
    from subterranean.serve import vllm_server

    model_path = vllm_server.resolve_model_path(_model_dir(example))
    vllm_server.serve(model_path, port=8000, host="0.0.0.0", served_model_name=example)


# --------------------------------------------------------------------------- #
# Reproduction entrypoints                                                     #
# --------------------------------------------------------------------------- #


def _reproduce(recipe: ExampleRecipe) -> dict[str, Any]:
    """Chain generate -> train -> evaluate for one reproduction recipe.

    Dispatches training to :func:`train_3b` or :func:`train_8b` based on the
    recipe's model size, then runs evaluation against the ``in_context`` upper
    bound. Each step is a remote Modal call (``.remote``).

    Args:
        recipe: The per-example reproduction recipe.

    Returns:
        The serialised evaluation result for the run.
    """
    from subterranean.generation.generator import DEFAULT_MODEL
    from subterranean.logging import logger

    logger.info(f"Reproducing {recipe.name} ({recipe.size}) end to end on Modal.")

    generate_data.remote(
        recipe.name,
        n=recipe.n_convos,
        budget_usd=recipe.gen_budget_usd,
        model=DEFAULT_MODEL,
    )

    trainer = train_3b if recipe.size == "3b" else train_8b
    trainer.remote(recipe.name)

    return evaluate.remote(  # type: ignore[no-any-return]
        recipe.name,
        n=recipe.eval_n,
        budget_usd=recipe.eval_budget_usd,
        baselines=("in_context",),
    )


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_travel() -> None:
    """Reproduce the Travel experiment (Qwen2.5-3B, ~2000 convos, 20 epochs).

    Run with ``modal run -m subterranean.cloud.modal_app::reproduce_travel``.
    """
    _reproduce(get_recipe("travel"))


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_zoom() -> None:
    """Reproduce the Zoom experiment (Qwen3-8B, ~6000 convos, 10 epochs).

    Run with ``modal run -m subterranean.cloud.modal_app::reproduce_zoom``.
    """
    _reproduce(get_recipe("zoom"))


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_insurance() -> None:
    """Reproduce the Insurance experiment (Qwen3-8B, 55 nodes, ~3000 convos, 20 epochs).

    Run with ``modal run -m subterranean.cloud.modal_app::reproduce_insurance``.
    """
    _reproduce(get_recipe("insurance"))
