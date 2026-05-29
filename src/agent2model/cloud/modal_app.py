"""Modal deployment recipes for the full agent2model pipeline (Phase 7).

This module exposes:

* Five **worker functions** (``generate_data``, ``train_3b``, ``train_8b``,
  ``evaluate``, ``serve``) parameterised by a :class:`~agent2model.cloud._recipes.Recipe`
  pydantic model. Modal cloudpickles Pydantic args fine, so the worker receives the
  full recipe — including the flowchart YAML text inline — without needing access
  to the caller's local filesystem.
* A **generic** ``run`` local entrypoint that takes a path to *any* user
  flowchart (``.yaml`` or LangGraph ``.py``) and chains generate → train → optional
  evaluate → optional serve, persisting outputs under the recipe's name on the
  Modal volumes.
* Three **paper-reproduction** entrypoints (``reproduce_travel``,
  ``reproduce_zoom``, ``reproduce_insurance``) that are thin wrappers around
  :func:`run` using the pre-built :data:`~agent2model.cloud._recipes.EXAMPLES`
  recipes. Kept for backward compatibility with the published paper-repro docs.

Why this module requires ``modal`` to import
---------------------------------------------
A Modal app is defined declaratively: functions are decorated with
``@app.function(...)`` at module scope, which needs ``import modal`` at the top.
So **this module deliberately requires ``modal``** and is not importable without
the ``[cloud]`` extra. The rest of the package stays modal-free — ``import
agent2model`` and ``import agent2model.cloud`` work without modal, and all the
pure recipe logic lives in :mod:`agent2model.cloud._recipes` (no modal import),
which is what the unit tests exercise. ``cloud/__init__.py`` does **not** import
this module, keeping the import contract intact.

Layout on Modal
---------------
* Three images: a CPU image (core + ``anthropic``) for the API-bound
  generate/evaluate steps, a ``[train]`` image for fine-tuning, and a ``[serve]``
  image (vLLM) for the inference endpoint.
* Two persisted volumes: ``BUILD_VOLUME`` for build artifacts (the flowchart IR,
  generated dataset, eval reports) and ``MODEL_VOLUME`` for fine-tuned weights.
  Outputs are keyed by ``recipe.name`` — two users with the same name will
  collide (last-write-wins is fine for v1).
* One secret: ``anthropic-secret`` providing ``ANTHROPIC_API_KEY`` to the
  API-bound functions.

Entrypoints (``modal run -m agent2model.cloud.modal_app::<name>``):
``run`` (generic), ``reproduce_travel``, ``reproduce_zoom``, ``reproduce_insurance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent2model.cloud._recipes import (
    DEFAULT_BASE_FOR_SIZE,
    EXAMPLES,
    GPU_3B,
    GPU_8B,
    ExampleRecipe,
    Recipe,
    build_training_config,
    get_recipe,
)
from agent2model.cloud.modal_app_constants import (
    ANTHROPIC_SECRET,
    APP_NAME,
    BUILD_ROOT,
    BUILD_VOLUME,
    CPU_IMAGE,
    EVALUATE_TIMEOUT,
    GENERATE_TIMEOUT,
    HOUR,
    MODEL_ROOT,
    MODEL_VOLUME,
    SERVE_APP,
    SERVE_IMAGE,
    TRAIN_3B_TIMEOUT,
    TRAIN_8B_TIMEOUT,
    TRAIN_IMAGE,
    VOLUMES,
)

# --------------------------------------------------------------------------- #
# App, images, volumes, secrets                                                #
# --------------------------------------------------------------------------- #

#: The Modal app. Defined in :mod:`modal_app_constants` so the serve class in
#: :mod:`_modal_serve` registers against the same instance.
app = SERVE_APP

#: Backward-compat aliases preserved for downstream code/tests that imported
#: the lower-case names from this module.
cpu_image = CPU_IMAGE
train_image = TRAIN_IMAGE
serve_image = SERVE_IMAGE
_VOLUMES = VOLUMES
_HOUR = HOUR

# Importing this triggers the @app.cls registration; do it after the app exists
# so the class lands on the same App instance.
from agent2model.cloud import _modal_serve  # noqa: E402  (intentional ordering)
from agent2model.cloud._modal_serve import ServeCls  # noqa: E402

_ = _modal_serve  # silence unused-import linters; we need the side effect


def _build_dir(name: str) -> str:
    """Return the on-volume build directory for a recipe name."""
    return f"{BUILD_ROOT}/{name}"


def _model_dir(name: str) -> str:
    """Return the on-volume model output directory for a recipe name."""
    return f"{MODEL_ROOT}/{name}"


def _materialise_flowchart(recipe: Recipe) -> Path:
    """Write the recipe's inline YAML to the build volume and return the build dir.

    The worker side of the boundary: the recipe carries ``flowchart_yaml`` over
    the wire from the caller; we persist it on the build volume so the rest of
    the pipeline (generation, eval) can compile it the same way the local CLI
    does. The compiled JSON IR also lands here as ``flowchart.json``.
    """
    import json

    from agent2model.ir.loader import load_flowchart_from_string
    from agent2model.ir.validator import validate

    build = Path(_build_dir(recipe.name))
    build.mkdir(parents=True, exist_ok=True)
    yaml_path = build / "flowchart.yaml"
    yaml_path.write_text(recipe.flowchart_yaml, encoding="utf-8")

    flowchart = load_flowchart_from_string(recipe.flowchart_yaml)
    validate(flowchart)
    ir_path = build / "flowchart.json"
    ir_path.write_text(
        json.dumps(flowchart.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return build


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
    recipe: Recipe,
    *,
    model: str | None = None,
    seed: int = 0,
    max_concurrent: int = 10,
) -> str:
    """Generate synthetic training data on a Modal CPU worker.

    Materialises the recipe's flowchart YAML to ``<build>/flowchart.{yaml,json}``
    on the build volume, runs the async
    :class:`~agent2model.generation.generator.ConversationGenerator`, and writes
    ``<build>/dataset.jsonl`` back to the volume. This step is API-bound (no GPU).

    Args:
        recipe: The recipe to run (carries flowchart, n_convos, budget, name).
        model: Anthropic model id for generation. Defaults to the generator's
            default.
        seed: Base RNG seed.
        max_concurrent: Maximum in-flight API calls.

    Returns:
        The path to the written ``dataset.jsonl`` on the build volume.
    """
    import asyncio

    from agent2model.generation.formatter import write_dataset
    from agent2model.generation.generator import (
        DEFAULT_MODEL,
        ConversationGenerator,
        GenerationConfig,
    )
    from agent2model.logging import logger

    build = _materialise_flowchart(recipe)
    from agent2model.ir.loader import load_flowchart_from_string

    flowchart = load_flowchart_from_string(recipe.flowchart_yaml)
    config = GenerationConfig(
        n=recipe.n_convos,
        model=model or DEFAULT_MODEL,
        budget_usd=recipe.gen_budget_usd,
        seed=seed,
        max_concurrent=max_concurrent,
    )
    generator = ConversationGenerator(flowchart, config)
    conversations = asyncio.run(generator.run(build))

    dataset_path = build / "dataset.jsonl"
    written = write_dataset(conversations, dataset_path)
    logger.info(f"Generated {written} conversations to {dataset_path}.")
    BUILD_VOLUME.commit()
    return str(dataset_path)


def _run_training(recipe: Recipe, dataset_path: str | None = None) -> str:
    """Shared body for the 3B/8B training functions.

    Loads the dataset from the build volume (default location, or the explicit
    ``dataset_path`` returned by :func:`generate_data`), builds the per-recipe
    :class:`~agent2model.training.config.TrainingConfig` (3B single-GPU or 8B
    ZeRO-3), trains via :func:`agent2model.training.trainer.train`, and persists
    the best checkpoint to the model volume.

    Args:
        recipe: The recipe to train against.
        dataset_path: Optional explicit dataset path; defaults to
            ``<build>/dataset.jsonl``.

    Returns:
        The path to the best checkpoint on the model volume.
    """
    from agent2model.logging import logger
    from agent2model.training.trainer import train

    build = Path(_build_dir(recipe.name))
    dataset = Path(dataset_path) if dataset_path else build / "dataset.jsonl"
    output_dir = _model_dir(recipe.name)

    config = build_training_config(recipe, output_dir)
    logger.info(
        f"Training {recipe.name} ({config.size}): base={config.base_model}, "
        f"epochs={config.epochs}, gpus={config.num_gpus}."
    )
    best = train(config, dataset)
    MODEL_VOLUME.commit()
    logger.info(f"Best checkpoint: {best.path} (eval_loss={best.eval_loss}).")
    return best.path


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=train_image,
    gpu=GPU_3B,
    volumes=_VOLUMES,
    timeout=TRAIN_3B_TIMEOUT,
)
def train_3b(recipe: Recipe, dataset_path: str | None = None) -> str:
    """Fine-tune the 3B path on a single A10G/A100.

    Args:
        recipe: A recipe with ``size="3b"``.
        dataset_path: Optional explicit dataset path.

    Returns:
        The path to the best checkpoint on the model volume.
    """
    return _run_training(recipe, dataset_path)


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=train_image,
    gpu=GPU_8B,
    volumes=_VOLUMES,
    timeout=TRAIN_8B_TIMEOUT,
)
def train_8b(recipe: Recipe, dataset_path: str | None = None) -> str:
    """Fine-tune the 8B path on 8x A100 80GB with DeepSpeed ZeRO-3.

    The ZeRO-3 config ships at
    :data:`agent2model.training.deepspeed.ZERO3_CONFIG_PATH`; the trainer wires
    it through the multi-GPU :meth:`TrainingConfig.for_8b` preset.

    Args:
        recipe: A recipe with ``size="8b"``.
        dataset_path: Optional explicit dataset path.

    Returns:
        The path to the best checkpoint on the model volume.
    """
    return _run_training(recipe, dataset_path)


@app.function(  # type: ignore[untyped-decorator]  # modal decorators are Any (see overrides)
    image=cpu_image,
    volumes=_VOLUMES,
    secrets=[ANTHROPIC_SECRET],
    timeout=EVALUATE_TIMEOUT,
    cpu=4.0,
)
def evaluate(
    recipe: Recipe,
    model_path: str | None = None,
    *,
    baselines: tuple[str, ...] = ("in_context",),
    served_url: str | None = None,
    judge_model: str | None = None,
    seed: int = 0,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """Run the evaluation harness, parallel across scenarios, on a CPU worker.

    Samples ``recipe.eval_n`` scenarios, runs each condition concurrently against
    a flowchart-blind user simulator, judges on the 5-criterion rubric, and writes
    ``eval_report.json``/``eval_report.pdf`` to the build volume under
    ``recipe.name``.

    Note: returns the eval result synchronously (not ``.spawn()``).

    Args:
        recipe: The recipe being evaluated.
        model_path: Optional path to the trained model (returned by ``train_*``).
            Unused here directly — eval drives a served URL or baselines — but
            accepted for symmetry with the pipeline so callers can pass it
            through verbatim from ``train.remote``.
        baselines: Baseline condition names to evaluate against.
        served_url: OpenAI-compatible base URL of a served compiled model; when
            given, the served ``compiled`` condition is added.
        judge_model: Anthropic model id for the judge (defaults to the harness's).
        seed: Base RNG seed.
        max_concurrent: Concurrent scenario evaluations.

    Returns:
        The serialised :class:`~agent2model.eval.runner.EvalRunResult`.
    """
    import asyncio

    from agent2model.eval.baselines import make_condition
    from agent2model.eval.judge import Judge, JudgeConfig
    from agent2model.eval.report import write_json_report, write_pdf_report
    from agent2model.eval.runner import EvalConfig, EvalRunner
    from agent2model.exceptions import EvalError
    from agent2model.generation.generator import DEFAULT_MODEL
    from agent2model.ir.loader import load_flowchart_from_string
    from agent2model.logging import logger

    _ = model_path  # accepted for pipeline symmetry; eval consumes served_url
    build = _materialise_flowchart(recipe)
    flowchart = load_flowchart_from_string(recipe.flowchart_yaml)

    names = list(baselines)
    if served_url:
        names.append("compiled")
    conditions = [make_condition(name, flowchart, served_url=served_url) for name in names]

    judge_cfg = JudgeConfig(model=judge_model or DEFAULT_MODEL)
    config = EvalConfig(
        n=recipe.eval_n,
        budget_usd=recipe.eval_budget_usd,
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
    logger.info(f"Evaluated {recipe.name}: total cost ${result.total_cost_usd:.4f}.")
    return result.model_dump(mode="json")


class _ServeProxy:
    """Thin proxy that gives :data:`serve` a ``.remote(recipe, model_path)`` API.

    Mirrors the call shape of the other workers (``generate_data.remote(recipe)``,
    ``train_3b.remote(recipe, dataset)``) so the generic ``run`` entrypoint reads
    uniformly. Internally builds the :class:`ServeCls` Modal class with the
    parameters set and calls the underlying nullary ``web_server``-decorated method.
    """

    @staticmethod
    def remote(recipe: Recipe, model_path: str | None = None) -> None:
        """Start the autoscaling serve endpoint for ``recipe``.

        Args:
            recipe: The recipe whose compiled model to serve.
            model_path: Optional explicit checkpoint path; defaults to the
                recipe's on-volume model directory.
        """
        # modal's @app.cls rewrites the class; mypy can't see modal.parameter()
        # kw-args, so silence the call-arg check here.
        instance = ServeCls(  # type: ignore[call-arg]
            recipe_name=recipe.name, model_path=model_path or ""
        )
        instance.run.remote()

    @staticmethod
    def spawn(recipe: Recipe, model_path: str | None = None) -> None:
        """Spawn the serve endpoint asynchronously (does not wait for startup)."""
        instance = ServeCls(  # type: ignore[call-arg]
            recipe_name=recipe.name, model_path=model_path or ""
        )
        instance.run.spawn()


#: Module-level serve handle exposing ``serve.remote(recipe, model_path=None)``.
serve = _ServeProxy()


# --------------------------------------------------------------------------- #
# Generic + reproduction entrypoints                                           #
# --------------------------------------------------------------------------- #


def _load_flowchart_yaml_text(path: Path) -> str:
    """Read a flowchart spec from disk as YAML text.

    A ``.py`` path is converted via the LangGraph adapter; a ``.yaml`` / ``.yml``
    path is read verbatim. Used by the local-entrypoint side of the Modal
    boundary — never runs on a worker.

    Args:
        path: Local filesystem path to a ``.py`` LangGraph file or YAML
            flowchart.

    Returns:
        The YAML text to embed in :attr:`Recipe.flowchart_yaml`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the suffix is not ``.py``, ``.yaml``, or ``.yml``.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such flowchart: {path}")
    suffix = path.suffix.lower()
    if suffix == ".py":
        from agent2model.adapters.langgraph import langgraph_to_yaml_text

        return langgraph_to_yaml_text(path)
    if suffix in {".yaml", ".yml"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported flowchart suffix {path.suffix!r}; expected .yaml, .yml, or .py.")


def build_recipe_from_path(
    flowchart_path: str | Path,
    *,
    name: str | None = None,
    size: str = "3b",
    n: int = 2000,
    epochs: int = 20,
    eval_n: int = 200,
    base_model: str | None = None,
    gen_budget_usd: float = 80.0,
    eval_budget_usd: float = 60.0,
) -> Recipe:
    """Build a :class:`Recipe` from a flowchart on disk and run parameters.

    Pure helper — no modal calls — used by the :func:`run` local entrypoint and
    exercised directly in unit tests. Resolves the path, reads the YAML (via the
    LangGraph adapter for ``.py``), picks the flowchart's own ``name`` if the
    caller did not pass one, and fills the default base model for the chosen
    size.

    Args:
        flowchart_path: Path to a ``.yaml`` / ``.yml`` flowchart or a ``.py``
            LangGraph file.
        name: Optional recipe name; defaults to the YAML's ``name`` field, or
            ``path.stem`` if neither is set.
        size: ``"3b"`` or ``"8b"``.
        n: Number of conversations to generate.
        epochs: Training epochs.
        eval_n: Scenarios per eval condition.
        base_model: Optional HF base model id; defaults to
            :data:`DEFAULT_BASE_FOR_SIZE` for ``size``.
        gen_budget_usd: Hard USD cap for the data-generation step.
        eval_budget_usd: Hard USD cap for the evaluation step.

    Returns:
        A fully-populated :class:`Recipe`.

    Raises:
        ValueError: If ``size`` is not ``"3b"`` / ``"8b"``, or the path suffix
            is not supported.
        FileNotFoundError: If ``flowchart_path`` does not exist.
    """
    from typing import cast

    import yaml as _yaml

    from agent2model.training.config import ModelSize

    size_lc = size.lower()
    if size_lc not in {"3b", "8b"}:
        raise ValueError(f"Unsupported size {size!r}; expected '3b' or '8b'.")
    size_typed = cast(ModelSize, size_lc)

    path = Path(flowchart_path).expanduser().resolve()
    yaml_text = _load_flowchart_yaml_text(path)
    parsed = _yaml.safe_load(yaml_text)
    parsed_name: str | None = None
    if isinstance(parsed, dict):
        candidate = parsed.get("name")
        if isinstance(candidate, str):
            parsed_name = candidate
    flow_name = name or parsed_name or path.stem

    resolved_base = base_model or DEFAULT_BASE_FOR_SIZE[size_typed]

    return Recipe(
        name=flow_name,
        flowchart_yaml=yaml_text,
        size=size_typed,
        base_model=resolved_base,
        n_convos=n,
        epochs=epochs,
        eval_n=eval_n,
        gen_budget_usd=gen_budget_usd,
        eval_budget_usd=eval_budget_usd,
    )


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def run(
    flowchart_path: str,
    name: str | None = None,
    size: str = "3b",
    n: int = 2000,
    epochs: int = 20,
    eval_n: int = 200,
    base_model: str | None = None,
    skip_eval: bool = False,
    serve_after: bool = False,
    yes: bool = False,
) -> None:
    """Generic pipeline entrypoint: generate → train → (optional) evaluate → (optional) serve.

    Reads the flowchart from disk (YAML or LangGraph ``.py``), embeds the YAML
    into a :class:`Recipe`, and chains the worker functions on Modal. Outputs
    land under ``recipe.name`` on the build / model volumes.

    Run with::

        modal run -m agent2model.cloud.modal_app::run -- \\
            --flowchart-path my_workflow.yaml --size 3b --n 2000 --epochs 20

    Args:
        flowchart_path: Path to a ``.yaml`` / ``.yml`` flowchart or LangGraph
            ``.py``.
        name: Recipe name; defaults to the YAML's ``name`` (or file stem).
        size: ``"3b"`` or ``"8b"`` training path.
        n: Number of conversations to generate.
        epochs: Training epochs.
        eval_n: Scenarios per eval condition.
        base_model: HF base model id; defaults to the size's preset.
        skip_eval: If True, skip the evaluation step.
        serve_after: If True, also launch the autoscaling serve endpoint after
            training (and eval, if not skipped).
        yes: If True, skip the interactive cost-confirmation prompt. Required
            for non-interactive (CI) invocations.
    """
    from agent2model.cloud._costs import confirm_cost_or_exit
    from agent2model.logging import logger

    recipe = build_recipe_from_path(
        flowchart_path,
        name=name,
        size=size,
        n=n,
        epochs=epochs,
        eval_n=eval_n,
        base_model=base_model,
    )
    logger.info(
        f"Running {recipe.name} ({recipe.size}) on Modal: "
        f"{recipe.n_convos} convos, {recipe.epochs} epochs."
    )
    confirm_cost_or_exit(recipe, yes=yes)

    dataset = generate_data.remote(recipe)
    train_fn = {"3b": train_3b, "8b": train_8b}[recipe.size]
    model = train_fn.remote(recipe, dataset)
    if not skip_eval:
        scores = evaluate.remote(recipe, model)
        print(f"[{recipe.name}] eval scores: {scores}")
    if serve_after:
        serve.remote(recipe, model)


def _reproduce(recipe: Recipe, *, yes: bool = False) -> dict[str, Any]:
    """Chain generate -> train -> evaluate for one paper-reproduction recipe.

    Dispatches training to :func:`train_3b` or :func:`train_8b` based on the
    recipe's model size, then runs evaluation against the ``in_context`` upper
    bound. Each step is a remote Modal call (``.remote``). Prints a cost
    estimate and prompts to continue unless ``yes`` is True.

    Args:
        recipe: The reproduction recipe.
        yes: If True, skip the interactive cost-confirmation prompt.

    Returns:
        The serialised evaluation result for the run.
    """
    from agent2model.cloud._costs import confirm_cost_or_exit
    from agent2model.logging import logger

    logger.info(f"Reproducing {recipe.name} ({recipe.size}) end to end on Modal.")
    confirm_cost_or_exit(recipe, yes=yes)

    dataset = generate_data.remote(recipe)
    trainer = train_3b if recipe.size == "3b" else train_8b
    model = trainer.remote(recipe, dataset)
    return evaluate.remote(  # type: ignore[no-any-return]
        recipe,
        model,
        baselines=("in_context",),
    )


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_travel(yes: bool = False) -> None:
    """Reproduce the Travel experiment (Qwen2.5-3B, ~2000 convos, 20 epochs).

    Run with ``modal run -m agent2model.cloud.modal_app::reproduce_travel``.

    Args:
        yes: If True, skip the interactive cost-confirmation prompt.
    """
    _reproduce(get_recipe("travel"), yes=yes)


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_zoom(yes: bool = False) -> None:
    """Reproduce the Zoom experiment (Qwen3-8B, ~6000 convos, 10 epochs).

    Run with ``modal run -m agent2model.cloud.modal_app::reproduce_zoom``.

    Args:
        yes: If True, skip the interactive cost-confirmation prompt.
    """
    _reproduce(get_recipe("zoom"), yes=yes)


@app.local_entrypoint()  # type: ignore[untyped-decorator]  # modal decorators are Any
def reproduce_insurance(yes: bool = False) -> None:
    """Reproduce the Insurance experiment (Qwen3-8B, 55 nodes, ~3000 convos, 20 epochs).

    Run with ``modal run -m agent2model.cloud.modal_app::reproduce_insurance``.

    Args:
        yes: If True, skip the interactive cost-confirmation prompt.
    """
    _reproduce(get_recipe("insurance"), yes=yes)


# Re-export for users who want to import the backward-compat alias from the
# modal module rather than the recipes module.
__all__ = [
    "APP_NAME",
    "EXAMPLES",
    "ExampleRecipe",
    "Recipe",
    "ServeCls",
    "app",
    "build_recipe_from_path",
    "evaluate",
    "generate_data",
    "reproduce_insurance",
    "reproduce_travel",
    "reproduce_zoom",
    "run",
    "serve",
    "train_3b",
    "train_8b",
]
