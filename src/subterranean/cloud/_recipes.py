"""Pure, modal-free recipe logic for the cloud deployment phase (Phase 7).

This module is deliberately **free of any ``modal`` import** so it can be unit
tested and reasoned about without the cloud SDK installed. ``modal_app.py``
consumes everything here to wire up the real Modal ``App``, while the choices it
encodes — which image/GPU/training-config a given recipe uses, and the ordered
pipeline steps a reproduction runs — live here as small, typed functions.

The reproduction presets mirror the paper (arXiv:2605.22502v1) per-example
training setup:

============  ==================  =====  =======  =========================
Example       Base model          Path   Convos   Epochs / hardware
============  ==================  =====  =======  =========================
travel        Qwen2.5-3B-Instruct 3B     ~2,125   20 / single A10G-or-A100
zoom          Qwen3-8B            8B     ~6,264   10 / 8x A100 ZeRO-3
insurance     Qwen3-8B (55 nodes) 8B     ~3,000   20 / 8x A100 ZeRO-3
============  ==================  =====  =======  =========================

A :class:`Recipe` carries the flowchart YAML text **inline** (``flowchart_yaml``)
rather than a path. This is what makes the generic ``run`` Modal entrypoint
possible: Modal containers don't see the user's local filesystem, so the spec
must travel over the wire with the recipe. The paper :data:`EXAMPLES` recipes
load their YAMLs from the repo's ``examples/<dir>/flowchart.yaml`` at import
time and embed that text into the recipe, so the three ``reproduce_*`` Modal
entrypoints work the same way the generic ``run`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from subterranean.training.config import (
    DEFAULT_3B_MODEL,
    DEFAULT_8B_MODEL,
    ModelSize,
    TrainingConfig,
)

__all__ = [
    "DEFAULT_BASE_FOR_SIZE",
    "EXAMPLES",
    "EXAMPLE_FLOWCHART_DIRS",
    "GPU_3B",
    "GPU_8B",
    "PIPELINE_STEPS",
    "ExampleRecipe",
    "PipelineStep",
    "Recipe",
    "build_training_config",
    "get_recipe",
    "gpu_for_size",
    "image_kind_for_step",
    "n_gpus_for_size",
    "pipeline_steps",
    "training_function_for_size",
]

#: GPU spec strings, in Modal's ``gpu=`` notation, for each training path.
GPU_3B = "A100-80GB"
"""Single-GPU spec for the 3B path.

Even A100-40GB OOMs on Qwen 2.5 3B with full-param bf16 + 8-bit AdamW + grad
checkpointing — bitsandbytes' 8-bit AdamW still keeps fp32 master weights
(~12 GB on its own), and with the 152K-vocab cross-entropy logits the total
sits at ~38-40 GB right at the failing forward pass. A100-80GB has clean
headroom. Cost diff vs 40GB on a 3.5h run is ~$5; safer than fighting OOM.
"""

GPU_8B = "A100-80GB:8"
"""8x A100 80GB for the 8B DeepSpeed ZeRO-3 path."""

#: Default HF base model id per training-path preset. Used by the generic
#: ``run`` Modal entrypoint and CLI when the user does not pass ``--base-model``.
DEFAULT_BASE_FOR_SIZE: dict[ModelSize, str] = {
    "3b": DEFAULT_3B_MODEL,
    "8b": DEFAULT_8B_MODEL,
}

ImageKind = Literal["cpu", "train", "serve"]
"""Which base image a pipeline step needs.

* ``cpu`` — core + anthropic only; the API-bound data-generation step.
* ``train`` — the ``[train]`` extra (torch/trl/deepspeed); fine-tuning.
* ``serve`` — the ``[serve]`` extra (vLLM); the inference endpoint.
"""

PipelineStep = Literal["generate", "train", "evaluate"]
"""An ordered step in a reproduction pipeline."""

#: The fixed end-to-end reproduction pipeline: generate, then train, then evaluate.
PIPELINE_STEPS: tuple[PipelineStep, ...] = ("generate", "train", "evaluate")


class Recipe(BaseModel):
    """A self-contained recipe for one Modal pipeline run.

    The recipe carries the flowchart YAML text inline (``flowchart_yaml``) so it
    can be shipped to a Modal worker that has no view of the caller's local
    filesystem. Two users with the same ``name`` will collide on the build/model
    volumes — last-write-wins is fine for v1.

    Attributes:
        name: Recipe id, used as the subdir under the build/model volumes
            (e.g. ``travel``).
        flowchart_yaml: The full flowchart YAML as a string. Persisted to
            ``<build>/flowchart.yaml`` by the worker before generation.
        size: Model-size path (``"3b"`` or ``"8b"``).
        base_model: HF base model id fine-tuned for this recipe.
        n_convos: Number of synthetic conversations to generate.
        epochs: Training epochs.
        eval_n: Scenarios per condition for the evaluation pass.
        gen_budget_usd: Hard USD cap for the data-generation step.
        eval_budget_usd: Hard USD cap for the evaluation step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    flowchart_yaml: str = Field(min_length=1)
    size: ModelSize
    base_model: str
    n_convos: int = Field(gt=0)
    epochs: int = Field(gt=0)
    eval_n: int = Field(default=200, gt=0)
    gen_budget_usd: float = Field(default=80.0, gt=0.0)
    eval_budget_usd: float = Field(default=60.0, gt=0.0)


#: Backward-compatible alias. ``Recipe`` replaces the old ``ExampleRecipe``
#: class; the alias is kept so tests and downstream code that imported the old
#: name keep working.
ExampleRecipe = Recipe

#: Mapping from a paper-example recipe id to its in-repo flowchart directory.
EXAMPLE_FLOWCHART_DIRS: dict[str, str] = {
    "travel": "travel_booking",
    "zoom": "zoom_support",
    "insurance": "insurance_claims",
}

# Resolve the repo's examples directory relative to this file. The package
# layout is ``<repo>/src/subterranean/cloud/_recipes.py``, with examples at
# ``<repo>/examples/<dir>/flowchart.yaml``. When the package is installed (e.g.
# into a Modal image) the ``examples`` directory is not shipped — we therefore
# read the YAML eagerly here, so the bytes travel with the recipe.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _load_example_flowchart_yaml(example: str) -> str:
    """Read the in-repo flowchart YAML for a paper example.

    Falls back to a tiny placeholder if the ``examples/`` directory is not on
    disk (e.g. when the package is installed from a wheel that does not ship the
    examples). The :data:`EXAMPLES` recipes are still constructed; importing
    ``subterranean.cloud._recipes`` never fails just because the YAML is absent.

    Args:
        example: Recipe id (key of :data:`EXAMPLE_FLOWCHART_DIRS`).

    Returns:
        The YAML text.
    """
    sub = EXAMPLE_FLOWCHART_DIRS[example]
    path = _EXAMPLES_DIR / sub / "flowchart.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Placeholder so the Recipe still validates (flowchart_yaml is non-empty).
    # The Modal workers will write this YAML to the build volume; if the user
    # is running a paper reproduction from a wheel install they should clone
    # the repo or supply their own flowchart via the generic `run` entrypoint.
    return f"name: {example}\nstart: placeholder\nnodes:\n  placeholder:\n    terminal: success\n"


#: The three paper reproductions, keyed by recipe id. Sizes/epochs/volumes
#: match arXiv:2605.22502v1's per-example training setup. Each recipe embeds
#: the corresponding ``examples/<dir>/flowchart.yaml`` text inline.
EXAMPLES: dict[str, Recipe] = {
    "travel": Recipe(
        name="travel",
        flowchart_yaml=_load_example_flowchart_yaml("travel"),
        size="3b",
        base_model=DEFAULT_3B_MODEL,
        n_convos=2000,
        epochs=20,
        eval_n=200,
        gen_budget_usd=60.0,
    ),
    "zoom": Recipe(
        name="zoom",
        flowchart_yaml=_load_example_flowchart_yaml("zoom"),
        size="8b",
        base_model=DEFAULT_8B_MODEL,
        n_convos=6000,
        epochs=10,
        eval_n=200,
        gen_budget_usd=120.0,
    ),
    "insurance": Recipe(
        name="insurance",
        flowchart_yaml=_load_example_flowchart_yaml("insurance"),
        size="8b",
        base_model=DEFAULT_8B_MODEL,
        n_convos=3000,
        epochs=20,
        eval_n=200,
        gen_budget_usd=90.0,
    ),
}


def get_recipe(example: str) -> Recipe:
    """Return the reproduction recipe for a paper example.

    Args:
        example: Recipe id — one of ``travel``, ``zoom``, ``insurance``.

    Returns:
        The :class:`Recipe` for that example.

    Raises:
        KeyError: If ``example`` is not a known reproduction.

    Example:
        >>> get_recipe("travel").size
        '3b'
        >>> get_recipe("travel").epochs
        20
    """
    try:
        return EXAMPLES[example]
    except KeyError as exc:
        known = ", ".join(sorted(EXAMPLES))
        raise KeyError(f"Unknown example {example!r}; known examples: {known}.") from exc


def gpu_for_size(size: ModelSize) -> str:
    """Return the Modal ``gpu=`` spec string for a model-size path.

    Args:
        size: ``"3b"`` (single GPU) or ``"8b"`` (8x A100 ZeRO-3).

    Returns:
        The GPU spec string (e.g. ``"A10G"`` or ``"A100-80GB:8"``).

    Raises:
        ValueError: If ``size`` is not a supported path.

    Example:
        >>> gpu_for_size("3b"), gpu_for_size("8b")
        ('A10G', 'A100-80GB:8')
    """
    if size == "3b":
        return GPU_3B
    if size == "8b":
        return GPU_8B
    raise ValueError(f"Unsupported model size: {size!r}")


def n_gpus_for_size(size: ModelSize) -> int:
    """Return the GPU count for a model-size path.

    Args:
        size: ``"3b"`` or ``"8b"``.

    Returns:
        ``1`` for the 3B path, ``8`` for the 8B ZeRO-3 path.

    Example:
        >>> n_gpus_for_size("3b"), n_gpus_for_size("8b")
        (1, 8)
    """
    return 1 if size == "3b" else 8


def training_function_for_size(size: ModelSize) -> str:
    """Return the name of the Modal training function for a model-size path.

    Args:
        size: ``"3b"`` or ``"8b"``.

    Returns:
        ``"train_3b"`` or ``"train_8b"`` — the Modal function to dispatch to.

    Example:
        >>> training_function_for_size("8b")
        'train_8b'
    """
    return f"train_{size}"


def build_training_config(recipe: Recipe, output_dir: str) -> TrainingConfig:
    """Build the :class:`TrainingConfig` for a recipe.

    Dispatches to :meth:`TrainingConfig.for_3b` or :meth:`TrainingConfig.for_8b`
    (the 8B variant carries the DeepSpeed ZeRO-3 multi-GPU defaults), threading
    the recipe's base model and epoch count through.

    Args:
        recipe: The recipe.
        output_dir: Directory the trainer writes checkpoints to.

    Returns:
        A configured :class:`TrainingConfig` matching the recipe.

    Example:
        >>> cfg = build_training_config(get_recipe("travel"), "/vol/travel/model")
        >>> cfg.size, cfg.epochs, cfg.effective_batch_size
        ('3b', 20, 16)
        >>> cfg8 = build_training_config(get_recipe("zoom"), "/vol/zoom/model")
        >>> cfg8.size, cfg8.num_gpus, cfg8.epochs
        ('8b', 8, 10)
    """
    if recipe.size == "3b":
        return TrainingConfig.for_3b(output_dir, base_model=recipe.base_model, epochs=recipe.epochs)
    return TrainingConfig.for_8b(
        output_dir,
        base_model=recipe.base_model,
        epochs=recipe.epochs,
        num_gpus=n_gpus_for_size(recipe.size),
    )


def image_kind_for_step(step: PipelineStep) -> ImageKind:
    """Return which base image a pipeline step needs.

    The data-generation step is API-bound and runs on a CPU image; training needs
    the ``[train]`` extra; evaluation is also API-bound (it judges via Anthropic)
    and runs on the CPU image.

    Args:
        step: A pipeline step (``generate`` / ``train`` / ``evaluate``).

    Returns:
        The image kind (``"cpu"`` / ``"train"`` / ``"serve"``).

    Raises:
        ValueError: If ``step`` is not a known pipeline step.

    Example:
        >>> image_kind_for_step("generate"), image_kind_for_step("train")
        ('cpu', 'train')
        >>> image_kind_for_step("evaluate")
        'cpu'
    """
    mapping: dict[PipelineStep, ImageKind] = {
        "generate": "cpu",
        "train": "train",
        "evaluate": "cpu",
    }
    try:
        return mapping[step]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline step: {step!r}") from exc


def pipeline_steps() -> tuple[PipelineStep, ...]:
    """Return the ordered reproduction pipeline steps.

    Returns:
        ``("generate", "train", "evaluate")`` — the fixed end-to-end sequence each
        ``reproduce_*`` entrypoint chains.

    Example:
        >>> pipeline_steps()
        ('generate', 'train', 'evaluate')
    """
    return PIPELINE_STEPS
