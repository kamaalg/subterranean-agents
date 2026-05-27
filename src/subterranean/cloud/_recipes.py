"""Pure, modal-free recipe logic for the cloud deployment phase (Phase 7).

This module is deliberately **free of any ``modal`` import** so it can be unit
tested and reasoned about without the cloud SDK installed. ``modal_app.py``
consumes everything here to wire up the real Modal ``App``, while the choices it
encodes — which image/GPU/training-config a given paper example uses, and the
ordered pipeline steps a reproduction runs — live here as small, typed functions.

The reproduction presets mirror the paper (arXiv:2605.22502v1) per-example
training setup:

============  ==================  =====  =======  =========================
Example       Base model          Path   Convos   Epochs / hardware
============  ==================  =====  =======  =========================
travel        Qwen2.5-3B-Instruct 3B     ~2,125   20 / single A10G-or-A100
zoom          Qwen3-8B            8B     ~6,264   10 / 8x A100 ZeRO-3
insurance     Qwen3-8B (55 nodes) 8B     ~3,000   20 / 8x A100 ZeRO-3
============  ==================  =====  =======  =========================
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from subterranean.training.config import (
    DEFAULT_3B_MODEL,
    DEFAULT_8B_MODEL,
    ModelSize,
    TrainingConfig,
)

__all__ = [
    "EXAMPLES",
    "GPU_3B",
    "GPU_8B",
    "PIPELINE_STEPS",
    "ExampleRecipe",
    "PipelineStep",
    "build_training_config",
    "get_recipe",
    "gpu_for_size",
    "image_kind_for_step",
    "n_gpus_for_size",
    "pipeline_steps",
    "training_function_for_size",
]

#: GPU spec strings, in Modal's ``gpu=`` notation, for each training path.
GPU_3B = "A10G"
"""Single-GPU spec for the 3B path (an A100 also satisfies it; A10G is cheaper)."""

GPU_8B = "A100-80GB:8"
"""8x A100 80GB for the 8B DeepSpeed ZeRO-3 path."""

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


class ExampleRecipe(BaseModel):
    """The per-example reproduction recipe for one paper experiment.

    Attributes:
        name: Example id (``travel`` / ``zoom`` / ``insurance``).
        size: Model-size path (``"3b"`` or ``"8b"``).
        base_model: HF base model id fine-tuned for this example.
        n_convos: Number of synthetic conversations to generate (paper volume).
        epochs: Training epochs (paper recipe).
        eval_n: Scenarios per condition for the evaluation pass.
        gen_budget_usd: Hard USD cap for the data-generation step.
        eval_budget_usd: Hard USD cap for the evaluation step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    size: ModelSize
    base_model: str
    n_convos: int = Field(gt=0)
    epochs: int = Field(gt=0)
    eval_n: int = Field(default=200, gt=0)
    gen_budget_usd: float = Field(default=80.0, gt=0.0)
    eval_budget_usd: float = Field(default=60.0, gt=0.0)


#: The three paper reproductions, keyed by example name. Sizes/epochs/volumes
#: match arXiv:2605.22502v1's per-example training setup.
EXAMPLES: dict[str, ExampleRecipe] = {
    "travel": ExampleRecipe(
        name="travel",
        size="3b",
        base_model=DEFAULT_3B_MODEL,
        n_convos=2000,
        epochs=20,
        eval_n=200,
        gen_budget_usd=60.0,
    ),
    "zoom": ExampleRecipe(
        name="zoom",
        size="8b",
        base_model=DEFAULT_8B_MODEL,
        n_convos=6000,
        epochs=10,
        eval_n=200,
        gen_budget_usd=120.0,
    ),
    "insurance": ExampleRecipe(
        name="insurance",
        size="8b",
        base_model=DEFAULT_8B_MODEL,
        n_convos=3000,
        epochs=20,
        eval_n=200,
        gen_budget_usd=90.0,
    ),
}


def get_recipe(example: str) -> ExampleRecipe:
    """Return the reproduction recipe for a paper example.

    Args:
        example: Example name — one of ``travel``, ``zoom``, ``insurance``.

    Returns:
        The :class:`ExampleRecipe` for that example.

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


def build_training_config(recipe: ExampleRecipe, output_dir: str) -> TrainingConfig:
    """Build the :class:`TrainingConfig` for a reproduction recipe.

    Dispatches to :meth:`TrainingConfig.for_3b` or :meth:`TrainingConfig.for_8b`
    (the 8B variant carries the DeepSpeed ZeRO-3 multi-GPU defaults), threading
    the example's base model and epoch count through.

    Args:
        recipe: The example recipe.
        output_dir: Directory the trainer writes checkpoints to.

    Returns:
        A configured :class:`TrainingConfig` matching the paper's per-example recipe.

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
