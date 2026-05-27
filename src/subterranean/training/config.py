"""Training configuration capturing the paper's full fine-tuning recipe.

The :class:`TrainingConfig` mirrors the hyperparameters in CLAUDE.md's
"Training recipe — match the paper" table for the two supported model sizes:

================  ============================  =================
Setting           3B                            8B
================  ============================  =================
Base model        Qwen/Qwen2.5-3B-Instruct      Qwen/Qwen3-8B
Precision         bf16                          bf16
Learning rate     2e-5 (cosine decay)           2e-5 (cosine decay)
Optimizer         adamw_8bit                    adamw_torch (DeepSpeed)
Effective batch   16                            32
Epochs            20                            10
================  ============================  =================

The config is a thin, ergonomic layer over TRL's ``SFTConfig``: the values it
carries map onto ``SFTConfig`` fields (see :meth:`TrainingConfig.to_sft_kwargs`),
and the ``extra_args`` passthrough lets power users set any ``SFTConfig`` field
the preset does not surface. **There are deliberately no LoRA fields** — the
paper's companion (Dennis et al. 2026b) shows LoRA fails to internalise
procedures, so v1 ships full fine-tuning only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelSize = Literal["3b", "8b"]
"""Supported model-size presets."""

DENNIS_2026B = "Dennis et al. 2026b, 'Procedural Knowledge is Not Low-Rank' (arXiv companion)"
"""Citation for the no-LoRA decision, surfaced in refusal messages."""

DEFAULT_3B_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_8B_MODEL = "Qwen/Qwen3-8B"


class TrainingConfig(BaseModel):
    """Full-parameter SFT configuration matching the paper's recipe.

    Attributes:
        base_model: HF model id to fine-tune (e.g. ``Qwen/Qwen2.5-3B-Instruct``).
        size: Which preset this config was built from (``"3b"`` or ``"8b"``).
        output_dir: Directory the best checkpoint is copied to.
        bf16: Use bfloat16 precision (always True for the paper recipe).
        learning_rate: Peak learning rate (2e-5 in the paper).
        lr_scheduler_type: LR schedule; cosine decay per the paper.
        warmup_ratio: Fraction of steps spent warming up the LR.
        per_device_train_batch_size: Micro-batch per GPU.
        gradient_accumulation_steps: Steps accumulated before an optimizer step.
        num_gpus: GPUs the effective batch size is computed against.
        epochs: Number of training epochs.
        optim: Optimizer name passed to TRL/transformers.
        eval_split: Held-out fraction for eval-loss-based checkpoint selection.
        seed: RNG seed for the train/eval split and trainer.
        max_seq_length: Maximum tokenised sequence length.
        use_lora: Always False; present only to make refusal explicit and typed.
        extra_args: Passthrough mapping merged into the underlying TRL
            ``SFTConfig`` (power-user escape hatch).

    Example:
        >>> cfg = TrainingConfig.for_3b(output_dir="build/travel/model")
        >>> cfg.effective_batch_size
        16
        >>> cfg.epochs
        20
    """

    model_config = ConfigDict(extra="forbid")

    base_model: str
    size: ModelSize
    output_dir: str

    bf16: bool = True
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    num_gpus: int = Field(default=1, ge=1)
    epochs: int = Field(default=20, ge=1)

    optim: str = "adamw_8bit"
    eval_split: float = Field(default=0.10, gt=0.0, lt=1.0)
    seed: int = 0
    max_seq_length: int = Field(default=4096, ge=1)

    use_lora: bool = False
    extra_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_lora(self) -> TrainingConfig:
        """Refuse LoRA at config-construction time."""
        if self.use_lora:
            raise ValueError(
                "LoRA is not supported in subterranean v1: it fails to internalise "
                f"procedural workflows. See {DENNIS_2026B}. Use full fine-tuning instead."
            )
        return self

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = micro-batch x grad-accum x GPUs."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * self.num_gpus

    @classmethod
    def for_3b(
        cls,
        output_dir: str,
        *,
        base_model: str = DEFAULT_3B_MODEL,
        epochs: int = 20,
        **overrides: Any,
    ) -> TrainingConfig:
        """Build the single-GPU 3B preset (effective batch size 16).

        Args:
            output_dir: Where the best checkpoint is copied.
            base_model: HF model id; defaults to ``Qwen/Qwen2.5-3B-Instruct``.
            epochs: Training epochs; the paper uses 20 (best checkpoint ~4).
            **overrides: Any other :class:`TrainingConfig` field to override.

        Returns:
            A configured :class:`TrainingConfig`.

        Example:
            >>> TrainingConfig.for_3b("out").effective_batch_size
            16
        """
        params: dict[str, Any] = {
            "base_model": base_model,
            "size": "3b",
            "output_dir": output_dir,
            "epochs": epochs,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_gpus": 1,
            "optim": "adamw_8bit",
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def for_8b(
        cls,
        output_dir: str,
        *,
        base_model: str = DEFAULT_8B_MODEL,
        epochs: int = 10,
        num_gpus: int = 8,
        **overrides: Any,
    ) -> TrainingConfig:
        """Build the multi-GPU DeepSpeed ZeRO-3 8B preset (effective batch size 32).

        Args:
            output_dir: Where the best checkpoint is copied.
            base_model: HF model id; defaults to ``Qwen/Qwen3-8B``.
            epochs: Training epochs; the paper uses 10 (best checkpoint ~2).
            num_gpus: GPUs for the ZeRO-3 run (paper uses 8x A100 80GB).
            **overrides: Any other :class:`TrainingConfig` field to override.

        Returns:
            A configured :class:`TrainingConfig`.

        Example:
            >>> TrainingConfig.for_8b("out").effective_batch_size
            32
        """
        params: dict[str, Any] = {
            "base_model": base_model,
            "size": "8b",
            "output_dir": output_dir,
            "epochs": epochs,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_gpus": num_gpus,
            "optim": "adamw_torch",
        }
        params.update(overrides)
        return cls(**params)

    def to_sft_kwargs(self) -> dict[str, Any]:
        """Render the keyword arguments for TRL's ``SFTConfig``.

        Maps this config's fields onto the corresponding ``SFTConfig`` fields and
        merges ``extra_args`` last so power users can override anything. The
        trainer is configured to keep every checkpoint and evaluate each epoch so
        the best one can be selected by held-out eval loss afterwards.

        Returns:
            A mapping suitable for ``SFTConfig(**kwargs)``.

        Example:
            >>> kw = TrainingConfig.for_3b("out").to_sft_kwargs()
            >>> kw["num_train_epochs"], kw["bf16"]
            (20, True)
        """
        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "optim": self.optim,
            "bf16": self.bf16,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            # Evaluate + checkpoint every epoch; keep all so we can pick the best.
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": None,
            "logging_steps": 10,
            "report_to": [],
        }
        kwargs.update(self.extra_args)
        return kwargs
