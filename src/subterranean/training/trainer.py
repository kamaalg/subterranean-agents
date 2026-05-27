"""Wrap TRL's ``SFTTrainer`` with the paper's full fine-tuning recipe.

This module is import-safe on machines without the heavy ML stack: every
``torch``/``trl``/``transformers``/``datasets`` import is performed lazily inside
the functions that need it, so the module imports cleanly with only the core
dependencies installed (unit tests and the CLI ``--help`` path rely on this).

The orchestration logic is factored into small, pure, unit-testable functions:

* :func:`split_dataset` — deterministic 90/10 train/eval partition.
* :func:`select_best_checkpoint` — pick the checkpoint with the lowest held-out
  eval loss.
* :func:`is_diverged` — detect NaN/Inf eval losses (raises
  :class:`~subterranean.exceptions.TrainingDivergedError` via :func:`check_divergence`).
* :func:`reject_lora` — refuse LoRA with a link to Dennis et al. 2026b.

Only :func:`train` touches the GPU stack; it is exercised end-to-end on Modal
(Phase 7), not in the local test suite.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from subterranean.exceptions import TrainingDivergedError
from subterranean.logging import logger
from subterranean.training.config import DENNIS_2026B, TrainingConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

Record = dict[str, Any]
"""One dataset row, e.g. ``{"messages": [...]}``."""


class CheckpointInfo(BaseModel):
    """A training checkpoint and its held-out eval loss.

    Attributes:
        step: Global step the checkpoint was saved at.
        path: Filesystem path to the checkpoint directory.
        eval_loss: Held-out evaluation loss; ``None`` if it was never evaluated.
    """

    model_config = ConfigDict(extra="forbid")

    step: int
    path: str
    eval_loss: float | None = None


def reject_lora(*, use_lora: bool) -> None:
    """Refuse any attempt to enable LoRA.

    Args:
        use_lora: Whether the caller asked for LoRA / PEFT.

    Raises:
        ValueError: Always, when ``use_lora`` is True, with a link to the
            companion paper justifying full fine-tuning.

    Example:
        >>> reject_lora(use_lora=False)
        >>> reject_lora(use_lora=True)
        Traceback (most recent call last):
        ...
        ValueError: LoRA is not supported in subterranean v1...
    """
    if use_lora:
        raise ValueError(
            "LoRA is not supported in subterranean v1: it fails to internalise "
            f"procedural workflows. See {DENNIS_2026B}. Use full fine-tuning instead."
        )


def load_jsonl(path: str | Path) -> list[Record]:
    """Load a JSONL dataset into a list of records.

    Args:
        path: Path to a ``.jsonl`` file (one JSON object per line).

    Returns:
        The parsed records, in file order. Blank lines are skipped.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")
    records: list[Record] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def split_dataset(
    records: Sequence[Record], eval_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    """Deterministically partition records into train/eval sets.

    The split is a seeded shuffle followed by a slice, so it is reproducible for
    a given ``seed`` and the two halves never overlap. At least one record is
    always assigned to eval when there are two or more records.

    Args:
        records: The full dataset.
        eval_fraction: Fraction held out for evaluation (e.g. ``0.10``).
        seed: RNG seed controlling the shuffle.

    Returns:
        A ``(train, eval)`` tuple.

    Raises:
        ValueError: If ``eval_fraction`` is not strictly between 0 and 1.

    Example:
        >>> train, ev = split_dataset([{"i": i} for i in range(10)], 0.1, seed=0)
        >>> len(train), len(ev)
        (9, 1)
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_eval = round(n * eval_fraction)
    if n >= 2:
        n_eval = max(1, min(n_eval, n - 1))
    eval_set = items[:n_eval]
    train_set = items[n_eval:]
    return train_set, eval_set


def is_diverged(eval_loss: float | None) -> bool:
    """Return True if an eval loss indicates divergence (NaN or Inf).

    Args:
        eval_loss: A logged eval loss, or ``None`` if unavailable.

    Returns:
        True for NaN/Inf, False otherwise (including ``None``).

    Example:
        >>> is_diverged(float("nan")), is_diverged(0.5), is_diverged(None)
        (True, False, False)
    """
    if eval_loss is None:
        return False
    return math.isnan(eval_loss) or math.isinf(eval_loss)


def check_divergence(checkpoints: Sequence[CheckpointInfo]) -> None:
    """Raise if any checkpoint recorded a NaN/Inf eval loss.

    Args:
        checkpoints: Checkpoints collected during training.

    Raises:
        TrainingDivergedError: If any eval loss is NaN/Inf.
    """
    for ckpt in checkpoints:
        if is_diverged(ckpt.eval_loss):
            raise TrainingDivergedError(
                f"Training diverged: eval loss {ckpt.eval_loss} at step {ckpt.step}. "
                "Lower the learning rate or check the dataset."
            )


def select_best_checkpoint(checkpoints: Sequence[CheckpointInfo]) -> CheckpointInfo:
    """Select the checkpoint with the lowest held-out eval loss.

    Checkpoints without an eval loss (or with a NaN/Inf one) are ignored. Ties
    are broken by preferring the earlier step, matching the paper's observation
    that the best checkpoint arrives early (~epoch 4 for 3B, ~epoch 2 for 8B).

    Args:
        checkpoints: Candidate checkpoints.

    Returns:
        The best :class:`CheckpointInfo`.

    Raises:
        ValueError: If there are no checkpoints with a finite eval loss.

    Example:
        >>> a = CheckpointInfo(step=1, path="a", eval_loss=0.9)
        >>> b = CheckpointInfo(step=2, path="b", eval_loss=0.4)
        >>> select_best_checkpoint([a, b]).path
        'b'
    """
    finite = [c for c in checkpoints if c.eval_loss is not None and not is_diverged(c.eval_loss)]
    if not finite:
        raise ValueError("No checkpoints with a finite eval loss to select from.")

    def _key(c: CheckpointInfo) -> tuple[float, int]:
        # ``finite`` only holds checkpoints with a non-None eval loss.
        assert c.eval_loss is not None
        return (c.eval_loss, c.step)

    return min(finite, key=_key)


def _checkpoints_from_log_history(log_history: list[dict[str, Any]]) -> list[CheckpointInfo]:
    """Build :class:`CheckpointInfo` records from a transformers log history.

    The HF ``Trainer`` log history interleaves train and eval entries; eval
    entries carry ``eval_loss``. We pair each eval loss with the checkpoint dir
    written at that step (``checkpoint-<step>``).
    """
    out: list[CheckpointInfo] = []
    for entry in log_history:
        if "eval_loss" in entry:
            step = int(entry.get("step", 0))
            out.append(
                CheckpointInfo(
                    step=step,
                    path=f"checkpoint-{step}",
                    eval_loss=float(entry["eval_loss"]),
                )
            )
    return out


def _require_train_deps() -> None:
    """Verify the heavy ML stack is importable, else raise a helpful error.

    Raises:
        RuntimeError: If ``trl``/``torch`` are not importable, with an actionable
            install hint.
    """
    try:
        import torch  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "Training requires the optional ML stack, which is not installed. "
            "Install it with `pip install -e '.[train]'` and run on a GPU host "
            "(DeepSpeed/bitsandbytes do not build on macOS and there is no GPU here)."
        ) from exc


def train(config: TrainingConfig, dataset_path: str | Path) -> CheckpointInfo:
    """Fine-tune ``config.base_model`` on a chat-template JSONL dataset.

    Lazily imports the ML stack, loads the dataset, performs a deterministic
    90/10 train/eval split, wraps TRL's ``SFTTrainer`` with the paper's recipe
    (passing ``config.extra_args`` through to ``SFTConfig``), trains while saving
    every checkpoint, detects divergence, and copies the best checkpoint (lowest
    held-out eval loss) to ``config.output_dir``.

    Args:
        config: The training configuration (a size preset, typically).
        dataset_path: Path to ``build/<name>/dataset.jsonl``.

    Returns:
        The :class:`CheckpointInfo` for the selected best checkpoint.

    Raises:
        RuntimeError: If the ML stack is not installed (with an install hint).
        TrainingDivergedError: If eval loss goes NaN/Inf during training.

    Example:
        >>> # Runs on a GPU host with the [train] extra installed:
        >>> # train(TrainingConfig.for_3b("build/travel/model"), "build/travel/dataset.jsonl")
    """
    reject_lora(use_lora=config.use_lora)
    _require_train_deps()

    # Lazy heavy imports — never executed at module import time.
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    records = load_jsonl(dataset_path)
    train_records, eval_records = split_dataset(records, config.eval_split, config.seed)
    logger.info(
        f"Training {config.base_model} ({config.size}): "
        f"{len(train_records)} train / {len(eval_records)} eval examples, "
        f"{config.epochs} epochs, effective batch size {config.effective_batch_size}."
    )

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    sft_config = SFTConfig(**config.to_sft_kwargs())
    trainer = SFTTrainer(
        model=config.base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()

    log_history: list[dict[str, Any]] = list(trainer.state.log_history)
    checkpoints = _checkpoints_from_log_history(log_history)
    # Resolve checkpoint dirs relative to the trainer output dir.
    out_root = Path(config.output_dir)
    checkpoints = [
        ckpt.model_copy(update={"path": str(out_root / ckpt.path)}) for ckpt in checkpoints
    ]

    check_divergence(checkpoints)
    best = select_best_checkpoint(checkpoints)
    logger.info(f"Best checkpoint: {best.path} (eval_loss={best.eval_loss}).")

    best_dir = out_root / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best.path, best_dir)
    logger.info(f"Copied best checkpoint to {best_dir}.")
    return best.model_copy(update={"path": str(best_dir)})
