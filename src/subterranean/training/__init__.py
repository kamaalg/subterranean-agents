"""Fine-tuning pipeline (Phase 4). Full-parameter SFT only; no LoRA in v1.

Heavy ML dependencies (``torch``/``trl``/``transformers``/``datasets``) are
imported lazily inside :func:`subterranean.training.trainer.train`, so importing
this package is safe on machines without the GPU training stack.
"""

from __future__ import annotations

from subterranean.training.config import TrainingConfig

__all__ = ["TrainingConfig"]
