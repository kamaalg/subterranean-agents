"""DeepSpeed configs for the 8B training path.

The 8B preset (``Qwen/Qwen3-8B``) trains across 8x A100 80GB with DeepSpeed
ZeRO-3 (parameters, gradients, and optimizer state sharded across GPUs). The
JSON config :mod:`agent2model.training.deepspeed.zero3` (file ``zero3.json``)
holds bf16 + ZeRO stage 3 defaults; most numeric fields are ``"auto"`` so
``accelerate``/transformers fill them from the ``SFTConfig`` at launch. The 3B
preset runs on a single GPU and does not use DeepSpeed.
"""

from __future__ import annotations

from pathlib import Path

ZERO3_CONFIG_PATH = Path(__file__).parent / "zero3.json"
"""Absolute path to the ZeRO-3 JSON config for the 8B path."""

__all__ = ["ZERO3_CONFIG_PATH"]
