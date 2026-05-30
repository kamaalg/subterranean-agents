"""Distributed training entry point run under ``accelerate launch``.

This module exists so the 8B DeepSpeed ZeRO-3 path can be launched as a real
multi-process job:

    accelerate launch --use_deepspeed --deepspeed_config_file zero3.json \\
        -m agent2model.training._entry --config training_config.json \\
        --dataset dataset.jsonl

It reconstructs the :class:`~agent2model.training.config.TrainingConfig` from the
JSON file written by :func:`agent2model.training.launch.launch_training`, runs
:func:`agent2model.training.trainer.train` (which the launcher cannot do
in-process for 8B), and — on the main process only — writes the selected
checkpoint's metadata to ``<output_dir>/best.json`` so the launching process can
read it back.

It is never imported at package import time; only the 8B subprocess executes it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent2model.logging import configure_logging, logger
from agent2model.training.config import TrainingConfig
from agent2model.training.trainer import _is_main_process, train


def main() -> None:
    """Parse ``--config``/``--dataset`` and run distributed training."""
    parser = argparse.ArgumentParser(description="agent2model distributed training entry")
    parser.add_argument("--config", required=True, help="Path to TrainingConfig JSON.")
    parser.add_argument("--dataset", required=True, help="Path to the chat-template JSONL.")
    args = parser.parse_args()

    configure_logging()
    config = TrainingConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    best = train(config, args.dataset)

    if _is_main_process():
        best_meta = Path(config.output_dir) / "best.json"
        best_meta.write_text(best.model_dump_json(), encoding="utf-8")
        logger.info(f"Wrote best-checkpoint metadata to {best_meta}.")


if __name__ == "__main__":
    main()
