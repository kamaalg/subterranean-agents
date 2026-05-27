"""Build ``accelerate launch`` commands for the 3B and 8B training paths.

These helpers do pure string/argument construction — no subprocess execution —
so they are fully unit-testable without the ML stack or a GPU. The 3B path is a
single-GPU launch; the 8B path adds the DeepSpeed ZeRO-3 config and a multi-GPU
process count.
"""

from __future__ import annotations

from pathlib import Path

from subterranean.training.config import TrainingConfig
from subterranean.training.deepspeed import ZERO3_CONFIG_PATH


def build_accelerate_command(
    config: TrainingConfig,
    train_script: str | Path,
    *,
    deepspeed_config: str | Path | None = None,
) -> list[str]:
    """Build the ``accelerate launch`` argv for a training run.

    The 3B preset launches a single process on one GPU. The 8B preset launches
    ``config.num_gpus`` processes wired to the DeepSpeed ZeRO-3 config.

    Args:
        config: The training configuration (its ``size``/``num_gpus`` drive the
            launch topology).
        train_script: Path to the Python entry point ``accelerate`` should run.
        deepspeed_config: Override DeepSpeed config path; defaults to the bundled
            :data:`~subterranean.training.deepspeed.ZERO3_CONFIG_PATH` for the 8B
            size and to ``None`` (no DeepSpeed) for the 3B size.

    Returns:
        The full command as a list of argv tokens.

    Example:
        >>> cmd = build_accelerate_command(TrainingConfig.for_3b("out"), "train.py")
        >>> cmd[:3]
        ['accelerate', 'launch', '--num_processes']
        >>> "--use_deepspeed" in cmd
        False
    """
    script = str(train_script)
    cmd: list[str] = ["accelerate", "launch"]

    if config.size == "8b":
        ds_path = Path(deepspeed_config) if deepspeed_config is not None else ZERO3_CONFIG_PATH
        cmd += [
            "--num_processes",
            str(config.num_gpus),
            "--num_machines",
            "1",
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(ds_path),
            "--mixed_precision",
            "bf16",
        ]
    else:
        cmd += [
            "--num_processes",
            "1",
            "--mixed_precision",
            "bf16",
        ]

    cmd.append(script)
    return cmd
