"""Build ``accelerate launch`` commands for the 3B and 8B training paths.

These helpers do pure string/argument construction — no subprocess execution —
so they are fully unit-testable without the ML stack or a GPU. The 3B path is a
single-GPU launch; the 8B path adds the DeepSpeed ZeRO-3 config and a multi-GPU
process count.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from agent2model.training.config import TrainingConfig
from agent2model.training.deepspeed import ZERO3_CONFIG_PATH

if TYPE_CHECKING:
    from agent2model.training.trainer import CheckpointInfo


def build_accelerate_command(
    config: TrainingConfig,
    train_script: str | Path,
    *,
    deepspeed_config: str | Path | None = None,
    as_module: bool = False,
    script_args: Sequence[str] | None = None,
) -> list[str]:
    """Build the ``accelerate launch`` argv for a training run.

    The 3B preset launches a single process on one GPU. The 8B preset launches
    ``config.num_gpus`` processes wired to the DeepSpeed ZeRO-3 config.

    Args:
        config: The training configuration (its ``size``/``num_gpus`` drive the
            launch topology).
        train_script: Path to the Python entry point ``accelerate`` should run.
        deepspeed_config: Override DeepSpeed config path; defaults to the bundled
            :data:`~agent2model.training.deepspeed.ZERO3_CONFIG_PATH` for the 8B
            size and to ``None`` (no DeepSpeed) for the 3B size.
        as_module: When True, ``train_script`` is a module name run with ``-m``
            (``accelerate launch -m <module>``) rather than a file path.
        script_args: Extra argv tokens appended after the script/module (e.g.
            ``["--config", "cfg.json"]``).

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

    if as_module:
        cmd += ["-m", script]
    else:
        cmd.append(script)
    if script_args:
        cmd += list(script_args)
    return cmd


#: Module run under ``accelerate launch`` for the multi-GPU 8B path.
_ENTRY_MODULE = "agent2model.training._entry"


def launch_training(config: TrainingConfig, dataset_path: str | Path) -> CheckpointInfo:
    """Run training, routing the 8B path through ``accelerate launch``.

    The 3B preset is single-GPU, so it trains in-process. The 8B preset needs
    DeepSpeed ZeRO-3 across ``config.num_gpus`` processes, which only happens
    under ``accelerate launch`` — running it in-process would silently fall back
    to a single process and OOM. For 8B this serialises the config to JSON and
    spawns ``accelerate launch -m agent2model.training._entry`` as a subprocess;
    the entry module reconstructs the config and calls :func:`...trainer.train`
    under the distributed launcher, writing the selected checkpoint's metadata to
    ``<output_dir>/best.json`` for this function to read back.

    Args:
        config: The training configuration (its ``size`` selects the path).
        dataset_path: Path to the chat-template JSONL dataset.

    Returns:
        The :class:`~agent2model.training.trainer.CheckpointInfo` for the best
        checkpoint.

    Raises:
        TrainingDivergedError: If the 8B subprocess reports divergence.
        RuntimeError: If the 8B subprocess fails for any other reason.
    """
    import subprocess

    from agent2model.training.trainer import CheckpointInfo, train

    if config.size != "8b":
        return train(config, dataset_path)

    # 8B: launch the distributed entry module via accelerate.
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = out_root / "training_config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    best_meta = out_root / "best.json"
    if best_meta.exists():
        best_meta.unlink()

    cmd = build_accelerate_command(
        config,
        _ENTRY_MODULE,
        as_module=True,
        script_args=[
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
        ],
    )
    # Prefer the current interpreter's accelerate so the right env is used.
    if cmd[0] == "accelerate":
        cmd = [sys.executable, "-m", "accelerate.commands.launch", *cmd[2:]]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"8B training launch failed (exit {result.returncode}). " f"Command: {' '.join(cmd)}"
        )
    if not best_meta.exists():
        raise RuntimeError(
            "8B training finished but wrote no best.json; the run may have crashed "
            "before selecting a checkpoint."
        )
    return CheckpointInfo.model_validate_json(best_meta.read_text(encoding="utf-8"))
