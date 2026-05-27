"""Unit tests for Phase 4 fine-tuning pipeline.

These tests never import torch/trl/transformers/datasets. They exercise the
config presets, the pure data-splitting and checkpoint-selection logic, the
divergence guard, the LoRA refusal, and the ``accelerate launch`` command
builder. The trainer orchestration's import guard is tested by monkeypatching
the lazy dependency check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from subterranean.exceptions import TrainingDivergedError
from subterranean.training.config import (
    DEFAULT_3B_MODEL,
    DEFAULT_8B_MODEL,
    TrainingConfig,
)
from subterranean.training.deepspeed import ZERO3_CONFIG_PATH
from subterranean.training.launch import build_accelerate_command
from subterranean.training.trainer import (
    CheckpointInfo,
    check_divergence,
    is_diverged,
    load_jsonl,
    reject_lora,
    select_best_checkpoint,
    split_dataset,
)

# --------------------------------------------------------------------------- #
# Config presets
# --------------------------------------------------------------------------- #


def test_3b_preset_matches_paper() -> None:
    cfg = TrainingConfig.for_3b("out")
    assert cfg.size == "3b"
    assert cfg.base_model == DEFAULT_3B_MODEL
    assert cfg.epochs == 20
    assert cfg.effective_batch_size == 16
    assert cfg.learning_rate == pytest.approx(2e-5)
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.bf16 is True
    assert cfg.optim == "adamw_8bit"
    assert cfg.eval_split == pytest.approx(0.10)
    assert cfg.num_gpus == 1


def test_8b_preset_matches_paper() -> None:
    cfg = TrainingConfig.for_8b("out")
    assert cfg.size == "8b"
    assert cfg.base_model == DEFAULT_8B_MODEL
    assert cfg.epochs == 10
    assert cfg.effective_batch_size == 32
    assert cfg.learning_rate == pytest.approx(2e-5)
    assert cfg.bf16 is True
    assert cfg.optim == "adamw_torch"
    assert cfg.num_gpus == 8


def test_preset_overrides() -> None:
    cfg = TrainingConfig.for_3b("out", base_model="my/model", epochs=3)
    assert cfg.base_model == "my/model"
    assert cfg.epochs == 3


def test_extra_args_passthrough_into_sft_kwargs() -> None:
    cfg = TrainingConfig.for_3b("out", extra_args={"weight_decay": 0.1, "bf16": False})
    kwargs = cfg.to_sft_kwargs()
    assert kwargs["weight_decay"] == 0.1
    # extra_args is merged last, so it overrides preset values.
    assert kwargs["bf16"] is False
    assert kwargs["num_train_epochs"] == 20
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["save_strategy"] == "epoch"


def test_config_no_lora_fields() -> None:
    # No LoRA fields exist on the model.
    assert "lora_rank" not in TrainingConfig.model_fields
    assert "use_peft" not in TrainingConfig.model_fields


def test_config_rejects_use_lora() -> None:
    with pytest.raises(ValueError, match="2026b"):
        TrainingConfig.for_3b("out", use_lora=True)


# --------------------------------------------------------------------------- #
# split_dataset
# --------------------------------------------------------------------------- #


def _records(n: int) -> list[dict[str, int]]:
    return [{"i": i} for i in range(n)]


def test_split_respects_fraction() -> None:
    train, ev = split_dataset(_records(100), 0.10, seed=0)
    assert len(ev) == 10
    assert len(train) == 90


def test_split_is_deterministic() -> None:
    a = split_dataset(_records(50), 0.10, seed=7)
    b = split_dataset(_records(50), 0.10, seed=7)
    assert a == b


def test_split_seed_changes_partition() -> None:
    _, ev0 = split_dataset(_records(50), 0.10, seed=0)
    _, ev1 = split_dataset(_records(50), 0.10, seed=1)
    assert ev0 != ev1


def test_split_no_overlap_and_complete() -> None:
    train, ev = split_dataset(_records(37), 0.10, seed=3)
    train_ids = {r["i"] for r in train}
    eval_ids = {r["i"] for r in ev}
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids | eval_ids == set(range(37))


def test_split_small_dataset_keeps_one_eval() -> None:
    train, ev = split_dataset(_records(3), 0.10, seed=0)
    assert len(ev) == 1
    assert len(train) == 2


def test_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="eval_fraction"):
        split_dataset(_records(10), 0.0, seed=0)
    with pytest.raises(ValueError, match="eval_fraction"):
        split_dataset(_records(10), 1.0, seed=0)


# --------------------------------------------------------------------------- #
# select_best_checkpoint / divergence
# --------------------------------------------------------------------------- #


def test_select_best_picks_lowest_eval_loss() -> None:
    ckpts = [
        CheckpointInfo(step=1, path="c1", eval_loss=0.9),
        CheckpointInfo(step=2, path="c2", eval_loss=0.4),
        CheckpointInfo(step=3, path="c3", eval_loss=0.6),
    ]
    assert select_best_checkpoint(ckpts).path == "c2"


def test_select_best_ignores_none_and_nan() -> None:
    ckpts = [
        CheckpointInfo(step=1, path="c1", eval_loss=None),
        CheckpointInfo(step=2, path="c2", eval_loss=float("nan")),
        CheckpointInfo(step=3, path="c3", eval_loss=0.5),
    ]
    assert select_best_checkpoint(ckpts).path == "c3"


def test_select_best_tie_breaks_on_earlier_step() -> None:
    ckpts = [
        CheckpointInfo(step=5, path="late", eval_loss=0.3),
        CheckpointInfo(step=2, path="early", eval_loss=0.3),
    ]
    assert select_best_checkpoint(ckpts).path == "early"


def test_select_best_raises_when_no_finite() -> None:
    with pytest.raises(ValueError, match="finite eval loss"):
        select_best_checkpoint([CheckpointInfo(step=1, path="c1", eval_loss=None)])


def test_is_diverged() -> None:
    assert is_diverged(float("nan")) is True
    assert is_diverged(float("inf")) is True
    assert is_diverged(-float("inf")) is True
    assert is_diverged(0.5) is False
    assert is_diverged(None) is False


def test_check_divergence_raises() -> None:
    ckpts = [
        CheckpointInfo(step=1, path="c1", eval_loss=0.5),
        CheckpointInfo(step=2, path="c2", eval_loss=math.inf),
    ]
    with pytest.raises(TrainingDivergedError, match="diverged"):
        check_divergence(ckpts)


def test_check_divergence_passes_clean() -> None:
    check_divergence([CheckpointInfo(step=1, path="c1", eval_loss=0.5)])


# --------------------------------------------------------------------------- #
# reject_lora
# --------------------------------------------------------------------------- #


def test_reject_lora_passes_when_false() -> None:
    reject_lora(use_lora=False)


def test_reject_lora_mentions_companion_paper() -> None:
    with pytest.raises(ValueError) as exc:
        reject_lora(use_lora=True)
    assert "2026b" in str(exc.value)
    assert "LoRA" in str(exc.value)


# --------------------------------------------------------------------------- #
# load_jsonl
# --------------------------------------------------------------------------- #


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "hi"}]}\n\n'
        '{"messages": [{"role": "assistant", "content": "yo"}]}\n',
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0]["messages"][0]["content"] == "hi"


def test_load_jsonl_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# launch command builder
# --------------------------------------------------------------------------- #


def test_launch_3b_single_gpu() -> None:
    cmd = build_accelerate_command(TrainingConfig.for_3b("out"), "train.py")
    assert cmd[:2] == ["accelerate", "launch"]
    assert "--num_processes" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "1"
    assert "--use_deepspeed" not in cmd
    assert "--mixed_precision" in cmd
    assert cmd[-1] == "train.py"


def test_launch_8b_deepspeed_multi_gpu() -> None:
    cfg = TrainingConfig.for_8b("out", num_gpus=8)
    cmd = build_accelerate_command(cfg, "train.py")
    assert "--use_deepspeed" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "8"
    ds_idx = cmd.index("--deepspeed_config_file")
    assert cmd[ds_idx + 1] == str(ZERO3_CONFIG_PATH)
    assert cmd[-1] == "train.py"


def test_launch_8b_custom_deepspeed_config() -> None:
    cfg = TrainingConfig.for_8b("out")
    cmd = build_accelerate_command(cfg, "train.py", deepspeed_config="/custom/zero.json")
    assert "/custom/zero.json" in cmd


def test_zero3_config_is_valid_json_stage3() -> None:
    data = json.loads(ZERO3_CONFIG_PATH.read_text(encoding="utf-8"))
    assert data["zero_optimization"]["stage"] == 3
    assert data["bf16"]["enabled"] is True


# --------------------------------------------------------------------------- #
# trainer import guard
# --------------------------------------------------------------------------- #


def test_train_raises_helpful_error_without_ml_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subterranean.training import trainer

    def _boom() -> None:
        raise RuntimeError(
            "Training requires the optional ML stack, which is not installed. "
            "Install it with `pip install -e '.[train]'`"
        )

    monkeypatch.setattr(trainer, "_require_train_deps", _boom)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    cfg = TrainingConfig.for_3b(str(tmp_path / "model"))
    with pytest.raises(RuntimeError, match=r"\[train\]"):
        trainer.train(cfg, dataset)
