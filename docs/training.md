# Training guide

`subterranean train` runs full-parameter supervised fine-tuning on the synthetic
dataset, wrapping TRL's `SFTTrainer` with the paper's recipe. It is GPU-only;
without a GPU, use the [Modal recipes](cloud.md).

```bash
subterranean train build/travel --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20
```

It reads `build/<name>/dataset.jsonl` and saves the best checkpoint (by held-out
eval loss, default 90/10 split) to `build/<name>/model/best`.

## Size presets

`--size` selects a preset; `--base` and `--epochs` override the preset defaults.

| Setting | `3b` | `8b` |
|---|---|---|
| Default base model | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen3-8B` |
| Precision | bf16 | bf16 |
| Learning rate | 2e-5 (cosine decay) | 2e-5 |
| Effective batch size | 16 (grad accum) | 32 |
| Default epochs | 20 (best ~4) | 10 (best ~2) |
| Hardware | 1× consumer GPU | 8× A100 80GB, DeepSpeed ZeRO-3 |
| Wall-clock | ~3.5 h | ~15–30 min |

The 8B path is launched with `accelerate launch` and DeepSpeed ZeRO-3 configs
(typically on a Modal 8×A100 host).

## Checkpoint selection

All checkpoints are saved during training, but only the **best by held-out eval
loss** is promoted to `<build>/model/best`. That is the directory `serve` and
`eval` use.

## No LoRA — by design

`--lora` is **refused**. The paper's companion (Dennis et al. 2026b, *Procedural
Knowledge is Not Low-Rank*) shows LoRA fails to internalise procedures even at
high rank, so shipping it would be a known-broken path. Full fine-tuning only.

## Power users

The CLI covers the common path. For finer control, use `TrainingConfig` directly:

```python
from pathlib import Path
from subterranean.training.config import TrainingConfig
from subterranean.training.trainer import train

config = TrainingConfig.for_3b("build/travel/model", epochs=20)
best = train(config, Path("build/travel/dataset.jsonl"))
print(best.path, best.eval_loss)
```

`TrainingConfig` exposes an `extra_args` passthrough to the underlying TRL
arguments for hyperparameters the preset doesn't surface.

## Divergence

If loss goes NaN or blows up, training raises `TrainingDivergedError` and the
command exits with an actionable message rather than silently producing a broken
checkpoint.

## How much data?

Per the paper, by procedure complexity:

| Procedure | Conversations |
|---|---|
| Simple (≈14 nodes, e.g. travel) | ~2,000 |
| Medium with domain knowledge (e.g. zoom) | ~6,000 |
| Complex (≈55 nodes, e.g. insurance) | ~3,000 |

Generate these with `subterranean generate` before training.
