# Cloud deployment

Most users have no local GPU. **Modal** is the primary, one-command recipe;
**RunPod** is the secondary, more-manual recipe. The `cloud/` module is kept
decoupled from the core library.

## Modal (primary)

```bash
pip install "subterranean-agents[cloud]"
modal token new            # one-time auth
modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...   # for generate/eval
```

### The headline demo

Reproduce the paper's travel-booking experiment end to end — generate, train a
Qwen 2.5 3B model, and evaluate — from a fresh laptop:

```bash
modal run -m subterranean.cloud.modal_app::reproduce_travel
```

The two 8B reproductions have their own entrypoints:

```bash
modal run -m subterranean.cloud.modal_app::reproduce_zoom
modal run -m subterranean.cloud.modal_app::reproduce_insurance
```

Each runs the full pipeline and produces a compiled model whose eval scores are
within 5% of the paper. See [`benchmarks/`](https://github.com/subterranean-agents/subterranean/tree/main/benchmarks)
for the targets.

### Building blocks

`subterranean/cloud/modal_app.py` defines reusable Modal functions you can
compose into your own pipeline:

| Function | Hardware | Role |
|---|---|---|
| `generate_data` | CPU (API-bound) | synthetic data generation |
| `train_3b` | 1× A10G/A100 | full fine-tune the 3B preset |
| `train_8b` | 8× A100 80GB, ZeRO-3 | full fine-tune the 8B preset |
| `evaluate` | CPU (parallel) | run the eval harness across scenarios |
| `serve` | GPU, autoscaling | deploy a vLLM OpenAI-compatible endpoint |

## RunPod (secondary)

RunPod is more manual: you launch a pod from a JSON spec, the pod runs `setup.sh`,
which installs the package and invokes the right CLI stage. Build artifacts live
on the pod's persistent volume at `/workspace`.

| File (`cloud/runpod/`) | Purpose |
|---|---|
| `train_3b.json` | Single-GPU pod for the 3B path |
| `train_8b.json` | 8× A100 80GB pod for the 8B ZeRO-3 path |
| `serve.json` | Single-GPU pod exposing the vLLM endpoint on `:8000` |
| `setup.sh` | Installs the package and runs a stage: `generate` / `train` / `evaluate` / `serve` |

Typical flow: compile locally and upload `build/<example>/` to the pod volume,
generate data (cheap pod or local), train on the GPU pod, evaluate, then serve.
The full walkthrough is in `cloud/runpod/README.md`. Only generate/eval need
`ANTHROPIC_API_KEY`; only train/serve need a GPU.

!!! note "Secrets"
    RunPod has no managed secret store as integrated as Modal's. Set
    `ANTHROPIC_API_KEY` in the pod `env` for the generate/evaluate stages and
    treat it as sensitive.
