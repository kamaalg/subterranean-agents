# Cloud recipes

Run the whole `agent2model` pipeline — generate, train, evaluate, serve — without
a local GPU. **[Modal](https://modal.com) is the primary recipe** (one command);
[RunPod](./runpod/README.md) is the secondary, more manual one.

This `cloud/` package is **decoupled from the core library**: the core never
imports `cloud`, and `import agent2model` / `import agent2model.cloud` work
*without* `modal` installed. The Modal app itself (`modal_app.py`) does require
`modal` (its `@app.function` decorators need it at import time), so it lives behind
the `[cloud]` extra. All pure recipe logic — which image/GPU/training-config each
example uses and the pipeline step order — lives in `_recipes.py`, which has **no
modal import** and is what the unit tests exercise.

## Install

```bash
pip install "agent2model[cloud]"   # installs modal
modal token new                             # one-time Modal auth
```

## Required secret

The generate and evaluate steps call the Anthropic API. Create a Modal secret
named `anthropic-secret` exposing `ANTHROPIC_API_KEY`:

```bash
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
```

The app also creates two persistent volumes on first run:
`agent2model-build` (flowchart IR, `dataset.jsonl`, eval reports) and
`agent2model-models` (fine-tuned weights).

## Reproduce a paper experiment (the headline demo)

```bash
modal run -m agent2model.cloud.modal_app::reproduce_travel
modal run -m agent2model.cloud.modal_app::reproduce_zoom
modal run -m agent2model.cloud.modal_app::reproduce_insurance
```

Each entrypoint chains **generate → train → evaluate** end to end. You must first
have the compiled flowchart on the build volume at `/build/<example>/flowchart.json`
(`agent2model compile <yaml> --out build/<example>` then upload), matching the
example name.

Per-example config (mirrors arXiv:2605.22502v1):

| Entrypoint | Model | Path | Convos | Epochs | Train GPU |
|---|---|---|---|---|---|
| `reproduce_travel` | Qwen2.5-3B-Instruct | 3B | ~2,000 | 20 | 1x A10G/A100 |
| `reproduce_zoom` | Qwen3-8B | 8B | ~6,000 | 10 | 8x A100 80GB (ZeRO-3) |
| `reproduce_insurance` | Qwen3-8B (55 nodes) | 8B | ~3,000 | 20 | 8x A100 80GB (ZeRO-3) |

## Modal functions

These can be invoked individually (e.g. via `.remote(...)` from your own driver):

| Function | Hardware | Role |
|---|---|---|
| `generate_data` | CPU (4 cores) | API-bound synthetic data generation. |
| `train_3b` | 1x A10G | 3B full fine-tuning. |
| `train_8b` | 8x A100 80GB | 8B DeepSpeed ZeRO-3 full fine-tuning. |
| `evaluate` | CPU (4 cores) | Parallel eval harness (LLM-judge + simulator). |
| `serve` | 1x A100 80GB | Autoscaling OpenAI-compatible vLLM endpoint. |

The `serve` function is a `@modal.web_server` on port 8000 with autoscaling
(`min_containers=0`, `max_containers=4`, 300s scaledown), exposing
`/v1/chat/completions` and `/v1/models`. Deploy it with:

```bash
modal deploy -m agent2model.cloud.modal_app
```

## RunPod

See [`runpod/README.md`](./runpod/README.md) for the secondary, JSON-pod-spec flow.
