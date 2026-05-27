# Quickstart

This walks the full `compile → generate → train → eval` pipeline on the
travel-booking example. Data generation and evaluation call the Anthropic API;
training and serving need a GPU (use [Modal](cloud.md) if you don't have one).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install subterranean-agents              # core: compile + generate + eval
```

Optional extras (install only what you need):

| Extra | Adds |
|---|---|
| `train` | torch, transformers, TRL, DeepSpeed — the fine-tuning stack (GPU) |
| `serve` | vLLM — the OpenAI-compatible serving endpoint (GPU) |
| `cloud` | the Modal client for the cloud recipes |
| `report` | matplotlib — renders the PDF eval report |
| `openai` | OpenAI-compatible client for the served-model eval condition |

```bash
pip install "subterranean-agents[train]"     # e.g. for local fine-tuning
```

Set your Anthropic key (used by `generate` and `eval`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 1. Compile

Validate a workflow and emit the canonical IR. Works on a YAML flowchart or a
`.py` file defining a LangGraph graph.

```bash
subterranean compile examples/travel_booking/flowchart.yaml --out build/travel
```

This writes `build/travel/flowchart.json`. If the flowchart violates an invariant
(unreachable terminal, dead-end cycle, …) the command prints every problem and
exits non-zero. See the [IR spec](ir-spec.md).

## 2. Generate

Generate synthetic training data by walking the flowchart with Claude. The
command **prints the expected cost before starting and the actual cost after**,
is resumable (checkpoints to `build/travel/generation_state.json`), and stops if
the `--budget` cap is hit.

```bash
subterranean generate build/travel --n 2000 --model claude-sonnet-4-5 --budget 60
```

Output: `build/travel/dataset.jsonl` (HF chat-template) and
`build/travel/cost.json`. Useful flags: `--seed`, `--max-concurrent` (default 10).

## 3. Train

Full-parameter fine-tune a base model on the dataset. Pick a `--size` preset
(`3b` single-GPU, `8b` DeepSpeed ZeRO-3); `--base` and `--epochs` default to the
preset.

```bash
subterranean train build/travel --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20
```

The best checkpoint (by held-out eval loss, default 90/10 split) is saved to
`build/travel/model/best`. LoRA is **not** supported — `--lora` is refused. See
the [Training guide](training.md).

## 4. Evaluate or serve

Evaluate the compiled model against baselines on the 5-criterion rubric, with
bootstrap CIs, significance tests, failure rates, and cost:

```bash
subterranean eval build/travel --baselines in_context,langgraph --n 200
```

Outputs `build/travel/eval_report.json` and `eval_report.pdf`. To include the
compiled model as a condition, serve it and pass `--served-url`. See the
[Evaluation guide](evaluation.md).

Serve the compiled model behind an OpenAI-compatible endpoint:

```bash
subterranean serve build/travel --port 8000
# POST http://localhost:8000/v1/chat/completions
```

## No GPU? Reproduce on Modal

```bash
pip install "subterranean-agents[cloud]"
modal run -m subterranean.cloud.modal_app::reproduce_travel
```

This runs generate → train → eval on Modal and produces a compiled 3B model
within 5% of the paper's Table 1. See [Cloud deployment](cloud.md).
