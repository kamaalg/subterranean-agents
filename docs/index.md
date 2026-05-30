# agent2model

> Turn your LangGraph/CrewAI agent into a small open model that runs with **no orchestrator** — near-frontier quality at a fraction of the inference cost.

`agent2model` (PyPI: [`agent2model`](https://pypi.org/project/agent2model/))
takes a procedural agent workflow — written as YAML or imported straight from a
LangGraph `StateGraph` — and bakes the whole procedure into a small model's
**weights** via synthetic-data fine-tuning. The result **self-orchestrates** at
runtime: there is no external orchestrator and no per-turn frontier call.

Unlike prompt-optimizers (DSPy, GEPA) that keep a runtime program, and unlike
agent frameworks (LangGraph, CrewAI) that run the procedure live every turn,
`agent2model` removes the orchestrator entirely — the procedure lives in the
weights.

Based on Dennis et al. 2026, *Compiling Agentic Workflows into LLM Weights*
(arXiv:2605.22502), which reports near-frontier quality at **128–462× lower
inference cost**. Those are the paper's figures; this repo's own
independently-reproduced numbers are tracked in the benchmarks and are still
being filled in.

## The idea

Most agent frameworks inject the procedure into a frontier model's prompt on
every turn. That is expensive and slow. Instead, `agent2model`:

1. takes your procedure as a **Flowchart IR** (YAML, or imported from LangGraph);
2. **generates** synthetic conversations that walk the procedure, via Claude;
3. **fine-tunes** a small open model (Qwen 2.5 3B / Qwen3 8B) on those
   conversations — full-parameter SFT, no LoRA;
4. **evaluates** the result against frontier baselines on the paper's
   5-criterion rubric, and **serves** it behind an OpenAI-compatible endpoint.

The flowchart structure never appears in the training data — the model learns to
run the procedure from natural dialogue alone.

## The four-command journey

```bash
agent2model compile examples/travel_booking/flowchart.yaml --out build/travel
agent2model generate build/travel --n 2000 --model claude-sonnet-4-5
agent2model train    build/travel --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20
agent2model eval     build/travel --baselines in_context,langgraph --n 200
# or: agent2model serve build/travel --port 8000
```

See the [Quickstart](quickstart.md) to run it end to end (including with no local
GPU via Modal).

## Where to go next

- **[Quickstart](quickstart.md)** — install and run the full pipeline.
- **[IR spec reference](ir-spec.md)** — the YAML procedure contract.
- **[Training guide](training.md)** — the paper recipe and presets.
- **[Evaluation guide](evaluation.md)** — the rubric, baselines, and report.
- **[Cloud deployment](cloud.md)** — Modal (primary) and RunPod recipes.
- **[Troubleshooting](troubleshooting.md)** and **[FAQ](faq.md)**.

## Scope

v1 ships full-parameter SFT only (no LoRA — the paper's companion shows it fails
on procedural tasks), single-agent procedural workflows, and cloud recipes.
RLHF/DPO, online learning, tool use during inference, and multi-agent handoffs
are v2+. License: Apache-2.0.
