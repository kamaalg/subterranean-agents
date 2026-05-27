# subterranean

> Compile agentic workflows into LLM weights.

`subterranean` (PyPI: [`subterranean-agents`](https://pypi.org/project/subterranean-agents/))
takes a procedural agent workflow and *compiles* it into a small model's weights
via synthetic-data fine-tuning, so the model **self-orchestrates** at runtime
instead of relying on an external orchestrator (LangGraph, CrewAI, …).

Based on Dennis et al. 2026, *Compiling Agentic Workflows into LLM Weights*
(arXiv:2605.22502): near-frontier quality at **128–462× lower inference cost**.

## The idea

Most agent frameworks inject the procedure into a frontier model's prompt on
every turn. That is expensive and slow. Instead, `subterranean`:

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
subterranean compile examples/travel_booking/flowchart.yaml --out build/travel
subterranean generate build/travel --n 2000 --model claude-sonnet-4-5
subterranean train    build/travel --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20
subterranean eval     build/travel --baselines in_context,langgraph --n 200
# or: subterranean serve build/travel --port 8000
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
