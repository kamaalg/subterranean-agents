# subterranean

> Compile agentic workflows into LLM weights.

`subterranean` (PyPI: `subterranean-agents`) takes a procedural agent workflow and
*compiles* it into a small model's weights via synthetic-data fine-tuning, so the model
self-orchestrates at runtime instead of relying on an external orchestrator (LangGraph,
CrewAI, …). Based on Dennis et al. 2026, *Compiling Agentic Workflows into LLM Weights*
(arXiv:2605.22502): near-frontier quality at 128–462× lower inference cost.

**License:** Apache-2.0 · **Status:** alpha (v1 in progress)

## The four-command journey

```bash
# 1. Validate a workflow and emit the canonical IR.
subterranean compile examples/travel_booking/flowchart.yaml --out build/travel

# 2. Generate synthetic training data.            (Phase 2)
subterranean generate build/travel --n 2000 --model claude-sonnet-4-5

# 3. Fine-tune a base model on the generated data. (Phase 4)
subterranean train build/travel --base Qwen/Qwen2.5-3B-Instruct --epochs 20

# 4. Evaluate or serve the compiled model.         (Phases 5–6)
subterranean eval build/travel --baselines in_context,langgraph --n 200
subterranean serve build/travel --port 8000
```

Currently implemented: **`compile`** (Flowchart IR + validator). The remaining commands
print the phase that adds them.

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tooling/tests
pip install -e ".[dev,train]"    # add the fine-tuning stack
```

## Flowchart IR

A procedure is described in YAML. This is the public contract — see
`examples/travel_booking/flowchart.yaml` for a complete one.

```yaml
name: travel_booking
description: Help a customer book a trip
start: greet
nodes:
  greet:
    role: agent           # agent | user | decision
    prompt: Warmly greet the customer and ask what they need.
    next: [gather_preferences]

  assess_readiness:
    role: decision        # an LLM picks the edge at generation time; no runtime router
    next:
      - to: present_options
        when: user has provided all required info
      - to: gather_preferences
        when: details are still missing

  booking_confirmed:
    terminal: success     # success | abandonment | escalation
scenario_variables:
  destination_pool: [Japan, Italy, Iceland]
  user_styles: [decisive, indecisive, skeptical]
```

**Invariants** (enforced by `subterranean.ir.validator`):

- Every non-terminal node has at least one outgoing edge.
- Every terminal node is reachable from `start`.
- `role: decision` nodes are resolved only during data generation — never at runtime.
- Cycles are allowed but must contain a terminal-reaching escape edge.

## Development

```bash
ruff check . && black --check . && mypy src && pytest tests/unit
```

Test tiers: `unit` (fast, mocked, every PR) · `integration` (real API, tiny budget, nightly)
· `e2e` (full reproduction, release candidates). See `pyproject.toml` markers.

## Scope

v1 ships full-parameter SFT only (no LoRA — see Dennis et al. 2026b), single-agent
procedural workflows, and cloud recipes (Modal/RunPod). RLHF/DPO, online learning, tool
use, and multi-agent handoffs are v2+.
