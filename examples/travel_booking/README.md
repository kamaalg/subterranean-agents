# Travel booking (paper Experiment 1 — 3B)

A 14-node procedure that helps a customer plan and book a trip end to end: greet,
gather preferences, present options, refine on feedback, book, and confirm — with
escape paths to escalation and abandonment.

This is the paper's **simple** procedure and the headline reproduction:
`modal run -m agent2model.cloud.modal_app::reproduce_travel` trains a Qwen 2.5
3B model. The **goal** is to land within 5% of the paper's Table 1; that
reproduction has **not yet been run to completion in this repo** — the
[`benchmarks/`](../../benchmarks) "Current" columns are still empty (`—`). This
README will be updated with our measured numbers once the run completes.

## What it does

```
greet → gather_preferences → assess_readiness ─┐
   ↑                                            ├─ present_options → handle_response → evaluate_response
   └────────────── (needs more info) ───────────┘                                        │
                                                       ┌── book → confirm → booking_confirmed (success)
   refine_options ←── (wants changes) ─────────────────┤
   abandon_check ←── (hesitant) ──→ abandoned (abandonment) / escalate_to_human → escalated (escalation)
```

See [`flowchart.yaml`](flowchart.yaml) for the full procedure. `decision` nodes
(`assess_readiness`, `evaluate_response`, `abandon_check`) are resolved by an LLM
*at data-generation time only* — the compiled model has no runtime router.

## Run the pipeline

The four commands (paper setup: Qwen 2.5 3B Instruct, ~2,125 conversations,
20 epochs, ~3.5 h on one consumer GPU):

```bash
# 1. Compile + validate the flowchart into the IR.
agent2model compile examples/travel_booking/flowchart.yaml --out build/travel

# 2. Generate synthetic training data (resumable, budgeted, prompt-cached).
agent2model generate build/travel --n 2000 --model claude-sonnet-4-5 --budget 60

# 3. Full fine-tune a 3B base model on the generated data.
agent2model train build/travel --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20

# 4. Evaluate against baselines (or serve).
agent2model eval build/travel --baselines in_context,langgraph --n 200
agent2model serve build/travel --port 8000
```

No local GPU? Run the whole thing on Modal:

```bash
modal run -m agent2model.cloud.modal_app::reproduce_travel
```

## Expected results

Per-criterion means (1–5), `n=200`, agent2model (compiled 3B) vs. the paper's
Table 1. Fill in **Your run** from `build/travel/eval_report.json` /
`eval_report.pdf`. The release gate fails if any criterion drops >5% below the
paper number (see [`benchmarks/`](../../benchmarks/README.md)).

| Criterion | Paper (Travel 3B) | Your run |
|---|---|---|
| Task Success | 4.11 | — |
| Information Accuracy | 4.75 | — |
| Consistency | 4.34 | — |
| Graceful Handling | 4.07 | — |
| Naturalness | 4.12 | — |

**Cost:** ~$0.0010 per conversation, **128×** cheaper than the in-context
frontier baseline, at 87–98% of frontier quality.
