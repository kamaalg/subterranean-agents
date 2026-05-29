# Zoom support (paper Experiment 2 — 8B)

A 15-node customer-support procedure for a Zoom-style product: greet, collect the
issue, classify it, run the right troubleshooting track (audio/video,
connectivity, or account/billing), check whether it was resolved, and either
confirm the fix, deep-dive, or escalate to Tier 2.

This is the paper's **medium / domain-knowledge** procedure, trained on Qwen3 8B.

## What it does

```
greet → collect_issue → classify_issue ─┬─ troubleshoot_audio_video ─┐
   ↑                                     ├─ troubleshoot_connectivity ├→ check_resolved → evaluate_outcome
   └──────── (too vague) ────────────────┴─ handle_account_billing ──┘                        │
                                                                                               │
   confirm_resolution → resolved (success)         ┌── (resolved) ──────────────────────┘
   escalate_to_tier2  → escalated (escalation)  ───┤   deep_dive ←── (persists) ──┐
   unresolved (abandonment)                        └── escalate_check ←── (frustrated)
```

See [`flowchart.yaml`](flowchart.yaml). The `decision` nodes (`classify_issue`,
`evaluate_outcome`, `escalate_check`) are resolved by an LLM at data-generation
time only; the compiled model self-orchestrates with no runtime router. The
`deep_dive`/`escalate_check` loop has terminal-reaching escape edges, so it
validates.

## Run the pipeline

Paper setup: Qwen3 8B, ~6,264 train conversations, 10 epochs. The 8B path uses
DeepSpeed ZeRO-3 across 8 GPUs (run it on Modal unless you have that locally).

```bash
# 1. Compile + validate.
agent2model compile examples/zoom_support/flowchart.yaml --out build/zoom

# 2. Generate synthetic data (medium volume, ~6k convos).
agent2model generate build/zoom --n 6000 --model claude-sonnet-4-5 --budget 60

# 3. Full fine-tune the 8B base model (ZeRO-3).
agent2model train build/zoom --base Qwen/Qwen3-8B --size 8b --epochs 10

# 4. Evaluate against baselines (or serve).
agent2model eval build/zoom --baselines in_context,langgraph --n 200
agent2model serve build/zoom --port 8000
```

No local GPU? Run it on Modal:

```bash
modal run -m agent2model.cloud.modal_app::reproduce_zoom
```

## Expected results

Per-criterion means (1–5), `n=200`, agent2model (compiled 8B) vs. the paper.
Fill in **Your run** from `build/zoom/eval_report.json`. The release gate fails
if any criterion drops >5% below the paper number.

| Criterion | Paper (Zoom 8B) | Your run |
|---|---|---|
| Task Success | 4.50 | — |
| Information Accuracy | 4.26 | — |
| Consistency | 4.42 | — |
| Graceful Handling | 4.62 | — |
| Naturalness | 4.87 | — |

**Cost:** ~$0.0003 per conversation, **296×** cheaper than the in-context
frontier baseline, at 87–98% of frontier quality.
