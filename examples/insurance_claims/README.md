# Insurance claims (paper Experiment 3 — 8B, ~55 nodes)

The paper's **complex** procedure: a 59-node first-notice-of-loss auto-claims
intake that verifies the policy, collects incident detail, branches by incident
type (collision / theft / comprehensive), handles injuries and third parties,
verifies coverage, screens for fraud, and routes to a repair or total-loss
resolution — with eight terminals across success, abandonment, and escalation.

This is the biggest authored flowchart in the repo and exercises the library's
ability to compile deep, many-branch procedures. It trains on Qwen3 8B.

## What it does

The procedure has many decision points; the high-level shape:

```
greet → collect_policy_number → verify_policy ─┬─ no_policy_found → (abandonment)
                                               └─ confirm_identity → check_policy_active ─┬─ policy_lapsed → (escalation/abandonment)
collect_incident_basics → collect_incident_type → classify_incident ─┬─ collision_details
                                                                     ├─ theft_details
                                                                     └─ comprehensive_details
   → check_injuries → (injury sub-claim) → check_third_party → assess_liability → collect_damage_photos
   → verify_coverage ─┬─ explain_liability_only → offer_alternatives → (third-party route / not covered / dispute)
                      └─ explain_collision_coverage → fraud_screen ─┬─ refer_siu → (escalation)
                                                                    └─ estimate_damage → assess_total_loss ─┬─ repair_path → … → claim_filed (success)
                                                                                                            └─ total_loss_path → explain_settlement → … → claim_filed (success)
```

Eight terminals: `claim_filed` and `resolved_no_payout` (success);
`claim_not_covered` and `abandoned_no_policy` (abandonment); and four escalation
exits (`escalated_coverage`, `escalated_dispute`, `escalated_fraud`,
`escalated_billing`). See [`flowchart.yaml`](flowchart.yaml) for all nodes. Every
`decision` node is resolved by an LLM at data-generation time only — the compiled
model has no runtime router. Loops (e.g. `final_questions`/`closing_decision`,
`explain_settlement`/`settlement_response`) each include a terminal-reaching
escape edge, so the procedure validates.

## Run the pipeline

Paper setup: Qwen3 8B, ~3,000 conversations (2,700 train), 20 epochs. The 8B path
uses DeepSpeed ZeRO-3; run it on Modal unless you have an 8×A100 host.

```bash
# 1. Compile + validate (this is the largest flowchart — worth eyeballing the node count).
agent2model compile examples/insurance_claims/flowchart.yaml --out build/insurance

# 2. Generate synthetic data (~3k convos cover the many branches).
agent2model generate build/insurance --n 3000 --model claude-sonnet-4-5 --budget 60

# 3. Full fine-tune the 8B base model (ZeRO-3), 20 epochs.
agent2model train build/insurance --base Qwen/Qwen3-8B --size 8b --epochs 20

# 4. Evaluate against baselines (or serve).
agent2model eval build/insurance --baselines in_context,langgraph --n 200
agent2model serve build/insurance --port 8000
```

No local GPU? Run it on Modal:

```bash
modal run -m agent2model.cloud.modal_app::reproduce_insurance
```

## Expected results

Per-criterion means (1–5), `n=200`, agent2model (compiled 8B) vs. the paper.
Fill in **Your run** from `build/insurance/eval_report.json`. The release gate
fails if any criterion drops >5% below the paper number.

| Criterion | Paper (Insurance 8B) | Your run |
|---|---|---|
| Task Success | 4.47 | — |
| Information Accuracy | 4.40 | — |
| Consistency | 4.51 | — |
| Graceful Handling | 4.81 | — |
| Naturalness | 4.92 | — |

**Cost:** ~$0.0007 per conversation, **462×** cheaper than the in-context
frontier baseline, at 87–98% of frontier quality. (The deep procedure makes the
in-context baseline especially expensive — the entire 59-node flowchart sits in
every system prompt — which is why the cost ratio is the largest of the three.)
