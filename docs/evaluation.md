# Evaluation guide

Evaluation is the library's differentiator: it ships the paper's methodology end
to end. `agent2model eval` runs your compiled model and a set of baselines
against a flowchart-blind user simulator, judges every conversation on a
5-criterion rubric, and writes a README-quality report.

```bash
agent2model eval build/travel --baselines in_context,langgraph --n 200
```

Outputs `build/travel/eval_report.json` and `eval_report.pdf`. The command prints
expected cost before starting and actual cost after, and stops if the `--budget`
cap is hit.

## The rubric

Five criteria, scored 1–5 with behavioral anchors (from the paper §3):

1. **Task Success** — did the agent execute the procedure correctly through to an
   appropriate terminal state?
2. **Information Accuracy** — did it correctly use and retain user-provided info?
3. **Consistency** — did it maintain coherent state across the conversation (no
   contradictions, no repeated questions)?
4. **Graceful Handling** — how well did it handle changes, ambiguity, and edge
   cases? **Capped at 3 if the user posed no challenges.**
5. **Naturalness** — does it read like talking to a skilled human agent?

## The user simulator

A separate Claude Sonnet 4.5 call role-plays a customer with sampled scenario
variables. **It has no knowledge of the flowchart** — that is what makes the
evaluation generalize rather than test memorisation.

## Baselines (`--baselines`)

Comma-separated, any of:

| Name | What it is |
|---|---|
| `in_context` | The **upper bound**: the entire serialized flowchart in the system prompt of a frontier model (Claude Sonnet 4.5) that self-orchestrates. |
| `langgraph` | The **industry baseline**: a LangGraph orchestrator wrapping the same frontier model. Needs the `langgraph` install. |
| `same_model_orch` | The same base model as the compiled one, but orchestrated — isolates the effect of compilation. |

To include the **compiled** model as a condition, serve it and pass
`--served-url http://host:port` (adds a `compiled` condition).

## What the report contains

For `--n` scenarios per condition (default 200):

- Mean per criterion with **95% bootstrap CIs** (10,000 resamples).
- Pairwise significance: Wilcoxon signed-rank (paired) or Mann-Whitney U
  (unpaired), with **Holm-Bonferroni** correction across the 5 criteria.
- **Failure rate** — fraction of conversations with Task Success ≤ 3.
- **Cost per conversation** and average wall-clock per conversation.

The PDF (per-criterion grouped bar charts with CI error bars, failure rates,
cost breakdown) needs the `report` extra (matplotlib); the JSON report is always
written. Stats use SciPy.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--baselines` | `in_context,langgraph` | conditions to compare |
| `--n` | 200 | scenarios per condition |
| `--judge-model` | `claude-sonnet-4-5` | LLM-as-judge model |
| `--budget` | 50.0 | hard USD cap across all LLM calls |
| `--served-url` | — | OpenAI-compatible URL of a served compiled model |
| `--seed` | 0 | base RNG seed |
| `--max-concurrent` | 10 | concurrent scenario evaluations |

## Regression gate

Reproduction targets and the >5% release gate live in
[`benchmarks/`](https://github.com/kamaalg/agent2model/tree/main/benchmarks);
`tests/e2e/` reads `benchmarks/targets.json` and fails the build if a measured
criterion drops more than 5% below the paper number.
