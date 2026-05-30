# Benchmarks

**The procedural-agent-adherence benchmark.** Unlike QA, math, or tool-use
benchmarks, this measures whether a model correctly *runs a multi-turn procedure*
through to an appropriate terminal state — the thing `agent2model` compiles. The
harness (5-criterion LLM-judge rubric + flowchart-blind user simulator + baselines
+ bootstrap CIs / Wilcoxon / Holm-Bonferroni) is reusable on any procedure, so
these tables are a standard others can reproduce, not just our self-report.

Reproduction numbers for `agent2model` against the paper
(Dennis et al. 2026a, *Compiling Agentic Workflows into LLM Weights*,
arXiv:2605.22502v1). These tables track how close the library's compiled models
get to the paper's published results.

> **Status (2026-05):** the **Paper** columns are the figures **as reported by
> Dennis et al. 2026a**, transcribed here as reproduction targets — they are
> **not independently verified by this project**, and we have not been able to
> confirm the source numbers beyond the preprint itself. **Current** columns are
> this repo's own reproduced numbers — being run now and filled in as each
> reproduction completes. Until those land, treat every Paper figure as *an
> unverified target we are working toward*, not a result this library has shown.

The target numbers live in [`targets.json`](targets.json), which is the **single
source of truth**: the end-to-end regression gate (`tests/e2e/`) reads it, the
example READMEs cite it, and these tables are derived from it. Update one place,
not five.

## How to read these tables

- **Per-criterion means** are the agent2model (compiled) model's scores on the
  paper's 5-criterion LLM-judge rubric, `n=200` scenarios per condition, scored
  1–5. Higher is better.
- **Paper** is the published target. **Current** is the most recent measured run
  in this repo. `—` means not yet measured on this machine (no GPU here).
- These numbers are **regression-gated**: a release is blocked if a measured
  criterion drops more than **5%** below the paper target. See
  [Regression gate](#regression-gate).

## Travel booking — Qwen 2.5 3B Instruct

20 epochs · ~2,125 conversations (1,912 train).

| Criterion | Paper (3B) | Current |
|---|---|---|
| Task Success | 4.11 | — |
| Information Accuracy | 4.75 | — |
| Consistency | 4.34 | — |
| Graceful Handling | 4.07 | — |
| Naturalness | 4.12 | — |

## Zoom support — Qwen3 8B

10 epochs · ~6,264 train conversations.

| Criterion | Paper (8B) | Current |
|---|---|---|
| Task Success | 4.50 | — |
| Information Accuracy | 4.26 | — |
| Consistency | 4.42 | — |
| Graceful Handling | 4.62 | — |
| Naturalness | 4.87 | — |

## Insurance claims — Qwen3 8B (~55-node procedure, 59 in this repo)

20 epochs · ~3,000 conversations (2,700 train).

| Criterion | Paper (8B) | Current |
|---|---|---|
| Task Success | 4.47 | — |
| Information Accuracy | 4.40 | — |
| Consistency | 4.51 | — |
| Graceful Handling | 4.81 | — |
| Naturalness | 4.92 | — |

## Cost per conversation

Compiled (self-hosted, amortised GPU) vs. the in-context frontier baseline
(Claude Sonnet 4.5 with the full flowchart in the system prompt). Quality
retention across the three procedures is **87–98%** of frontier.

| Procedure | agent2model $/conv | Cost ratio vs in-context |
|---|---|---|
| Travel booking | $0.0010 | 128× |
| Zoom support | $0.0003 | 296× |
| Insurance claims | $0.0007 | 462× |

## Regression gate

`benchmarks/targets.json` holds the paper means and the `tolerance` (0.05). The
e2e tier (`tests/e2e/`, marked `@pytest.mark.e2e`) runs a full reproduction and
calls the pure helper
`agent2model`-adjacent `within_tolerance` / `assert_no_regression`
(`tests/e2e/regression.py`) to compare each measured criterion against its
target. A criterion that falls more than 5% below the paper number fails the
build, which blocks the release.

The e2e tier needs a trained model and is **skipped by default**. It runs only
when `AGENT2MODEL_E2E=1` is set and a built model/report path is present (see
`tests/e2e/README` notes in `tests/e2e/test_reproductions.py`). The regression
math itself is unit-tested in `tests/unit/test_regression_gate.py`, so the
>5% logic is covered even where the full pipeline cannot run.
