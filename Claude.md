# CLAUDE.md — Subterranean Agents

## Project

This is an open-source library for **compiling agentic workflows into LLM weights**,
based on Dennis et al. 2026 ("Compiling Agentic Workflows into LLM Weights: Near-Frontier
Quality at Two Orders of Magnitude Less Cost", arXiv:2605.22502).

The thesis: instead of running an agent via an external orchestrator (LangGraph, CrewAI,
etc.) that injects prompts at every turn, fine-tune the procedure directly into a small
model's weights. The model learns to self-orchestrate. Result in the paper:
~98% of frontier quality at 128–462× lower inference cost.

This library makes that pipeline reproducible and usable on any procedural workflow.

**Package name:** `subterranean` (PyPI: `subterranean-agents`)
**License:** Apache 2.0
**Target audience:** ML engineers building production agents on stable workflows.

## What v1 ships

1. **Flowchart IR** — a canonical YAML spec for procedures (nodes, edges, conditions, terminals).
2. **LangGraph adapter** — convert an existing `StateGraph` into the YAML IR automatically.
3. **Synthetic data generation** — traverse the flowchart, sample paths and scenario variables,
   generate turn-by-turn conversations via Claude Sonnet 4.5.
4. **Fine-tuning pipeline** — full-parameter SFT for Qwen 2.5 3B and Qwen3 8B (with hooks
   for any HF causal LM). DeepSpeed ZeRO-3 for 8B, single-GPU for 3B.
5. **Cloud recipes** — Modal and RunPod templates that run the whole pipeline end-to-end
   without local GPU.
6. **Evaluation harness** — LLM-as-judge with the paper's 5 criteria (task success, info accuracy,
   consistency, graceful handling, naturalness), dynamic user simulation, baseline comparisons.
7. **Serving** — vLLM-based inference server with OpenAI-compatible API.

What v1 does NOT ship: LoRA support (paper companion shows it fails on procedural tasks),
RLHF/DPO, online learning, multi-turn tool use. These are v2+.

## Repository layout

```
subterranean/
├── src/subterranean/
│   ├── ir/                 # Flowchart IR — the canonical procedure representation
│   │   ├── schema.py       # Pydantic models: Flowchart, Node, Edge
│   │   ├── loader.py       # YAML → IR
│   │   └── validator.py    # path enumeration, cycle detection, terminal reachability
│   ├── adapters/
│   │   ├── langgraph.py    # StateGraph → IR
│   │   └── crewai.py       # stub for v2
│   ├── generation/
│   │   ├── traversal.py    # path sampling through the flowchart
│   │   ├── scenarios.py    # scenario variable sampling (personalities, domains, etc)
│   │   ├── generator.py    # Claude Sonnet 4.5 turn-by-turn conversation generation
│   │   └── formatter.py    # conversations → HF chat-template dataset
│   ├── training/
│   │   ├── config.py       # TrainingConfig dataclass
│   │   ├── trainer.py      # wraps TRL SFTTrainer with paper's recipe
│   │   ├── deepspeed/      # ZeRO-3 configs for 8B
│   │   └── launch.py       # accelerate launch helpers
│   ├── eval/
│   │   ├── rubric.py       # 5-criterion rubric with behavioral anchors
│   │   ├── judge.py        # Claude/GPT-4 LLM-as-judge implementation
│   │   ├── simulator.py    # dynamic user simulator (Claude Sonnet 4.5)
│   │   ├── baselines.py    # in-context, LangGraph orchestrator baselines
│   │   └── runner.py       # run n=200 scenarios across conditions
│   ├── serve/
│   │   └── vllm_server.py  # OpenAI-compatible server wrapping a compiled model
│   ├── cli.py              # Typer CLI: `subterranean compile`, `eval`, `serve`
│   └── cloud/
│       ├── modal_app.py    # Modal pipeline definition
│       └── runpod/         # RunPod templates + setup scripts
├── examples/
│   ├── travel_booking/     # paper Experiment 1 reproduction
│   ├── zoom_support/       # paper Experiment 2 reproduction
│   ├── insurance_claims/   # paper Experiment 3 reproduction (55 nodes)
│   └── langgraph_demo/     # start with LangGraph, compile to weights
├── tests/                  # pytest, see "testing" section below
├── docs/                   # mkdocs site
├── benchmarks/             # numbers vs paper, kept up to date
└── pyproject.toml
```

## The canonical user journey

There are exactly four commands a user runs. The CLI must make all four feel obvious.

```bash
# 1. Convert a workflow (YAML or LangGraph) into the IR and validate it.
subterranean compile examples/travel/flowchart.yaml --out build/travel

# 2. Generate synthetic training data.
subterranean generate build/travel --n 2000 --model claude-sonnet-4-5

# 3. Fine-tune a base model on the generated data.
subterranean train build/travel --base Qwen/Qwen3-8B --epochs 10

# 4. Evaluate or serve the compiled model.
subterranean eval build/travel --baselines in_context,langgraph --n 200
subterranean serve build/travel --port 8000
```

When this works smoothly from a fresh clone on Modal, v1 is done.

## Flowchart IR — the spec

YAML format. This is the public-facing contract; don't break it casually.

```yaml
name: travel_booking
description: Help a customer book a trip
start: greet
nodes:
  greet:
    role: agent
    prompt: |
      Warmly greet the customer and ask what they need help with today.
    next: [gather_preferences]

  gather_preferences:
    role: agent
    prompt: |
      Ask about destination, dates, budget, and group size. One question per turn.
    next:
      - to: assess_readiness
        when: user_has_provided_info

  assess_readiness:
    role: decision    # decision nodes are LLM-classified at gen time, never at runtime
    next:
      - to: present_options
        when: ready
      - to: gather_preferences
        when: needs_more_info

  present_options:
    role: agent
    prompt: |
      Present 2-3 travel options matching their preferences and constraints.
    next: [handle_response]

  # ... more nodes ...

  booking_confirmed:
    terminal: success
  abandoned:
    terminal: abandonment
  escalated:
    terminal: escalation

scenario_variables:
  destination_pool: [Japan, Italy, Iceland, ...]
  budget_range: [500, 5000]
  user_styles: [decisive, indecisive, skeptical, enthusiastic]
```

Key invariants:
- Every non-terminal node must have at least one outgoing edge.
- Every terminal node must be reachable from `start`.
- `role: decision` nodes are evaluated only during training-data generation (an LLM
  picks the next edge given conversation history). At runtime the compiled model just
  generates — there is no router.
- Cycles are allowed but must include at least one terminal-reaching escape edge.

The validator (`ir/validator.py`) checks all of this and reports human-readable errors.

## Synthetic data generation — the core trick

This is where compilation either succeeds or fails. Read the paper §2 and §3 before
touching this module.

Algorithm:
1. Sample a path from `start` through the flowchart to a terminal. Use weighted random
   walk with config-driven weights so common paths dominate but rare paths get coverage.
2. Sample scenario variables (destinations, budgets, user personality, etc) from pools
   defined in the YAML.
3. Walk the path turn by turn. At each node:
   - Format the node's prompt template with scenario variables and history.
   - Call Claude Sonnet 4.5 to generate the turn's content.
   - For `role: user` turns, the simulator is told to be that personality with those
     scenario variables, no flowchart knowledge.
4. The output is a conversation: alternating user/agent turns. **The flowchart structure
   never appears in the final training data.** The model only sees natural dialogue.

Target volume per paper:
- Simple flow (14 nodes): ~2,000 conversations
- Medium with domain knowledge: ~6,000 conversations
- Complex (55 nodes, many paths): ~3,000 conversations

Cost: roughly $20–60 in Claude API calls per dataset. Add a `--budget` flag that hard-stops.

Save datasets as JSONL with HF-compatible chat-template format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Resumable generation is non-negotiable for the user experience — checkpoint every N
conversations so a network hiccup or budget cap doesn't lose hours of generation. Store
progress in `build/<name>/generation_state.json`.

## Training recipe — match the paper

Default hyperparameters (paper §4):

| Setting | 3B | 8B |
|---|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct | Qwen/Qwen3-8B |
| Precision | bf16 | bf16 |
| Learning rate | 2e-5 (cosine decay) | 2e-5 |
| Optimizer | AdamW 8-bit | AdamW (DeepSpeed) |
| Effective batch size | 16 (grad accum) | 32 |
| Epochs | 20 (best checkpoint ~4) | 10 (best checkpoint ~2) |
| Hardware | 1× consumer GPU | 8× A100 80GB ZeRO-3 |
| Wall-clock | ~3.5 hours | ~15–30 min |

**Full fine-tuning only.** Do not add LoRA in v1. The paper's companion (Dennis et al.
2026b) shows LoRA fails to internalize procedures even at high ranks. If a user asks for
LoRA, the CLI should refuse with a link to that paper.

Use TRL's `SFTTrainer` as the foundation; don't reinvent the training loop. Wrap it with
the project's `TrainingConfig` for ergonomics, but expose the underlying TRL args via
`extra_args` for power users.

Best-checkpoint selection by held-out eval loss. Default 90/10 train/eval split. Save
all checkpoints during training but only push the best to the output dir.

## Evaluation — this is the differentiator

Anyone can fine-tune a model. The reason this library is interesting is that it ships
the paper's evaluation methodology end-to-end. Treat eval as a first-class concern.

The rubric has 5 criteria, scored 1–5 with behavioral anchors (see paper §3):
1. **Task Success** — did the agent execute the procedure correctly through to an
   appropriate terminal state?
2. **Information Accuracy** — did the agent correctly use and retain user-provided info?
3. **Consistency** — did the agent maintain coherent state across the conversation?
4. **Graceful Handling** — how well did it handle ambiguity and edge cases? (capped at 3
   if user posed no challenges)
5. **Naturalness** — does it read like talking to a skilled human?

The behavioral anchors at every level matter. Don't paraphrase from memory — port them
verbatim from the paper's §3.

Baselines to support:
- **in_context** — the entire serialized flowchart in the system prompt, frontier model
  (Claude Sonnet 4.5 by default) self-orchestrates. Upper bound.
- **langgraph** — LangGraph orchestrator wrapping the same frontier model. Industry baseline.
- **same_model_orch** — same base model as the compiled one, but orchestrated. Isolates
  the effect of compilation.

The dynamic user simulator is a separate Claude Sonnet 4.5 call that role-plays a customer
with given scenario variables. **The simulator must have no knowledge of the flowchart.**
This is the only way the eval generalizes.

Default `n=200` scenarios per condition. Report:
- Mean per criterion with 95% bootstrap CIs (10,000 resamples)
- Pairwise Wilcoxon signed-rank (paired) or Mann-Whitney U (unpaired)
- Holm-Bonferroni correction across the 5 criteria
- Failure rate (% of conversations with task success ≤ 3)
- Cost per conversation (Claude API tokens × pricing, vs estimated self-hosted GPU cost)
- Average wall-clock per conversation

Use SciPy for stats. Don't roll your own.

## Cloud-only deployment

Users have no GPU. Modal is the primary recipe, RunPod is secondary.

`subterranean/cloud/modal_app.py` should define:
- `generate_data` function: runs the synth pipeline on Modal CPUs (it's API-bound, no GPU needed)
- `train_3b` function: runs on a single A10G or A100
- `train_8b` function: runs on 8× A100 with DeepSpeed ZeRO-3
- `evaluate` function: runs the eval harness in parallel across scenarios
- `serve` function: deploys the compiled model behind a vLLM endpoint with autoscaling

A user should be able to run the entire travel-booking reproduction from a fresh laptop
with:
```bash
modal run -m subterranean.cloud.modal_app::reproduce_travel
```
This is the single most important demo. Optimize for it.

RunPod templates live in `cloud/runpod/` as JSON pod specs + setup shell scripts. Less
polished than Modal but documented.

## Coding conventions

- **Python 3.11+.** Use modern type hints (`list[X]`, `X | None`, not `List[X]`, `Optional[X]`).
- **Pydantic v2** for all config/IR/data schemas. No dataclasses for anything user-facing.
- **Typer** for CLI, not Click directly. Rich progress bars.
- **Loguru** for logging. Default to INFO, `--verbose` for DEBUG.
- **Async** for any code that makes batched API calls (data generation, eval). Use
  `asyncio` with semaphores for rate limiting. The Anthropic SDK's `AsyncAnthropic` client.
- **Never block on the API without a progress bar.** Users will think the program is hung.
- **No global state.** Pass config explicitly. The only acceptable "global" is the logger.
- **Errors are exceptions, not return values.** Use typed exceptions:
  `FlowchartValidationError`, `GenerationBudgetExceeded`, `TrainingDivergedError`.
- **Imports:** stdlib, then third-party, then `subterranean.*`. Absolute imports only.
- **Black + Ruff + Mypy strict.** CI fails on any of them.
- **Docstrings:** Google style. Every public function has one. Examples in the docstring
  for anything a user would call.
- **No `# type: ignore` without a comment explaining why.**

## Testing

- **pytest** with `pytest-asyncio` for async, `pytest-mock` for mocking.
- **Three test tiers:**
  1. **Unit** (`tests/unit/`) — fast, no network, no GPU. Must pass in <30s total.
     Mock all LLM calls. This is what runs on every PR.
  2. **Integration** (`tests/integration/`) — uses real Anthropic API but with tiny
     budgets (~10 conversations). Tagged `@pytest.mark.integration`, runs on nightly CI.
  3. **End-to-end** (`tests/e2e/`) — full pipeline on the travel-booking example.
     Tagged `@pytest.mark.e2e`, runs on release candidates. Compares accuracy against
     paper's published numbers; CI fails if we regress > 5%.
- **Coverage target:** 85% for `src/subterranean/ir`, `generation`, `eval`. The training
  module is hard to unit-test; aim for 60% there and lean on e2e.
- **Fixtures** in `tests/conftest.py`. The travel flowchart YAML is a shared fixture.
- **Snapshot testing** with `syrupy` for generated conversations and eval reports.

## Anthropic API usage rules

- Always use the official `anthropic` Python SDK, never raw `requests`.
- Always use `AsyncAnthropic` for batch operations.
- Concurrency: default `max_concurrent=10`, exposed via config. Anthropic rate limits
  are tier-dependent; respect 429s with exponential backoff (the SDK does this by default
  but verify).
- Prompt caching: use it aggressively for the data generator's system prompt (the
  flowchart spec is identical across all turns of a generation run). Should cut data-gen
  cost ~60%.
- Model strings: `claude-sonnet-4-5` is the default everywhere the paper used Sonnet 4.5.
- Token counting: log token usage to `build/<name>/cost.json` so users can see what they
  spent.

## What "done" looks like for v1

The library is v1-ready when all of the following are true:

1. `pip install subterranean-agents && modal run -m subterranean.cloud.modal_app::reproduce_travel`
   produces a compiled 3B model with eval scores within 5% of the paper's Table 1.
2. The same works for `reproduce_zoom` and `reproduce_insurance` on 8B.
3. A user with a LangGraph `StateGraph` can run
   `subterranean compile path/to/graph.py --out build/mine` and proceed through the
   pipeline without writing YAML by hand.
4. The eval harness produces a report PDF with per-criterion bar charts, baseline
   comparisons, failure rates, cost breakdowns. Looks like something you'd put in a
   GitHub README.
5. `subterranean serve` exposes an OpenAI-compatible chat endpoint backed by vLLM.
6. The docs site has: quickstart, IR spec reference, training guide, eval guide,
   cloud deployment guide, troubleshooting, FAQ.
7. CI is green: ruff, mypy strict, pytest unit + integration on the latest tag.
8. The three paper reproductions are documented as separate `examples/` with README
   walkthroughs and expected numbers.

## Things to avoid

- **Don't ship LoRA.** Even if it seems like a nice-to-have, the paper's companion shows
  it fails on procedural tasks. Shipping a known-broken path damages credibility.
- **Don't reinvent training infra.** TRL + DeepSpeed are battle-tested. Wrap, don't replace.
- **Don't add multi-agent or tool-use in v1.** The paper is about single-agent procedural
  workflows. Stay scoped. v2 can expand.
- **Don't claim parity with the paper until eval numbers prove it.** Every reproduction
  example's README must show the actual scores side by side with the paper's.
- **Don't make users write Python to use the library.** The CLI + YAML is the contract.
  Python API is for power users.
- **Don't depend on a specific cloud provider in the core.** The `cloud/` module is
  separate from `src/subterranean/` for a reason.
- **Don't hide API costs.** Every command that calls an LLM prints expected cost before
  starting and actual cost after.

## Roadmap signals (for context, not v1 work)

v2 candidates, in rough priority order:
- Tool use during compiled inference (the agent calls real APIs)
- Multi-agent procedures (handoffs between specialist agents)
- Continual learning when the procedure changes (avoid full recompile)
- Procedure mining from conversation logs (induce the flowchart from real data)
- CrewAI and OpenAI Agents SDK adapters
- ONNX export for edge deployment

Don't build these in v1. Listing them so contributors aren't surprised when they're punted.

## Maintainer notes

- Cut releases with `cz bump` (commitizen). Conventional commits enforced.
- Changelog auto-generated.
- PyPI publishes from GitHub Actions on tag.
- Bench numbers in `benchmarks/` updated on every minor release; if numbers regress, that
  blocks the release.

## References

- Dennis et al. 2026a. *Compiling Agentic Workflows into LLM Weights*. arXiv:2605.22502.
- Dennis et al. 2026b. *Procedural Knowledge is Not Low-Rank* (companion, justifies no-LoRA).
- TRL docs: https://huggingface.co/docs/trl
- DeepSpeed ZeRO docs: https://www.deepspeed.ai/tutorials/zero/
- Modal docs: https://modal.com/docs
- vLLM docs: https://docs.vllm.ai