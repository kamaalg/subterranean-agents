# Troubleshooting

## `compile` reports flowchart errors

The validator prints one human-readable line per problem. Common causes:

- **"non-terminal node 'X' has no outgoing edges"** — give it a `next`, or make
  it a terminal (`terminal: success|abandonment|escalation`).
- **"terminal node 'X' is not reachable from 'start'"** — no path leads to it;
  add the missing edge or remove the terminal.
- **"node 'X' is reachable but cannot reach any terminal"** — a cycle with no
  escape. Add a terminal-reaching edge somewhere in the loop.
- **"`start` points to unknown node"** / **"edge to unknown node"** — a typo'd
  node id. Ids are case-sensitive.

See the [IR spec](ir-spec.md) for the full invariant list.

## Compiling a LangGraph `.py` file fails

- **"No LangGraph StateGraph found"** — expose the graph as a module-level
  variable named `graph`, `app`, or `workflow`, or a zero-arg factory named
  `build_graph` / `make_graph` / `create_graph`.
- **"Conditional edge … has no path map"** — `add_conditional_edges` needs an
  explicit path map (`{label: target}`) so the targets are statically knowable.
- **"Conditional edge from START is not supported"** — add a deterministic
  `add_edge(START, <node>)` entry node.

After a successful compile, agent nodes carry `TODO` placeholder prompts — fill
them in before `generate`.

## `generate` is slow or looks hung

Generation always shows a Rich progress bar. If throughput is low, raise
`--max-concurrent` (default 10), but mind Anthropic rate limits — 429s are
retried with exponential backoff. Generation is resumable: re-running picks up
from `build/<name>/generation_state.json`.

## `GenerationBudgetExceeded`

The run hit the `--budget` USD cap and stopped cleanly; partial data and
`cost.json` are preserved. Raise `--budget` and re-run (it resumes) if you need
more conversations.

## `train` exits with an install hint

The fine-tuning stack is GPU-only and not installed by default:

```bash
pip install "subterranean-agents[train]"
```

…and run on a CUDA host. No GPU? Use the [Modal recipes](cloud.md).

## `--lora` is refused

Intentional. Full fine-tuning only — LoRA fails to internalise procedures (Dennis
et al. 2026b). Re-run without `--lora`.

## `TrainingDivergedError`

Loss went NaN or blew up. Lower the learning rate via `TrainingConfig.extra_args`,
check the dataset is well-formed chat-template JSONL, and confirm the base model
id is correct.

## `serve` can't start / vLLM missing

vLLM is GPU/CUDA-only:

```bash
pip install "subterranean-agents[serve]"
```

The command resolves `<build>/model/best` (falling back to the dir itself); make
sure training produced a `best` checkpoint.

## The PDF eval report wasn't written

PDF rendering needs matplotlib:

```bash
pip install "subterranean-agents[report]"
```

The JSON report (`eval_report.json`) is always written regardless.

## `eval` with `langgraph` baseline errors

The `langgraph` baseline needs the LangGraph install
(`pip install "subterranean-agents[dev]"` includes it, or `pip install langgraph`).
The `compiled` condition needs a served model URL via `--served-url`.
