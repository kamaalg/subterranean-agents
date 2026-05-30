# LangGraph adapter

`agent2model compile my_graph.py --out build/mine` imports a LangGraph
`StateGraph` and converts its **structure** into the [Flowchart IR](ir-spec.md),
so you don't have to hand-write YAML. This page documents exactly what the
adapter does and does not recover, the manual steps you must take afterwards, and
the security implications of compiling a `.py` file.

## Security: compiling `.py` executes code

Compiling a LangGraph file **imports and runs it** (`importlib` →
`exec_module`), and calls a zero-argument `build_graph`/`make_graph`/
`create_graph` factory if present. That is arbitrary Python execution. **Only
compile `.py` files you trust** — treat a graph file exactly like any script you
would run directly. The CLI prints a warning naming the file before executing it.
Pure-YAML flowcharts (`compile flowchart.yaml`) execute nothing.

## What the adapter recovers

| LangGraph construct | Mapped to |
|---|---|
| Graph node | IR node with `role: agent` |
| `add_edge(a, b)` | unconditional edge `a → b` |
| `add_conditional_edges(a, router, {"k": "b"})` | `a` becomes a `decision` node; one guarded edge per path-map key, with `when: "k"` |
| `add_edge(START, n)` | the IR `start` node |
| `END` (and other sinks) | terminal node with `terminal: success` |

The conversion is deterministic and validated; unsupported shapes raise a clear
`FlowchartValidationError` (for example, a conditional edge declared without a
path map, or a conditional edge straight from `START`).

## What you must fill in after converting

A LangGraph node is a Python callable, not a prompt or a dialogue turn, so two
things cannot be recovered from structure and the compiler **warns** about both:

1. **Prompts are placeholders.** Every agent node gets a `TODO:` prompt. The
   compile summary reports how many remain, and `agent2model generate` **refuses
   to run** while any `TODO:` prompt is present — replace them with real
   instructions in the emitted `flowchart.json` first, otherwise you would pay to
   generate garbage data.
2. **There are no user turns.** LangGraph graphs model the agent side only, so the
   converted flowchart has no `role: user` nodes. Generated conversations would be
   agent-only monologue. Add `role: user` nodes where the customer speaks.

## Known limitations (silently flattened)

The structural mapping intentionally does **not** model these LangGraph features;
they are dropped during conversion, so review the IR if your graph uses them:

- **`when` branch labels are descriptive only.** Synthetic-data traversal picks
  edges by configured weight, not by evaluating the router, so the recovered
  `when` labels document intent but do not drive generation.
- **Tool nodes, nested subgraphs, parallel fan-out, and state reducers** are
  flattened to plain nodes/edges.
- **Terminals all become `success`.** Abandonment/escalation terminals (which the
  evaluation failure-rate logic distinguishes) cannot be inferred from structure;
  mark them by hand in the IR if you need them.

When in doubt, compile, then open `build/mine/flowchart.json` and edit it as
normal YAML/JSON — it is just the [IR](ir-spec.md).
