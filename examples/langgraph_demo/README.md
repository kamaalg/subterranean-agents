# LangGraph demo — compile a `StateGraph` into IR

You don't have to hand-write YAML. If your procedure already exists as a
LangGraph `StateGraph`, `agent2model compile` discovers it from a `.py` file and
converts the **structure** into the Flowchart IR.

[`demo.py`](demo.py) defines a small order-support graph and exposes it as a
module-level `graph` variable (the discovery contract also accepts `app` /
`workflow` variables, or a `build_graph()` / `make_graph()` / `create_graph()`
factory).

## Compile it

```bash
agent2model compile examples/langgraph_demo/demo.py --out build/demo
```

This writes `build/demo/flowchart.json`. The adapter maps:

- each graph node → an IR `agent` node with a **`TODO` placeholder prompt**
  (LangGraph nodes are Python callables, not natural-language instructions);
- `add_edge(START, x)` → the IR `start` node;
- plain `add_edge` → unconditional edges;
- `add_conditional_edges(src, router, {label: target})` → a `decision` node whose
  branches carry the path-map keys as `when` labels (here, `triage` routes to
  `refund` / `status` / `other`, and `check_resolved` routes to
  `resolved` / `retry` / `stuck`);
- `END` → a terminal node (defaults to `success` — structure can't tell apart
  abandonment/escalation).

## Then fill in the prompts and continue

Open `build/demo/flowchart.json` (or export it back to YAML) and replace each
`TODO:` prompt with real instructions for that step. After that the rest of the
pipeline is identical to the YAML examples:

```bash
agent2model generate build/demo --n 2000 --model claude-sonnet-4-5 --budget 60
agent2model train    build/demo --base Qwen/Qwen2.5-3B-Instruct --size 3b --epochs 20
agent2model eval     build/demo --baselines in_context,langgraph --n 200
```

## Notes

- A conditional edge **without a path map** can't be represented (its targets
  aren't statically knowable) — the adapter raises a clear error asking you to
  pass a path map.
- A conditional edge from `START` is rejected; add a deterministic
  `add_edge(START, <node>)` entry node instead.
- Compiled graphs (`graph.compile()`) work too — the adapter unwraps them via
  `.builder`.
