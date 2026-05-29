# IR spec reference

The Flowchart IR is the library's public contract: a YAML description of a
procedure. `agent2model compile` parses it, validates it, and emits the
canonical normalised form to `<out>/flowchart.json`.

## Top-level fields

```yaml
name: travel_booking            # required — stable id, used in build paths
description: Help a customer …   # optional one-liner
start: greet                     # required — id of the entry node
nodes: { … }                     # required — map of node id → node
scenario_variables: { … }        # optional — pools sampled at data-gen time
```

## Nodes

A node is **either** non-terminal (has a `role` and outgoing `next` edges)
**or** terminal (has `terminal` set, no role/prompt/next).

### Non-terminal nodes

```yaml
greet:
  role: agent          # agent | user | decision
  prompt: |            # required for agent/user; omit for decision
    Warmly greet the customer and ask what they need help with.
  next: [gather_preferences]
```

The three roles:

| Role | Who speaks | Notes |
|---|---|---|
| `agent` | the compiled model | needs a `prompt` (the instruction for the turn) |
| `user` | the user simulator | needs a `prompt`; the simulator has **no** flowchart knowledge |
| `decision` | nobody — a router | no `prompt`; an LLM picks the edge **at data-generation time only**. There is no runtime router. |

### Edges (`next`)

`next` accepts a scalar id, a list of ids, or a list of edge mappings. These are
equivalent ways to write an unconditional edge:

```yaml
next: gather_preferences
next: [gather_preferences]
next:
  - to: gather_preferences
```

Conditional edges carry a natural-language `when` label, evaluated by an LLM
during generation (typically on `decision` nodes):

```yaml
assess_readiness:
  role: decision
  next:
    - to: present_options
      when: user has provided destination, dates, budget, and group size
    - to: gather_preferences
      when: one or more required details are still missing
```

### Terminal nodes

```yaml
booking_confirmed:
  terminal: success      # success | abandonment | escalation
```

Terminals have no `role`, `prompt`, or `next`.

## Scenario variables

Arbitrary YAML pools sampled per conversation at generation time (personalities,
domains, ranges, …). They never constrain the graph; they enrich the synthetic
data.

```yaml
scenario_variables:
  destination_pool: [Japan, Italy, Iceland]
  budget_range: [500, 5000]
  user_styles: [decisive, indecisive, skeptical]
```

## Invariants

Enforced by `agent2model.ir.validator`; violations print one human-readable line
each and exit non-zero:

1. `start` names an existing node.
2. Every edge target names an existing node.
3. Every non-terminal node has at least one outgoing edge.
4. Every terminal node is reachable from `start`.
5. From every reachable node a terminal is reachable — **cycles are allowed but
   must contain a terminal-reaching escape edge** (no dead-end traps).

## Compiling from LangGraph

You don't have to write YAML by hand. Point `compile` at a `.py` file that
defines a LangGraph `StateGraph`:

```bash
agent2model compile path/to/graph.py --out build/mine
```

The adapter recovers structure only — every agent node gets a `TODO` placeholder
prompt you fill in before generating data. See the
[`langgraph_demo` example](https://github.com/kamaalg/agent2model/tree/main/examples/langgraph_demo)
for the discovery contract and mapping rules.

## Example

A complete, valid flowchart lives at
[`examples/travel_booking/flowchart.yaml`](https://github.com/kamaalg/agent2model/blob/main/examples/travel_booking/flowchart.yaml).
The larger `insurance_claims` example (55+ nodes, eight terminals) shows how the
same contract scales to deep, many-branch procedures.
