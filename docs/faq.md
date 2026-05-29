# FAQ

### What does "compiling a workflow into weights" actually mean?

Instead of running your procedure with an external orchestrator that injects the
flowchart into a frontier model's prompt every turn, `agent2model` generates
synthetic conversations that walk the procedure and **fine-tunes a small model**
on them. The model learns to self-orchestrate. At inference there is no router
and no flowchart in the prompt — just a small model generating turns.

### How much cheaper is it, really?

The paper reports 128–462× lower inference cost than the in-context frontier
baseline, at 87–98% of frontier quality, across the three reproductions. See
[`benchmarks/`](https://github.com/kamaalg/agent2model/tree/main/benchmarks).

### Do I need a GPU?

For `generate` and `eval`, no — those are API-bound (Anthropic). For `train` and
`serve`, yes. If you don't have one, run the whole thing on
[Modal](cloud.md) (`modal run -m agent2model.cloud.modal_app::reproduce_travel`).

### Why no LoRA?

The paper's companion (Dennis et al. 2026b, *Procedural Knowledge is Not
Low-Rank*) shows LoRA fails to internalise procedures even at high rank. Shipping
a known-broken path would hurt credibility, so v1 is full fine-tuning only and
the CLI refuses `--lora`.

### Which base models are supported?

The presets target Qwen 2.5 3B Instruct (`--size 3b`) and Qwen3 8B
(`--size 8b`), matching the paper. Any HF causal LM works via `--base`, but the
hyperparameters are tuned for those two.

### Does the flowchart end up in the training data?

No. Conversations are natural user/agent dialogue. `decision` nodes are resolved
by an LLM **at generation time only** to choose which path to walk; the structure
never appears in the JSONL. That's why the compiled model generalizes.

### How is the decision routing handled at runtime?

It isn't — there is no runtime router. `decision` nodes exist only to steer data
generation. The fine-tuned model decides what to do next on its own.

### Can I start from an existing LangGraph agent?

Yes. `agent2model compile graph.py --out build/mine` imports a `StateGraph` and
converts its structure to IR. You then fill in the `TODO` prompts and proceed
through the pipeline. See the [IR spec](ir-spec.md) and the `langgraph_demo`
example.

### How big can a procedure be?

The `insurance_claims` example is 55+ nodes with eight terminals and many
branches, and it compiles and trains. Deeper procedures need more — but not
unboundedly more — synthetic data (the paper uses ~3,000 conversations for the
55-node case).

### How do I know my reproduction matches the paper?

Run `agent2model eval` and compare `eval_report.json` to the per-criterion
targets in each example README and in `benchmarks/targets.json`. The e2e tier
(`tests/e2e/`) automates this and fails the build on a >5% regression.

### What's out of scope for v1?

LoRA, RLHF/DPO, online learning, tool use during compiled inference, multi-agent
handoffs, and CrewAI/OpenAI-Agents adapters. These are v2+.
