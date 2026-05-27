# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/). This file is maintained automatically
by [commitizen](https://commitizen-tools.github.io/commitizen/) (`cz bump`) from
[Conventional Commits](https://www.conventionalcommits.org/); see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

<!-- cz-changelog-insertion-marker -->

## Unreleased

### Added

- Flowchart IR (Pydantic v2 schema, YAML loader, graph validator) — the public
  procedure contract.
- LangGraph `StateGraph` → IR adapter, including `.py` file discovery.
- Synthetic data generation: resumable, budgeted, prompt-cached Claude pipeline
  producing HF chat-template JSONL.
- Full-parameter SFT training pipeline (TRL `SFTTrainer`), Qwen 2.5 3B single-GPU
  and Qwen3 8B DeepSpeed ZeRO-3 presets. LoRA intentionally refused.
- Evaluation harness: 5-criterion LLM-judge, flowchart-blind user simulator,
  `in_context` / `langgraph` / `same_model_orch` baselines, SciPy stats, and a
  PDF/JSON report.
- vLLM-backed OpenAI-compatible serving.
- Cloud recipes: Modal (primary) and RunPod (secondary).
- Reproduction examples: `travel_booking`, `zoom_support`, `insurance_claims`,
  and a `langgraph_demo`, each with a README and the paper's expected numbers.
- mkdocs documentation site (quickstart, IR spec, training, evaluation, cloud,
  troubleshooting, FAQ).
- Benchmarks with regression targets (`benchmarks/targets.json`) and an e2e
  regression gate that fails on a >5% drop from the paper.
