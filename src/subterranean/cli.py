"""Typer CLI — the primary user entry point.

The canonical user journey is ``compile`` → ``generate`` → ``train`` →
``eval``/``serve``. Each command prints the expected LLM cost before running and
the actual cost after, and exits with an actionable message on typed failures.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from subterranean.adapters.langgraph import (
    flowchart_from_stategraph,
    load_stategraph_from_pyfile,
)
from subterranean.exceptions import (
    FlowchartValidationError,
    GenerationBudgetExceeded,
    ServingError,
    TrainingDivergedError,
)
from subterranean.generation.formatter import write_dataset
from subterranean.generation.generator import (
    DEFAULT_MODEL,
    ConversationGenerator,
    GenerationConfig,
    estimate_cost,
)
from subterranean.ir.loader import load_flowchart
from subterranean.ir.schema import Flowchart
from subterranean.ir.validator import validate
from subterranean.logging import configure_logging, logger
from subterranean.training.config import DENNIS_2026B, TrainingConfig

app = typer.Typer(
    name="subterranean",
    help="Compile agentic workflows into LLM weights.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging.")] = False,
) -> None:
    """Configure global logging before any command runs."""
    configure_logging(verbose=verbose)


@app.command()
def compile(
    source: Annotated[
        Path, typer.Argument(help="Flowchart YAML, or a .py file defining a LangGraph graph.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Build directory for the compiled IR.")],
) -> None:
    """Validate a workflow and emit the canonical IR.

    Loads a YAML flowchart or a LangGraph ``.py`` source, enforces every graph
    invariant, and writes the normalised IR to ``<out>/flowchart.json``. IR
    derived from LangGraph contains TODO placeholder prompts (LangGraph nodes
    carry no natural-language instructions); these still validate and should be
    filled in before generating data.
    """
    try:
        if source.suffix == ".py":
            graph = load_stategraph_from_pyfile(source)
            flowchart = flowchart_from_stategraph(graph, name=source.stem)
        else:
            flowchart = load_flowchart(source)
        validate(flowchart)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    out.mkdir(parents=True, exist_ok=True)
    ir_path = out / "flowchart.json"
    ir_path.write_text(
        json.dumps(flowchart.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    n_nodes = len(flowchart.nodes)
    n_terminals = len(flowchart.terminals)
    logger.info(
        f"Compiled '{flowchart.name}': {n_nodes} nodes, {n_terminals} terminals → {ir_path}"
    )


def _load_compiled_flowchart(build_dir: Path) -> Flowchart:
    """Load and graph-validate the compiled ``flowchart.json`` from a build dir."""
    ir_path = build_dir / "flowchart.json"
    if not ir_path.exists():
        logger.error(f"No compiled flowchart at {ir_path}. Run `subterranean compile` first.")
        raise typer.Exit(code=1)
    flowchart = Flowchart.model_validate(json.loads(ir_path.read_text(encoding="utf-8")))
    validate(flowchart)
    return flowchart


@app.command()
def generate(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the compiled flowchart.json.")
    ],
    n: Annotated[int, typer.Option("--n", help="Number of conversations to generate.")] = 100,
    model: Annotated[str, typer.Option("--model", help="Anthropic model id.")] = DEFAULT_MODEL,
    budget: Annotated[
        float, typer.Option("--budget", help="Hard USD spending cap; generation stops if hit.")
    ] = 50.0,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed for reproducibility.")] = 0,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Maximum in-flight API calls.")
    ] = 10,
) -> None:
    """Generate synthetic training data by walking the compiled flowchart.

    Reads ``<BUILD_DIR>/flowchart.json``, samples ``--n`` conversations via
    Claude, prints the expected cost before starting and the actual cost after,
    and writes the HF chat-template dataset to ``<BUILD_DIR>/dataset.jsonl``.
    Generation is resumable and stops if the ``--budget`` cap is reached.
    """
    try:
        flowchart = _load_compiled_flowchart(build_dir)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    config = GenerationConfig(
        n=n, model=model, budget_usd=budget, seed=seed, max_concurrent=max_concurrent
    )
    expected = estimate_cost(config)
    logger.info(f"Expected cost for {n} conversations with {model}: ~${expected:.2f}")
    if expected > budget:
        logger.warning(
            f"Expected cost ~${expected:.2f} exceeds the ${budget:.2f} budget; "
            "generation may stop before completing all conversations."
        )

    generator = ConversationGenerator(flowchart, config)
    try:
        conversations = asyncio.run(generator.run(build_dir))
    except GenerationBudgetExceeded as exc:
        logger.error(str(exc))
        logger.error(f"Actual cost when stopped: ${generator.cost.cost_usd:.4f}")
        raise typer.Exit(code=1) from exc

    dataset_path = build_dir / "dataset.jsonl"
    written = write_dataset(conversations, dataset_path)
    logger.info(f"Actual cost: ${generator.cost.cost_usd:.4f}")
    logger.info(f"Wrote {written} conversations to {dataset_path}")


@app.command()
def train(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the generated dataset.jsonl.")
    ],
    base: Annotated[
        str | None,
        typer.Option("--base", help="HF base model id. Defaults to the size preset's model."),
    ] = None,
    size: Annotated[
        str, typer.Option("--size", help="Model size preset: '3b' (single-GPU) or '8b' (ZeRO-3).")
    ] = "3b",
    epochs: Annotated[
        int | None,
        typer.Option("--epochs", help="Training epochs. Defaults to the preset (3B: 20, 8B: 10)."),
    ] = None,
    lora: Annotated[
        bool,
        typer.Option(
            "--lora/--no-lora",
            help="LoRA is NOT supported; full fine-tuning only. Passing --lora is refused.",
        ),
    ] = False,
) -> None:
    """Fine-tune a base model on generated data with the paper's recipe.

    Reads ``<BUILD_DIR>/dataset.jsonl`` (HF chat-template JSONL from
    ``subterranean generate``), builds a :class:`TrainingConfig` from the chosen
    ``--size`` preset, and runs full-parameter SFT, saving the best checkpoint
    (by held-out eval loss) to ``<BUILD_DIR>/model/best``.

    Full fine-tuning only: ``--lora`` is refused with a link to the companion
    paper. The heavy ML stack is GPU-only and not installed locally; if it is
    missing the command exits with an install hint rather than crashing.
    """
    size = size.lower()
    if size not in {"3b", "8b"}:
        logger.error(f"Unknown --size '{size}'. Use '3b' or '8b'.")
        raise typer.Exit(code=2)

    if lora:
        logger.error(
            "LoRA is not supported in subterranean v1: it fails to internalise procedural "
            f"workflows. See {DENNIS_2026B}. Re-run without --lora to use full fine-tuning."
        )
        raise typer.Exit(code=2)

    dataset_path = build_dir / "dataset.jsonl"
    if not dataset_path.exists():
        logger.error(
            f"No dataset at {dataset_path}. Run `subterranean generate {build_dir}` first."
        )
        raise typer.Exit(code=1)

    output_dir = str(build_dir / "model")
    overrides: dict[str, Any] = {}
    if base is not None:
        overrides["base_model"] = base
    if epochs is not None:
        overrides["epochs"] = epochs
    if size == "3b":
        config = TrainingConfig.for_3b(output_dir, **overrides)
    else:
        config = TrainingConfig.for_8b(output_dir, **overrides)

    logger.info(
        f"Training plan: {config.base_model} ({config.size}), {config.epochs} epochs, "
        f"effective batch size {config.effective_batch_size}, lr {config.learning_rate} "
        f"({config.lr_scheduler_type}). Best checkpoint by held-out eval loss "
        f"({config.eval_split:.0%} split) → {output_dir}/best."
    )
    if size == "8b":
        logger.info(
            f"8B uses DeepSpeed ZeRO-3 across {config.num_gpus} GPUs "
            "(launch via `accelerate launch`, e.g. on a Modal 8x A100 host)."
        )

    # Lazy import: keeps `subterranean train --help` working without the ML stack.
    from subterranean.training.trainer import train as run_training

    try:
        best = run_training(config, dataset_path)
    except RuntimeError as exc:
        # Raised when the optional [train] extra / GPU host is unavailable.
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except TrainingDivergedError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info(f"Done. Best checkpoint (eval_loss={best.eval_loss}) saved to {best.path}.")


@app.command()
def eval(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the compiled flowchart.json.")
    ],
    baselines: Annotated[
        str,
        typer.Option(
            "--baselines",
            help="Comma-separated baselines: in_context, langgraph, same_model_orch.",
        ),
    ] = "in_context,langgraph",
    n: Annotated[int, typer.Option("--n", help="Scenarios per condition.")] = 200,
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Anthropic model id for the LLM judge.")
    ] = DEFAULT_MODEL,
    budget: Annotated[
        float, typer.Option("--budget", help="Hard USD spending cap across all LLM calls.")
    ] = 50.0,
    served_url: Annotated[
        str | None,
        typer.Option(
            "--served-url",
            help="OpenAI-compatible base URL of a `subterranean serve` endpoint; "
            "adds the served compiled model as a condition.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed.")] = 0,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Concurrent scenario evaluations.")
    ] = 10,
) -> None:
    """Evaluate a compiled model against baselines with the paper's rubric.

    Samples ``--n`` scenarios, runs each condition (the baselines, plus the served
    ``compiled`` model when ``--served-url`` is given) against a flowchart-blind
    user simulator, judges every conversation on the 5-criterion rubric, computes
    bootstrap CIs / Wilcoxon + Holm-Bonferroni significance / failure rates / cost,
    and writes ``<BUILD_DIR>/eval_report.pdf`` and ``eval_report.json``. Prints
    the expected cost before starting and the actual cost after.
    """
    import asyncio

    from subterranean.eval.baselines import make_condition
    from subterranean.eval.judge import Judge, JudgeConfig
    from subterranean.eval.report import write_json_report, write_pdf_report
    from subterranean.eval.runner import EvalConfig, EvalRunner, estimate_eval_cost
    from subterranean.exceptions import EvalBudgetExceeded, EvalError

    try:
        flowchart = _load_compiled_flowchart(build_dir)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    names = [b.strip() for b in baselines.split(",") if b.strip()]
    if served_url:
        names.append("compiled")
    try:
        conditions = [make_condition(name, flowchart, served_url=served_url) for name in names]
    except EvalError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=2) from exc

    config = EvalConfig(
        n=n,
        budget_usd=budget,
        seed=seed,
        max_concurrent=max_concurrent,
        judge=JudgeConfig(model=judge_model),
    )
    expected = estimate_eval_cost(config, len(conditions))
    logger.info(
        f"Evaluating {len(conditions)} conditions ({', '.join(names)}) x {n} scenarios. "
        f"Expected cost: ~${expected:.2f}"
    )
    if expected > budget:
        logger.warning(
            f"Expected cost ~${expected:.2f} exceeds the ${budget:.2f} budget; "
            "the run may stop before completing."
        )

    runner = EvalRunner(flowchart, conditions, config, judge=Judge(config.judge))
    try:
        result = asyncio.run(runner.run())
    except EvalBudgetExceeded as exc:
        logger.error(str(exc))
        logger.error(f"Actual cost when stopped: ${runner.cost.cost_usd:.4f}")
        raise typer.Exit(code=1) from exc

    logger.info(f"Actual cost: ${result.total_cost_usd:.4f}")
    json_path = write_json_report(result, build_dir / "eval_report.json")
    logger.info(f"Wrote {json_path}")
    try:
        pdf_path = write_pdf_report(result, build_dir / "eval_report.pdf")
        logger.info(f"Wrote {pdf_path}")
    except EvalError as exc:
        logger.warning(str(exc))


@app.command()
def serve(
    build_dir: Annotated[
        Path,
        typer.Argument(help="Build directory holding the compiled model (or a model dir)."),
    ],
    port: Annotated[int, typer.Option("--port", help="TCP port to bind.")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "0.0.0.0",
    model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Public model id exposed via the API."),
    ] = None,
) -> None:
    """Serve a compiled model via an OpenAI-compatible vLLM endpoint.

    Resolves the servable checkpoint under ``<BUILD_DIR>`` (prefers
    ``<BUILD_DIR>/best`` from ``subterranean train``, falling back to the
    directory itself), prints what it is about to serve and on what address,
    then launches vLLM's OpenAI-compatible API server (``/v1/chat/completions``,
    ``/v1/models``). vLLM is GPU/CUDA-only; if it is not installed the command
    exits with an actionable install hint rather than crashing.
    """
    from subterranean.serve.vllm_server import resolve_model_path
    from subterranean.serve.vllm_server import serve as run_server

    try:
        model_path = resolve_model_path(build_dir)
    except ServingError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info(
        f"Serving '{model_name or model_path}' on http://{host}:{port} "
        "(OpenAI-compatible: /v1/chat/completions, /v1/models)."
    )
    try:
        run_server(model_path, port=port, host=host, served_model_name=model_name)
    except ServingError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
