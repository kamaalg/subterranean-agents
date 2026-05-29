"""Async turn-by-turn synthetic conversation generation via Claude.

This is the core of Phase 2. For each conversation we:

1. Sample a ``start``-to-terminal path through the flowchart
   (:mod:`agent2model.generation.traversal`).
2. Sample scenario variables (:mod:`agent2model.generation.scenarios`).
3. Walk the path turn by turn. At each ``agent`` or ``user`` node we format that
   node's prompt with the scenario and the running history and call the Anthropic
   API for the turn's content. (``decision`` nodes carry no prompt — the path was
   already chosen at sampling time, so they are skipped during the walk.)

The user simulator is told its personality and scenario variables but **never**
the flowchart — that separation is what lets the eval generalise. The agent
turns are produced under a system prompt that carries the static flowchart spec,
which is identical across the whole run and therefore prompt-cached for a large
cost saving.

The generator is budgeted (a USD hard-stop raising
:class:`~agent2model.exceptions.GenerationBudgetExceeded`), resumable (a
checkpoint of completed conversations on disk), concurrent (an
``asyncio.Semaphore``), and never blocks the API without a Rich progress bar.
Token usage and cost are written to ``build/<name>/cost.json``.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from agent2model.exceptions import GenerationBudgetExceeded
from agent2model.generation.formatter import Conversation, Turn
from agent2model.generation.scenarios import Scenario, sample_scenario
from agent2model.generation.traversal import TraversalConfig, sample_path
from agent2model.ir.schema import Flowchart, Node
from agent2model.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types import Message

DEFAULT_MODEL = "claude-sonnet-4-5"
"""Default generation model — the paper used Claude Sonnet 4.5."""

# USD per 1M tokens, per model. ``input``/``output`` are the base rates; cache
# writes cost 1.25x input and cache reads cost 0.1x input (Anthropic pricing).
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}
_FALLBACK_PRICING = {"input": 3.0, "output": 15.0}
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


class GenerationConfig(BaseModel):
    """Configuration for a synthetic data-generation run.

    Attributes:
        n: Number of conversations to generate.
        model: Anthropic model id. Defaults to :data:`DEFAULT_MODEL`.
        budget_usd: Hard spending cap in USD. When the accumulated cost would
            exceed this, generation stops and
            :class:`~agent2model.exceptions.GenerationBudgetExceeded` is raised.
        seed: Base RNG seed; conversation ``i`` derives a deterministic per-item
            seed from it, so a whole run is reproducible.
        max_concurrent: Maximum in-flight API calls (an ``asyncio.Semaphore``).
        max_tokens: ``max_tokens`` passed to each API call.
        checkpoint_every: Persist progress to ``generation_state.json`` after this
            many completed conversations (and always at the end).
        traversal: Edge-weighting configuration for path sampling.
    """

    model_config = ConfigDict(extra="forbid")

    n: int = Field(gt=0)
    model: str = DEFAULT_MODEL
    budget_usd: float = Field(gt=0.0)
    seed: int = 0
    max_concurrent: int = Field(default=10, gt=0)
    max_tokens: int = Field(default=1024, gt=0)
    checkpoint_every: int = Field(default=50, gt=0)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)


class CostTracker(BaseModel):
    """Accumulates token usage and dollar cost across a generation run.

    Attributes:
        model: Model id used to price tokens.
        input_tokens: Uncached input tokens billed at the base input rate.
        output_tokens: Output tokens billed at the base output rate.
        cache_creation_input_tokens: Tokens written to the prompt cache (1.25x).
        cache_read_input_tokens: Tokens served from the prompt cache (0.1x).
        api_calls: Number of API calls accounted for.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    api_calls: int = 0

    def add_usage(self, usage: Any) -> None:
        """Fold one API response's ``usage`` block into the running totals."""
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.api_calls += 1

    @property
    def cost_usd(self) -> float:
        """Total cost in USD for the accumulated usage."""
        rates = _PRICING.get(self.model, _FALLBACK_PRICING)
        per_token_in = rates["input"] / 1_000_000
        per_token_out = rates["output"] / 1_000_000
        return (
            self.input_tokens * per_token_in
            + self.output_tokens * per_token_out
            + self.cache_creation_input_tokens * per_token_in * _CACHE_WRITE_MULTIPLIER
            + self.cache_read_input_tokens * per_token_in * _CACHE_READ_MULTIPLIER
        )

    def to_report(self) -> dict[str, Any]:
        """Serialise totals plus the derived cost for ``cost.json``."""
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def estimate_cost(config: GenerationConfig, *, turns_per_convo: float = 8.0) -> float:
    """Roughly estimate a run's cost in USD before it starts.

    The estimate assumes a modest fixed token footprint per turn; it exists so
    the CLI can print an expected cost up front, not to be precise.

    Args:
        config: The run configuration (``n`` and ``model`` drive the estimate).
        turns_per_convo: Assumed average API calls per conversation.

    Returns:
        Estimated total cost in USD.

    Example:
        >>> estimate_cost(GenerationConfig(n=100, budget_usd=10))  # doctest: +SKIP
        2.4
    """
    rates = _PRICING.get(config.model, _FALLBACK_PRICING)
    calls = config.n * turns_per_convo
    # Assume ~600 cached+uncached input tokens and ~120 output tokens per call.
    input_cost = calls * 600 * rates["input"] / 1_000_000
    output_cost = calls * 120 * rates["output"] / 1_000_000
    return input_cost + output_cost


def build_system_prompt(flowchart: Flowchart) -> str:
    """Build the static agent-side system prompt carrying the flowchart spec.

    This text is identical for every agent turn across the entire run, which is
    exactly what makes it a good prompt-cache prefix. It is **only** used for
    ``agent`` turns — the user simulator never sees it.

    Args:
        flowchart: The flowchart being compiled.

    Returns:
        A system prompt describing the procedure the agent must follow.
    """
    lines = [
        f"You are role-playing the AGENT in a '{flowchart.name}' procedure.",
    ]
    if flowchart.description:
        lines.append(flowchart.description)
    lines.append(
        "Follow the procedure naturally. Produce only the agent's spoken turn — "
        "no narration, no stage directions, no mention of any internal steps, "
        "states, or flowchart. Speak as a skilled human agent would."
    )
    return "\n".join(lines)


def _format_prompt(node: Node, scenario: Scenario) -> str:
    """Interpolate scenario variables into a node's prompt template.

    Unknown placeholders are left intact so a stray brace never crashes a run.
    """
    template = node.prompt or ""
    try:
        return template.format(**scenario)
    except (KeyError, IndexError, ValueError):
        return template


def _history_text(turns: Sequence[Turn]) -> str:
    """Render the running dialogue as plain ``Customer:``/``Agent:`` lines."""
    if not turns:
        return "(no conversation yet)"
    label = {"user": "Customer", "assistant": "Agent"}
    return "\n".join(f"{label[t.role]}: {t.content}" for t in turns)


def _user_simulator_system(scenario: Scenario) -> str:
    """System prompt for a user-simulator turn — personality, no flowchart.

    The simulator is told who it is and the scenario facts it cares about, but is
    given zero knowledge of the procedure so its behaviour generalises.
    """
    lines = [
        "You are role-playing a CUSTOMER talking to a support agent. "
        "You have no knowledge of any internal procedure, script, or system — "
        "you are simply a person with a need. Reply with only your own spoken "
        "turn, in a natural conversational voice.",
    ]
    if scenario:
        facts = "\n".join(f"- {k}: {v}" for k, v in scenario.items())
        lines.append("Your situation and personality:\n" + facts)
    return "\n".join(lines)


class ConversationGenerator:
    """Generates synthetic conversations for one flowchart, concurrently.

    The generator owns an :class:`~anthropic.AsyncAnthropic` client and a
    semaphore; it samples a path and scenario per conversation, walks the path
    calling the API for each speaking turn, and tracks cost against the budget.

    Example:
        >>> gen = ConversationGenerator(fc, GenerationConfig(n=10, budget_usd=5))  # doctest: +SKIP
        >>> convos = await gen.run(Path("build/travel"))  # doctest: +SKIP
    """

    def __init__(
        self,
        flowchart: Flowchart,
        config: GenerationConfig,
        *,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            flowchart: A flowchart that has already passed graph validation.
            config: Run configuration.
            client: Optional pre-built async client (tests inject a mock). When
                omitted a default :class:`~anthropic.AsyncAnthropic` is created.
        """
        self.flowchart = flowchart
        self.config = config
        self.client = client or AsyncAnthropic()
        self.cost = CostTracker(model=config.model)
        self._system_prompt = build_system_prompt(flowchart)
        self._budget_hit = False

    async def _call(self, system: str, user_content: str, *, cache_system: bool) -> str:
        """Make one API call and fold its usage into the cost tracker.

        Args:
            system: System prompt for this turn.
            user_content: The user-role content for this turn.
            cache_system: When True, mark the system prompt with ``cache_control``
                so the static agent spec is cached across the run.

        Returns:
            The concatenated text of the response.
        """
        system_param: Any
        if cache_system:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        message: Message = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": user_content}],
        )
        self.cost.add_usage(message.usage)
        self._check_budget()
        parts = [block.text for block in message.content if isinstance(block, TextBlock)]
        return "".join(parts).strip()

    def _check_budget(self) -> None:
        """Raise if the accumulated cost has exceeded the configured budget."""
        if self.cost.cost_usd > self.config.budget_usd:
            self._budget_hit = True
            raise GenerationBudgetExceeded(self.cost.cost_usd, self.config.budget_usd)

    async def generate_one(self, index: int) -> Conversation:
        """Generate a single conversation deterministically from its index.

        Args:
            index: Conversation index in ``[0, n)``. Combined with the config seed
                to derive a reproducible per-conversation RNG.

        Returns:
            The generated :class:`Conversation`.
        """
        rng = random.Random(f"{self.config.seed}:{index}")
        path = sample_path(self.flowchart, rng, config=self.config.traversal)
        scenario = sample_scenario(self.flowchart, rng)

        turns: list[Turn] = []
        for node_id in path:
            node = self.flowchart.nodes[node_id]
            if node.role == "agent":
                prompt = (
                    f"Conversation so far:\n{_history_text(turns)}\n\n"
                    f"Your instruction for this turn:\n{_format_prompt(node, scenario)}\n\n"
                    "Write the agent's next turn."
                )
                text = await self._call(self._system_prompt, prompt, cache_system=True)
                turns.append(Turn(role="assistant", content=text))
            elif node.role == "user":
                prompt = (
                    f"Conversation so far:\n{_history_text(turns)}\n\n"
                    "Write your next message as the customer."
                )
                text = await self._call(
                    _user_simulator_system(scenario), prompt, cache_system=False
                )
                turns.append(Turn(role="user", content=text))
            # decision/terminal nodes contribute no spoken turn.
        return Conversation(turns=turns)

    async def run(self, build_dir: Path) -> list[Conversation]:
        """Generate ``config.n`` conversations, resuming and checkpointing.

        Completed conversations are checkpointed to
        ``<build_dir>/generation_state.json`` every ``checkpoint_every`` items;
        on a re-run, already-completed indices are skipped. Cost is written to
        ``<build_dir>/cost.json`` whenever progress is checkpointed and at the
        end. A Rich progress bar tracks completion.

        Args:
            build_dir: Directory holding ``flowchart.json`` and receiving the
                checkpoint and cost files.

        Returns:
            All conversations (resumed + newly generated), ordered by index.

        Raises:
            GenerationBudgetExceeded: If the dollar budget is hit. Work completed
                before the stop is still checkpointed.
        """
        build_dir.mkdir(parents=True, exist_ok=True)
        state_path = build_dir / "generation_state.json"
        completed, self.cost = _load_checkpoint(state_path, self.config.model)

        remaining = [i for i in range(self.config.n) if str(i) not in completed]
        logger.info(
            f"Generating {len(remaining)} of {self.config.n} conversations "
            f"({len(completed)} already done) with {self.config.model}."
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        lock = asyncio.Lock()
        done_since_checkpoint = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "Generating conversations", total=self.config.n, completed=len(completed)
            )

            async def worker(index: int) -> None:
                nonlocal done_since_checkpoint
                async with semaphore:
                    conv = await self.generate_one(index)
                async with lock:
                    completed[str(index)] = conv.model_dump(mode="json")
                    progress.advance(task)
                    done_since_checkpoint += 1
                    if done_since_checkpoint >= self.config.checkpoint_every:
                        _save_checkpoint(state_path, completed, self.cost)
                        done_since_checkpoint = 0

            try:
                await asyncio.gather(*(worker(i) for i in remaining))
            except GenerationBudgetExceeded:
                _save_checkpoint(state_path, completed, self.cost)
                _write_cost(build_dir, self.cost)
                raise

        _save_checkpoint(state_path, completed, self.cost)
        _write_cost(build_dir, self.cost)
        logger.info(
            f"Generated {len(completed)} conversations; " f"actual cost ${self.cost.cost_usd:.4f}."
        )
        return [
            Conversation.model_validate(completed[str(i)])
            for i in range(self.config.n)
            if str(i) in completed
        ]


def _load_checkpoint(state_path: Path, model: str) -> tuple[dict[str, Any], CostTracker]:
    """Load completed conversations and restored cost totals from a checkpoint.

    Args:
        state_path: Path to ``generation_state.json``.
        model: Model id for the rebuilt cost tracker.

    Returns:
        A ``(conversations, cost_tracker)`` pair. The conversations map is keyed
        by index string; the cost tracker carries forward token totals from a
        prior run so the final ``cost.json`` reflects the whole effort.
    """
    if not state_path.exists():
        return {}, CostTracker(model=model)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    convos: dict[str, Any] = data.get("conversations", {})
    saved = data.get("cost", {})
    cost = CostTracker(
        model=model,
        input_tokens=saved.get("input_tokens", 0),
        output_tokens=saved.get("output_tokens", 0),
        cache_creation_input_tokens=saved.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=saved.get("cache_read_input_tokens", 0),
        api_calls=saved.get("api_calls", 0),
    )
    return convos, cost


def _save_checkpoint(state_path: Path, completed: dict[str, Any], cost: CostTracker) -> None:
    """Persist completed conversations and cost totals to the checkpoint file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"conversations": completed, "cost": cost.to_report()}
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_cost(build_dir: Path, cost: CostTracker) -> None:
    """Write the run's token usage and dollar cost to ``cost.json``."""
    cost_path = build_dir / "cost.json"
    cost_path.write_text(
        json.dumps(cost.to_report(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
