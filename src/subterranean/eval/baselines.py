"""Evaluation conditions: the compiled model and the three baselines.

Every condition implements the same job — *given a scenario, hold a full
multi-turn conversation against the user simulator and return the transcript* —
behind one :class:`Condition` protocol. The runner drives all conditions
identically; only the agent-side turn differs.

Conditions
----------
- :class:`InContextCondition` (``in_context``) — the **upper bound**: the entire
  serialised flowchart is placed in a frontier model's system prompt and it
  self-orchestrates.
- :class:`LangGraphCondition` (``langgraph``) — the industry baseline: a LangGraph
  orchestrator wraps the frontier model. Kept real but minimal (a thin documented
  adapter — see the class docstring); requires the optional ``langgraph`` install.
- :class:`SameModelOrchCondition` (``same_model_orch``) — the same base model as
  the compiled one, but orchestrated, isolating the effect of compilation.
- :class:`CompiledCondition` (``compiled``) — the served fine-tuned model, reached
  through an OpenAI-compatible client pointed at ``subterranean serve``.

Import safety: every condition is importable with only the core deps. Conditions
that need an external service (``compiled`` needs a served model; ``langgraph``
needs the langgraph install) construct their client/graph lazily, so the module
unit-tests with mocked agents and no services running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict, runtime_checkable

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock

from subterranean.eval.simulator import UserSimulator
from subterranean.exceptions import EvalError
from subterranean.generation.formatter import Conversation, Turn
from subterranean.generation.generator import DEFAULT_MODEL, CostTracker

if TYPE_CHECKING:
    from anthropic.types import Message

    from subterranean.generation.scenarios import Scenario
    from subterranean.ir.schema import Flowchart

__all__ = [
    "BASELINE_NAMES",
    "AgentResponder",
    "CompiledCondition",
    "Condition",
    "ConditionContext",
    "InContextCondition",
    "LangGraphCondition",
    "SameModelOrchCondition",
    "make_condition",
    "run_condition",
]

#: Names accepted by the ``--baselines`` CLI flag, plus the always-present compiled.
BASELINE_NAMES = ("in_context", "langgraph", "same_model_orch")

DEFAULT_MAX_TURNS = 12
"""Hard cap on conversation length (one agent turn + one user turn = 2)."""


@runtime_checkable
class AgentResponder(Protocol):
    """The agent side of a condition: produce the next agent turn.

    Implementations may call a frontier model, a LangGraph orchestrator, or a
    served compiled model. They receive the running transcript and return the
    agent's next utterance.
    """

    async def respond(self, transcript: list[Turn]) -> str:
        """Return the agent's next message given the transcript so far."""
        ...


@runtime_checkable
class Condition(Protocol):
    """A pluggable evaluation condition.

    A condition pairs a name with an :class:`AgentResponder` factory. The runner
    calls :meth:`run_scenario` for each sampled scenario.
    """

    name: str

    async def run_scenario(self, context: ConditionContext) -> Conversation:
        """Hold one full conversation for ``context`` and return the transcript."""
        ...


class ConditionContext:
    """Everything a condition needs to run one scenario.

    Attributes:
        scenario: The sampled scenario variables.
        simulator: The user simulator for this scenario (no flowchart knowledge).
        max_turns: Hard cap on total turns.
    """

    def __init__(
        self,
        scenario: Scenario,
        simulator: UserSimulator,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.scenario = scenario
        self.simulator = simulator
        self.max_turns = max_turns


async def run_condition(responder: AgentResponder, context: ConditionContext) -> Conversation:
    """Drive a conversation between a user simulator and an agent responder.

    The simulator opens; thereafter the agent and the user alternate until the
    simulator signals completion or ``context.max_turns`` is reached.

    Args:
        responder: The agent side of the condition.
        context: Scenario, simulator, and turn cap.

    Returns:
        The full conversation transcript.
    """
    transcript: list[Turn] = []
    user_text, done = await context.simulator.next_message(transcript)
    transcript.append(Turn(role="user", content=user_text))

    while len(transcript) < context.max_turns and not done:
        agent_text = await responder.respond(transcript)
        transcript.append(Turn(role="assistant", content=agent_text))
        if len(transcript) >= context.max_turns:
            break
        user_text, done = await context.simulator.next_message(transcript)
        transcript.append(Turn(role="user", content=user_text))

    return Conversation(turns=transcript)


# --------------------------------------------------------------------------- #
# Frontier-model responders (in_context / same_model_orch share the mechanism) #
# --------------------------------------------------------------------------- #


class _AnthropicResponder:
    """Agent responder backed by an Anthropic model with a system prompt."""

    def __init__(
        self,
        system_prompt: str,
        *,
        model: str,
        client: AsyncAnthropic,
        cost: CostTracker,
        max_tokens: int = 512,
        cache_system: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.client = client
        self.cost = cost
        self.max_tokens = max_tokens
        self.cache_system = cache_system

    async def respond(self, transcript: list[Turn]) -> str:
        messages: list[MessageParam] = [{"role": t.role, "content": t.content} for t in transcript]
        system: Any
        if self.cache_system:
            system = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system = self.system_prompt
        message: Message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        self.cost.add_usage(message.usage)
        return "".join(b.text for b in message.content if isinstance(b, TextBlock)).strip()


def _serialize_flowchart(flowchart: Flowchart) -> str:
    """Serialise the full flowchart into system-prompt text for in-context use."""
    lines = [
        f"You are the AGENT in the '{flowchart.name}' procedure.",
    ]
    if flowchart.description:
        lines.append(flowchart.description)
    lines.append(
        "Follow this procedure to help the customer. Self-orchestrate: decide which "
        "step you are on from the conversation and act accordingly. Speak only your "
        "own turn, as a skilled human agent would — never mention the procedure, "
        "states, or any internal step.\n\nPROCEDURE:"
    )
    lines.append(f"start: {flowchart.start}")
    for nid, node in flowchart.nodes.items():
        if node.is_terminal:
            lines.append(f"- {nid} [terminal: {node.terminal}]")
            continue
        edges = ", ".join((f"{e.to} (when {e.when})" if e.when else e.to) for e in node.next)
        prompt = (node.prompt or "").strip().replace("\n", " ")
        lines.append(f"- {nid} [{node.role}]: {prompt} -> {edges}")
    return "\n".join(lines)


class InContextCondition:
    """Upper-bound baseline: full flowchart in a frontier model's system prompt.

    The entire serialised flowchart is given to a frontier model
    (``claude-sonnet-4-5`` by default) which self-orchestrates the procedure. This
    is the quality ceiling the compiled model is measured against.
    """

    name = "in_context"

    def __init__(
        self,
        flowchart: Flowchart,
        *,
        model: str = DEFAULT_MODEL,
        client: AsyncAnthropic | None = None,
        cost: CostTracker | None = None,
    ) -> None:
        self.flowchart = flowchart
        self.model = model
        self.client = client or AsyncAnthropic()
        self.cost = cost if cost is not None else CostTracker(model=model)
        self._system = _serialize_flowchart(flowchart)

    async def run_scenario(self, context: ConditionContext) -> Conversation:
        responder = _AnthropicResponder(
            self._system, model=self.model, client=self.client, cost=self.cost
        )
        return await run_condition(responder, context)


class SameModelOrchCondition:
    """Same base model as the compiled one, but orchestrated in-context.

    Isolates the effect of compilation: the *only* difference from
    ``in_context`` is the model id (the small base model rather than a frontier
    model), reached here through the same orchestration mechanism. By default the
    base model is also served through an OpenAI-compatible endpoint, but for the
    unit-testable path it shares the Anthropic responder so an injected mock can
    drive it; production points ``--served-url`` at the base model.
    """

    name = "same_model_orch"

    def __init__(
        self,
        flowchart: Flowchart,
        *,
        model: str,
        client: AsyncAnthropic | None = None,
        cost: CostTracker | None = None,
    ) -> None:
        self.flowchart = flowchart
        self.model = model
        self.client = client or AsyncAnthropic()
        self.cost = cost if cost is not None else CostTracker(model=model)
        self._system = _serialize_flowchart(flowchart)

    async def run_scenario(self, context: ConditionContext) -> Conversation:
        responder = _AnthropicResponder(
            self._system, model=self.model, client=self.client, cost=self.cost
        )
        return await run_condition(responder, context)


# --------------------------------------------------------------------------- #
# LangGraph baseline (industry orchestrator)                                   #
# --------------------------------------------------------------------------- #


class _LangGraphResponder:
    """Agent responder backed by a compiled LangGraph orchestrator.

    Deliberately minimal: it builds a single-node ``StateGraph`` whose node calls
    the frontier model with the in-context system prompt. This is a *real*
    LangGraph orchestration (the model is invoked through a compiled graph, the
    industry-standard pattern) without re-implementing a multi-node router, which
    would not change what is being measured — that the orchestration is external.
    """

    def __init__(
        self,
        system_prompt: str,
        *,
        model: str,
        client: AsyncAnthropic,
        cost: CostTracker,
        max_tokens: int = 512,
    ) -> None:
        self._inner = _AnthropicResponder(
            system_prompt, model=model, client=client, cost=cost, max_tokens=max_tokens
        )
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - optional extra
            raise EvalError(
                "The 'langgraph' baseline needs the optional langgraph package. "
                "Install it with `pip install langgraph`, or drop 'langgraph' from "
                "--baselines."
            ) from exc

        responder = self._inner

        class _State(TypedDict):
            transcript: list[Turn]
            reply: NotRequired[str]

        async def agent_node(state: _State) -> dict[str, Any]:
            reply = await responder.respond(state["transcript"])
            return {"reply": reply}

        graph = StateGraph(_State)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        return graph.compile()

    async def respond(self, transcript: list[Turn]) -> str:
        result = await self._graph.ainvoke({"transcript": transcript})
        return str(result["reply"])


class LangGraphCondition:
    """Industry baseline: a LangGraph orchestrator wrapping the frontier model.

    Thin but real (see :class:`_LangGraphResponder`): the frontier model is
    invoked through a compiled LangGraph graph rather than directly, matching how
    teams run agents today. Requires the optional ``langgraph`` install at run
    time; the class itself imports without it.
    """

    name = "langgraph"

    def __init__(
        self,
        flowchart: Flowchart,
        *,
        model: str = DEFAULT_MODEL,
        client: AsyncAnthropic | None = None,
        cost: CostTracker | None = None,
        responder: AgentResponder | None = None,
    ) -> None:
        self.flowchart = flowchart
        self.model = model
        self.client = client or AsyncAnthropic()
        self.cost = cost if cost is not None else CostTracker(model=model)
        self._system = _serialize_flowchart(flowchart)
        # Allow tests to inject a responder so the langgraph import is not required.
        self._responder = responder

    async def run_scenario(self, context: ConditionContext) -> Conversation:
        responder = self._responder or _LangGraphResponder(
            self._system, model=self.model, client=self.client, cost=self.cost
        )
        return await run_condition(responder, context)


# --------------------------------------------------------------------------- #
# Compiled condition (the served fine-tuned model)                            #
# --------------------------------------------------------------------------- #


class _OpenAICompatResponder:
    """Agent responder talking to an OpenAI-compatible chat endpoint.

    Used for the compiled (served) model: ``subterranean serve`` exposes
    ``/v1/chat/completions``, so we drive it with an async OpenAI client pointed
    at that base URL.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str | None,
        cost: CostTracker,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.cost = cost
        self.max_tokens = max_tokens

    async def respond(self, transcript: list[Turn]) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend({"role": t.role, "content": t.content} for t in transcript)
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.cost.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.cost.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.cost.api_calls += 1
        return response.choices[0].message.content or ""


class CompiledCondition:
    """The served compiled (fine-tuned) model, via an OpenAI-compatible endpoint.

    The compiled model self-orchestrates from its weights, so **no flowchart is
    placed in its prompt** — that is the entire point of compilation. It is
    reached through an OpenAI-compatible client pointed at ``subterranean serve``
    (``served_url``). Requires the optional ``openai`` package and a running
    served model at run time; the class imports without either.
    """

    name = "compiled"

    def __init__(
        self,
        *,
        model: str,
        served_url: str,
        api_key: str = "EMPTY",
        client: Any | None = None,
        cost: CostTracker | None = None,
        responder: AgentResponder | None = None,
    ) -> None:
        self.model = model
        self.served_url = served_url
        self.api_key = api_key
        self.cost = cost if cost is not None else CostTracker(model=model)
        self._client = client
        self._responder = responder

    def _make_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise EvalError(
                "The 'compiled' condition needs the optional openai package to talk "
                "to the served model. Install it with `pip install -e '.[openai]'`."
            ) from exc
        return AsyncOpenAI(base_url=self.served_url, api_key=self.api_key)

    async def run_scenario(self, context: ConditionContext) -> Conversation:
        responder = self._responder
        if responder is None:
            client = self._client or self._make_client()
            responder = _OpenAICompatResponder(
                client, model=self.model, system_prompt=None, cost=self.cost
            )
        return await run_condition(responder, context)


def make_condition(
    name: str,
    flowchart: Flowchart,
    *,
    frontier_model: str = DEFAULT_MODEL,
    base_model: str = "qwen2.5-3b",
    compiled_model: str = "compiled",
    served_url: str | None = None,
    client: AsyncAnthropic | None = None,
) -> Condition:
    """Construct a condition by name.

    Args:
        name: One of ``in_context``, ``langgraph``, ``same_model_orch``,
            ``compiled``.
        flowchart: The compiled flowchart (serialised into the baselines' prompts).
        frontier_model: Frontier model id for ``in_context``/``langgraph``.
        base_model: Base model id for ``same_model_orch``.
        compiled_model: Served model id for ``compiled``.
        served_url: OpenAI-compatible base URL for ``compiled`` (required for it).
        client: Optional shared Anthropic client for the frontier conditions.

    Returns:
        The constructed :class:`Condition`.

    Raises:
        EvalError: If ``name`` is unknown, or ``compiled`` is requested without a
            ``served_url``.
    """
    if name == "in_context":
        return InContextCondition(flowchart, model=frontier_model, client=client)
    if name == "langgraph":
        return LangGraphCondition(flowchart, model=frontier_model, client=client)
    if name == "same_model_orch":
        return SameModelOrchCondition(flowchart, model=base_model, client=client)
    if name == "compiled":
        if not served_url:
            raise EvalError(
                "The 'compiled' condition requires --served-url pointing at a running "
                "`subterranean serve` endpoint."
            )
        return CompiledCondition(model=compiled_model, served_url=served_url)
    raise EvalError(
        f"Unknown condition '{name}'. Valid conditions: "
        f"{', '.join((*BASELINE_NAMES, 'compiled'))}."
    )
