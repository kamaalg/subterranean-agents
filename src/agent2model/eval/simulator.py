"""Dynamic user simulator for evaluation (Claude Sonnet 4.5).

During eval each :mod:`condition <agent2model.eval.baselines>` is driven by a
*simulated user* that role-plays a customer given the sampled scenario variables
and a personality. The simulator is a separate :class:`~anthropic.AsyncAnthropic`
call that holds the conversation from the customer's side.

**The simulator has zero knowledge of the flowchart.** It is told who it is and
the scenario facts it cares about — nothing about the procedure, nodes, decision
branches, or any internal script. This is the only way the eval generalises:
the model-under-test must self-orchestrate against a user who behaves like a real
person, not a scripted walk. The no-flowchart system prompt is built the same way
as generation's ``_user_simulator_system`` (see
:mod:`agent2model.generation.generator`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock

from agent2model.generation.generator import DEFAULT_MODEL, CostTracker
from agent2model.generation.scenarios import Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types import Message

    from agent2model.generation.formatter import Turn

__all__ = ["UserSimulator", "simulator_system_prompt"]

_DONE_SENTINEL = "[[END]]"
"""Token the simulator emits when it considers the conversation finished."""


def simulator_system_prompt(scenario: Scenario, *, personality: str | None = None) -> str:
    """Build the user-simulator system prompt — personality and facts, no flowchart.

    The simulator is told who it is and the scenario facts it cares about, plus
    how to end the conversation, but is given **zero** knowledge of the procedure
    so its behaviour generalises. No node ids, ``when`` conditions, decision
    branches, or any internal script appear here.

    Args:
        scenario: The sampled scenario variables grounding the customer (a
            destination, a budget, a personality, etc.).
        personality: Optional explicit personality/style override. When omitted,
            any ``style``/``personality`` value present in ``scenario`` is used.

    Returns:
        A system prompt for a single simulated-user turn.

    Example:
        >>> p = simulator_system_prompt({"destination": "Japan"})
        >>> "flowchart" not in p.lower() and "procedure" not in p.lower()
        True
    """
    style = personality or scenario.get("style") or scenario.get("personality")
    lines = [
        "You are role-playing a CUSTOMER talking to a support agent. You have no "
        "knowledge of any internal procedure, script, workflow, or system the agent "
        "follows — you are simply a person with a need. Pursue your goal naturally, "
        "answer the agent's questions, and react like a real person would. Reply with "
        "only your own spoken turn, in a natural conversational voice — no narration, "
        "no stage directions.",
    ]
    if style:
        lines.append(f"Your personality/style: {style}.")
    if scenario:
        facts = "\n".join(f"- {k}: {v}" for k, v in scenario.items())
        lines.append("Your situation:\n" + facts)
    lines.append(
        "When your need has been fully resolved (or you have decided to give up), end "
        f"your final message with the token {_DONE_SENTINEL} on its own. Do not use that "
        "token before you are truly finished."
    )
    return "\n".join(lines)


class UserSimulator:
    """Stateless-per-call simulated customer for one evaluation scenario.

    The simulator owns an async client and a cost tracker; each
    :meth:`next_message` call produces the customer's next turn given the running
    transcript. It never sees the flowchart.

    Example:
        >>> sim = UserSimulator({"destination": "Japan"})  # doctest: +SKIP
        >>> first = await sim.next_message([])  # doctest: +SKIP
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        personality: str | None = None,
        model: str = DEFAULT_MODEL,
        client: AsyncAnthropic | None = None,
        cost: CostTracker | None = None,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the simulator.

        Args:
            scenario: Sampled scenario variables grounding the customer.
            personality: Optional explicit personality override.
            model: Anthropic model id. Defaults to :data:`DEFAULT_MODEL`.
            client: Optional pre-built async client (tests inject a mock).
            cost: Optional shared :class:`CostTracker` to fold usage into. When
                omitted a private one is created.
            max_tokens: ``max_tokens`` per API call.
        """
        self.scenario = scenario
        self.model = model
        self.client = client or AsyncAnthropic()
        self.cost = cost if cost is not None else CostTracker(model=model)
        self.max_tokens = max_tokens
        self._system = simulator_system_prompt(scenario, personality=personality)

    @property
    def system_prompt(self) -> str:
        """The simulator's system prompt (exposed for inspection/tests)."""
        return self._system

    async def next_message(self, transcript: Sequence[Turn]) -> tuple[str, bool]:
        """Produce the customer's next message given the running transcript.

        The agent is the ``assistant`` in the transcript; the customer is the
        ``user``. From the simulator's point of view the roles are flipped, so
        agent turns are presented to it as ``user`` content and its own prior
        customer turns as ``assistant`` content.

        Args:
            transcript: The conversation so far (the simulator's own turns are
                ``role == "user"``; agent turns are ``role == "assistant"``).

        Returns:
            A ``(message, done)`` pair: the customer's spoken turn (sentinel
            stripped) and whether the customer signalled the conversation is over.
        """
        messages: list[MessageParam] = []
        for turn in transcript:
            # Flip roles: the agent (assistant) addresses the customer (us), so it
            # is the "user" from the simulator's perspective.
            role: Any = "user" if turn.role == "assistant" else "assistant"
            messages.append({"role": role, "content": turn.content})
        if not messages or messages[0]["role"] != "user":
            # The conversation always opens with the customer; prime the model.
            messages.insert(0, {"role": "user", "content": "(You start the conversation.)"})

        message: Message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system,
            messages=messages,
        )
        self.cost.add_usage(message.usage)
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock)).strip()
        done = _DONE_SENTINEL in text
        if done:
            text = text.replace(_DONE_SENTINEL, "").strip()
        return text, done
