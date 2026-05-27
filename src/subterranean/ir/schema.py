"""Pydantic v2 models for the Flowchart IR.

The YAML format these models parse is the library's public contract; see the IR
spec in the project README. Graph-level invariants (reachability, escape edges,
etc.) are enforced separately by :mod:`subterranean.ir.validator` so that this
module stays a pure structural description.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NodeRole = Literal["agent", "user", "decision"]
"""Role of a non-terminal node.

- ``agent``: the compiled model speaks this turn.
- ``user``: the user simulator speaks this turn (no flowchart knowledge).
- ``decision``: an LLM picks the next edge *at generation time only*; there is no
  runtime router.
"""

TerminalKind = Literal["success", "abandonment", "escalation"]
"""Outcome category for a terminal node."""


class Edge(BaseModel):
    """A directed edge from one node to another, optionally guarded by a condition.

    Attributes:
        to: Target node id.
        when: Natural-language condition evaluated by an LLM during data generation
            (e.g. ``"user_has_provided_info"``). ``None`` means an unconditional edge.
    """

    model_config = ConfigDict(extra="forbid")

    to: str
    when: str | None = None


class Node(BaseModel):
    """A single node in the flowchart.

    A node is either *non-terminal* (has a ``role`` and outgoing ``next`` edges) or
    *terminal* (has ``terminal`` set and no outgoing edges). The structural
    either/or is enforced here; reachability is checked by the validator.

    Attributes:
        role: Required for non-terminal nodes. ``None`` for terminals.
        prompt: Instruction template for ``agent``/``user`` nodes. Decision nodes
            and terminals omit it.
        next: Outgoing edges. A bare node id (string) is normalised to an
            unconditional :class:`Edge`.
        terminal: Outcome kind for terminal nodes; ``None`` for non-terminals.
    """

    model_config = ConfigDict(extra="forbid")

    role: NodeRole | None = None
    prompt: str | None = None
    next: list[Edge] = Field(default_factory=list)
    terminal: TerminalKind | None = None

    @field_validator("next", mode="before")
    @classmethod
    def _normalize_next(cls, value: Any) -> list[Any]:
        """Accept ``next`` as a scalar, list of ids, or list of edge mappings."""
        if value is None:
            return []
        if isinstance(value, (str, dict)):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("`next` must be a node id, an edge mapping, or a list of those")
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append({"to": item})
            else:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def _check_terminal_xor_role(self) -> Node:
        """A node is terminal xor non-terminal, with the matching fields populated."""
        is_terminal = self.terminal is not None
        if is_terminal:
            if self.role is not None:
                raise ValueError("terminal node must not declare a `role`")
            if self.next:
                raise ValueError("terminal node must not declare outgoing `next` edges")
            if self.prompt is not None:
                raise ValueError("terminal node must not declare a `prompt`")
        else:
            if self.role is None:
                raise ValueError("non-terminal node must declare a `role`")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this node ends a conversation."""
        return self.terminal is not None


class Flowchart(BaseModel):
    """A complete procedure: the top-level IR object parsed from YAML.

    Attributes:
        name: Stable identifier for the procedure (used in build paths).
        description: Human-readable one-liner.
        start: Id of the entry node.
        nodes: Mapping of node id to :class:`Node`.
        scenario_variables: Pools sampled during data generation (e.g. destinations,
            budget ranges, user personalities). Values are arbitrary YAML.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    start: str
    nodes: dict[str, Node]
    scenario_variables: dict[str, Any] = Field(default_factory=dict)

    @property
    def terminals(self) -> dict[str, Node]:
        """All terminal nodes, keyed by id."""
        return {nid: node for nid, node in self.nodes.items() if node.is_terminal}
