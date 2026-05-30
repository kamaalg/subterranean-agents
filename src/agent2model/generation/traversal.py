"""Weighted random walk through a flowchart.

Path sampling is the first step of synthetic data generation: we walk the graph
from ``start`` to a terminal, choosing one outgoing edge at each non-terminal
node. Edge selection is *weighted* so that common paths dominate the dataset
while rare paths still receive coverage, and it is fully deterministic under a
provided :class:`random.Random` so generation runs are reproducible.

Crucially, ``decision`` nodes are resolved **here**, at generation time — the
weighted walk picks which conditional edge a decision node takes. There is no
runtime router; the compiled model self-orchestrates.
"""

from __future__ import annotations

import random
from collections import deque

from pydantic import BaseModel, ConfigDict, Field

from agent2model.exceptions import FlowchartValidationError
from agent2model.ir.schema import Edge, Flowchart


class TraversalConfig(BaseModel):
    """Configuration for the weighted random walk.

    Attributes:
        default_weight: Relative weight given to any edge that is not named in
            ``edge_weights``. Must be positive.
        edge_weights: Per-edge weight overrides keyed by ``"<from>-><to>"`` (for
            example ``"assess_readiness->present_options"``). Higher weights make
            an edge more likely; common-path edges should be weighted up so they
            dominate, while rare-path edges keep a small but non-zero weight for
            coverage.
        max_steps: Hard cap on path length. Cyclic flowcharts can in principle
            walk forever; this bounds a single sample so a pathological seed
            cannot hang generation. The walk biases toward terminals once it has
            taken more than half of ``max_steps``.
    """

    model_config = ConfigDict(extra="forbid")

    default_weight: float = Field(default=1.0, gt=0.0)
    edge_weights: dict[str, float] = Field(default_factory=dict)
    max_steps: int = Field(default=100, gt=0)


def _edge_key(src: str, edge: Edge) -> str:
    """Canonical ``"<from>-><to>"`` key for an edge weight lookup."""
    return f"{src}->{edge.to}"


def _distance_to_terminal(flowchart: Flowchart) -> dict[str, int]:
    """Shortest forward hop-distance from each node to its nearest terminal.

    Computed by a reverse BFS seeded at every terminal node, so terminals get
    distance ``0`` and every other node gets ``1 +`` the minimum distance of its
    successors. Used to bias the late-walk toward terminals by *progress* rather
    than only boosting edges that reach a terminal in a single hop — otherwise a
    valid flowchart whose only escape is several hops away can exhaust
    ``max_steps`` and raise a spurious "trap cycles" error.
    """
    # Reverse adjacency: predecessor -> list of nodes it has an edge to is the
    # forward graph; we walk it backwards from terminals.
    predecessors: dict[str, list[str]] = {nid: [] for nid in flowchart.nodes}
    for src, node in flowchart.nodes.items():
        for edge in node.next:
            if edge.to in predecessors:
                predecessors[edge.to].append(src)

    distance: dict[str, int] = {}
    queue: deque[str] = deque()
    for nid, node in flowchart.nodes.items():
        if node.is_terminal:
            distance[nid] = 0
            queue.append(nid)
    while queue:
        nid = queue.popleft()
        for pred in predecessors[nid]:
            if pred not in distance:
                distance[pred] = distance[nid] + 1
                queue.append(pred)
    return distance


def sample_path(
    flowchart: Flowchart,
    rng: random.Random,
    *,
    config: TraversalConfig | None = None,
) -> list[str]:
    """Sample a single ``start``-to-terminal path via a weighted random walk.

    At each non-terminal node one outgoing edge is chosen with probability
    proportional to its configured weight. Decision-node branches are resolved
    here. The walk is deterministic for a given ``rng`` state, so seeding the
    ``rng`` makes the whole run reproducible.

    Args:
        flowchart: A flowchart that has already passed
            :func:`agent2model.ir.validator.validate`.
        rng: Seeded random source. The same seed yields the same path.
        config: Edge-weighting and step-bound configuration. Defaults to uniform
            weights when omitted.

    Returns:
        A list of node ids beginning at ``flowchart.start`` and ending at a
        terminal node.

    Raises:
        FlowchartValidationError: If the walk reaches a non-terminal node with no
            outgoing edges, or exceeds ``config.max_steps`` without terminating
            (both indicate the flowchart was not validated first).

    Example:
        >>> import random
        >>> path = sample_path(fc, random.Random(0))
        >>> path[0] == fc.start and fc.nodes[path[-1]].is_terminal
        True
    """
    cfg = config or TraversalConfig()
    distance = _distance_to_terminal(flowchart)
    path = [flowchart.start]
    current = flowchart.start

    for step in range(cfg.max_steps):
        node = flowchart.nodes[current]
        if node.is_terminal:
            return path
        if not node.next:
            raise FlowchartValidationError(
                f"node '{current}' is non-terminal but has no outgoing edges; "
                "validate the flowchart before sampling"
            )
        edge = _choose_edge(flowchart, current, node.next, rng, cfg, step, distance)
        path.append(edge.to)
        current = edge.to

    raise FlowchartValidationError(
        f"path sampling exceeded max_steps={cfg.max_steps} without reaching a "
        f"terminal (last node '{current}'); the flowchart may contain a cycle with "
        "no escape edge toward a terminal (validate the flowchart first), or "
        "max_steps is too low for its longest path"
    )


def _choose_edge(
    flowchart: Flowchart,
    src: str,
    edges: list[Edge],
    rng: random.Random,
    cfg: TraversalConfig,
    step: int,
    distance: dict[str, int],
) -> Edge:
    """Pick one edge by weight, biasing toward terminals late in the walk.

    Once the walk has taken more than half of ``max_steps``, edges that make
    *progress* toward a terminal — i.e. whose target is strictly closer to a
    terminal than the current node (by hop-distance) — have their weight boosted
    so cyclic flowcharts converge instead of looping until the hard cap. Biasing
    by distance, not just by "reaches a terminal in one hop", lets flowcharts
    whose only escape is several hops away still terminate reliably.
    """
    bias_terminals = step > cfg.max_steps // 2
    here = distance.get(src)
    weights: list[float] = []
    for edge in edges:
        weight = cfg.edge_weights.get(_edge_key(src, edge), cfg.default_weight)
        if bias_terminals:
            target = distance.get(edge.to)
            if target is not None and (here is None or target < here):
                weight *= 10.0
        weights.append(weight)
    return rng.choices(edges, weights=weights, k=1)[0]
