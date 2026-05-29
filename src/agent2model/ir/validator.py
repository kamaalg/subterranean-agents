"""Graph-level validation and path utilities for the Flowchart IR.

Structural parsing is handled by the schema/loader; this module enforces the
*graph* invariants from the IR spec and provides reachability/path helpers reused
by data generation (:mod:`agent2model.generation.traversal`).

Invariants enforced by :func:`validate`:

1. ``start`` names an existing node.
2. Every edge target names an existing node.
3. Every non-terminal node has at least one outgoing edge.
4. Every terminal node is reachable from ``start``.
5. From every node reachable from ``start`` a terminal is reachable — i.e. cycles
   are allowed but must contain a terminal-reaching escape edge (no dead-end traps).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from agent2model.exceptions import FlowchartValidationError
from agent2model.ir.schema import Flowchart


def validate(flowchart: Flowchart) -> None:
    """Validate all graph-level invariants of a flowchart.

    Args:
        flowchart: The parsed flowchart to check.

    Raises:
        FlowchartValidationError: If any invariant is violated. The error
            aggregates every problem found, one human-readable line each.

    Example:
        >>> validate(load_flowchart("examples/travel_booking/flowchart.yaml"))  # no error
    """
    errors: list[str] = []
    nodes = flowchart.nodes

    # (1) start exists
    if flowchart.start not in nodes:
        errors.append(f"`start` points to unknown node '{flowchart.start}'")

    # (2) edge targets exist + (3) non-terminal nodes have outgoing edges
    for nid, node in nodes.items():
        if node.is_terminal:
            continue
        if not node.next:
            errors.append(f"non-terminal node '{nid}' has no outgoing edges")
        for edge in node.next:
            if edge.to not in nodes:
                errors.append(f"node '{nid}' has an edge to unknown node '{edge.to}'")

    # Reachability checks only make sense once start and edges are sane.
    if not errors:
        reachable = _reachable_from(flowchart, flowchart.start)

        # (4) every terminal reachable from start
        for tid in flowchart.terminals:
            if tid not in reachable:
                errors.append(f"terminal node '{tid}' is not reachable from '{flowchart.start}'")

        # (5) every reachable node can itself reach a terminal (escape-edge rule)
        can_terminate = _nodes_that_reach_terminal(flowchart)
        trapped = sorted(reachable - can_terminate)
        for nid in trapped:
            errors.append(
                f"node '{nid}' is reachable but cannot reach any terminal "
                f"(a cycle needs an escape edge)"
            )

    if errors:
        raise FlowchartValidationError(
            f"Flowchart '{flowchart.name}' is invalid:\n  - " + "\n  - ".join(errors),
            errors=errors,
        )


def _reachable_from(flowchart: Flowchart, start: str) -> set[str]:
    """Forward BFS: all node ids reachable from ``start`` (inclusive)."""
    seen: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        nid = queue.popleft()
        if nid in seen or nid not in flowchart.nodes:
            continue
        seen.add(nid)
        for edge in flowchart.nodes[nid].next:
            if edge.to not in seen:
                queue.append(edge.to)
    return seen


def _nodes_that_reach_terminal(flowchart: Flowchart) -> set[str]:
    """Reverse reachability: every node from which some terminal is reachable.

    Computed by seeding with terminals and walking edges backwards to a fixed point.
    """
    predecessors: dict[str, set[str]] = {nid: set() for nid in flowchart.nodes}
    for nid, node in flowchart.nodes.items():
        for edge in node.next:
            if edge.to in predecessors:
                predecessors[edge.to].add(nid)

    can_terminate: set[str] = set(flowchart.terminals)
    queue: deque[str] = deque(can_terminate)
    while queue:
        nid = queue.popleft()
        for pred in predecessors.get(nid, ()):
            if pred not in can_terminate:
                can_terminate.add(pred)
                queue.append(pred)
    return can_terminate


def enumerate_paths(
    flowchart: Flowchart,
    *,
    max_paths: int = 1000,
    max_revisits: int = 1,
) -> Iterator[list[str]]:
    """Enumerate node-id paths from ``start`` to a terminal.

    Cycles are bounded by ``max_revisits`` (how many times a single node may repeat
    on one path) so enumeration always terminates. This powers coverage checks and
    seeds weighted sampling in data generation; it is not the sampler itself.

    Args:
        flowchart: A flowchart that has already passed :func:`validate`.
        max_paths: Stop after yielding this many paths.
        max_revisits: Max times any node may appear on a single path.

    Yields:
        Lists of node ids beginning at ``start`` and ending at a terminal node.

    Example:
        >>> paths = list(enumerate_paths(fc, max_paths=5))
        >>> all(fc.nodes[p[-1]].is_terminal for p in paths)
        True
    """
    yielded = 0

    def _dfs(nid: str, path: list[str], visits: dict[str, int]) -> Iterator[list[str]]:
        nonlocal yielded
        if yielded >= max_paths:
            return
        node = flowchart.nodes[nid]
        if node.is_terminal:
            yielded += 1
            yield list(path)
            return
        for edge in node.next:
            if visits.get(edge.to, 0) >= max_revisits:
                continue
            visits[edge.to] = visits.get(edge.to, 0) + 1
            path.append(edge.to)
            yield from _dfs(edge.to, path, visits)
            path.pop()
            visits[edge.to] -= 1
            if yielded >= max_paths:
                return

    yield from _dfs(flowchart.start, [flowchart.start], {flowchart.start: 1})
