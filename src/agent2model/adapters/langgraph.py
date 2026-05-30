"""LangGraph ``StateGraph`` → Flowchart IR adapter.

This adapter gives you the *structure* of a procedure for free: it maps a
LangGraph graph's nodes and edges onto the canonical :class:`~agent2model.ir.schema.Flowchart`
so you don't have to hand-write YAML. It cannot, however, recover the
*natural-language instructions* a node should follow — a LangGraph node is a
Python callable, not a prompt. Every ``agent`` node therefore gets a TODO
placeholder prompt; you fill those in before generating data.

Mapping rules
-------------
- Each graph node becomes an IR node with ``role: agent`` and a TODO ``prompt``.
- LangGraph's ``START`` (``"__start__"``) becomes the IR ``start`` node: the
  single successor of ``START`` is used as the entry node directly (no synthetic
  node is created).
- LangGraph's ``END`` (``"__end__"``) and any other sink reachable from the graph
  become IR terminal nodes with ``terminal: success`` (we cannot infer
  abandonment/escalation from structure alone, so we default to ``success``).
- Plain edges (``add_edge``) become unconditional :class:`~agent2model.ir.schema.Edge`.
- Conditional edges (``add_conditional_edges``) turn the source node into a
  ``decision`` node. Each branch becomes a guarded edge whose ``when`` label is the
  path-map key (e.g. ``add_conditional_edges("a", route, {"yes": "b"})`` yields a
  ``when: "yes"`` edge to ``b``). A conditional edge declared *without* a path map
  has statically-unknowable targets, which we cannot represent — that raises a
  :class:`~agent2model.exceptions.FlowchartValidationError` asking for a path map.

``.py`` loader contract
------------------------
:func:`load_stategraph_from_pyfile` imports a ``.py`` module and looks for the
graph in this order:

1. A zero-argument factory named ``build_graph``, ``make_graph``, or
   ``create_graph`` (called and its result used).
2. A module-level variable named ``graph``, ``app``, or ``workflow``.

The object found may be a :class:`~langgraph.graph.StateGraph` or a compiled
graph (``CompiledStateGraph``); compiled graphs are unwrapped via their
``.builder`` attribute. If none is found, a
:class:`~agent2model.exceptions.FlowchartValidationError` is raised.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

from agent2model.exceptions import FlowchartValidationError
from agent2model.ir.schema import Edge, Flowchart, Node
from agent2model.logging import logger

#: Alias for a LangGraph ``StateGraph`` in type signatures. ``langgraph`` is an
#: optional extra (``pip install 'agent2model[langgraph]'``) imported lazily
#: inside the functions below, so the rest of the package — and the whole CLI —
#: works without it installed. We keep this as ``Any`` rather than the real
#: generic type so nothing in this module forces a top-level langgraph import.
AnyStateGraph = Any


def _import_langgraph() -> tuple[Any, Any, Any]:
    """Import langgraph lazily, returning ``(END, START, StateGraph)``.

    Raises:
        FlowchartValidationError: With an install hint if the optional
            ``langgraph`` extra is not installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise FlowchartValidationError(
            "Converting a LangGraph graph requires the optional 'langgraph' extra, "
            "which is not installed. Install it with: pip install 'agent2model[langgraph]'."
        ) from exc
    return END, START, StateGraph


TODO_PROMPT = "TODO: describe what the agent should do at this step."
"""Placeholder prompt for every agent node; the user replaces it with real instructions."""

_FACTORY_NAMES = ("build_graph", "make_graph", "create_graph")
_VARIABLE_NAMES = ("graph", "app", "workflow")


def _as_stategraph(obj: Any) -> Any:
    """Return the underlying :class:`StateGraph` for ``obj``, or ``None``.

    Accepts a ``StateGraph`` directly or a compiled graph (which exposes the
    builder as ``.builder``).
    """
    _, _, StateGraph = _import_langgraph()
    if isinstance(obj, StateGraph):
        return obj
    builder = getattr(obj, "builder", None)
    if isinstance(builder, StateGraph):
        return builder
    return None


def _terminal_id(raw: str, end_sentinel: Any) -> str:
    """Map LangGraph's ``END`` sentinel to a friendly IR terminal id."""
    return "end" if raw == end_sentinel else raw


def flowchart_from_stategraph(
    graph: AnyStateGraph,
    *,
    name: str,
    description: str = "",
) -> Flowchart:
    """Convert a LangGraph ``StateGraph`` into a :class:`Flowchart`.

    The conversion is purely structural: nodes, plain edges, and conditional
    edges are mapped onto IR nodes/edges. Agent nodes receive a TODO placeholder
    prompt (LangGraph carries no natural-language prompt), terminals default to
    ``success``, and conditional edges become ``decision`` nodes whose branches'
    ``when`` labels come from the path-map keys.

    Args:
        graph: The LangGraph graph (or compiled graph) to convert.
        name: Stable identifier for the resulting procedure.
        description: Optional human-readable one-liner.

    Returns:
        A :class:`Flowchart`. It is well-formed but not graph-validated here; run
        :func:`agent2model.ir.validator.validate` on it.

    Raises:
        FlowchartValidationError: If ``graph`` is not a ``StateGraph``/compiled
            graph, has no nodes, has no entry edge from ``START``, or declares a
            conditional edge without a path map.

    Example:
        >>> fc = flowchart_from_stategraph(graph, name="support_bot")
        >>> fc.start
        'triage'
    """
    END, START, _ = _import_langgraph()
    builder = _as_stategraph(graph)
    if builder is None:
        raise FlowchartValidationError(
            "Expected a LangGraph StateGraph (or compiled graph), got " f"{type(graph).__name__!r}."
        )
    if not builder.nodes:
        raise FlowchartValidationError("LangGraph graph has no nodes to convert.")

    # Collect outgoing edges per source node. Conditional sources become decisions.
    outgoing: dict[str, list[Edge]] = {nid: [] for nid in builder.nodes}
    decision_sources: set[str] = set()
    terminal_ids: set[str] = set()

    # Plain edges (set of (src, dst) tuples). START/END are sentinels, not real nodes.
    start_node: str | None = None
    plain_edge_sources: set[str] = set()
    for src, dst in builder.edges:
        if src == START:
            start_node = dst
            continue
        plain_edge_sources.add(src)
        if dst == END:
            terminal_ids.add(_terminal_id(dst, END))
            outgoing.setdefault(src, []).append(Edge(to=_terminal_id(dst, END)))
            continue
        outgoing.setdefault(src, []).append(Edge(to=dst))

    # A node with BOTH a plain edge and conditional edges would become a decision
    # node carrying an unconditional (when=None) branch alongside guarded ones —
    # a contradiction the IR can't represent. LangGraph allows it; we reject it
    # loudly rather than silently corrupt the procedure's routing.
    mixed = plain_edge_sources & set(builder.branches)
    if mixed:
        raise FlowchartValidationError(
            f"Node(s) {', '.join(sorted(mixed))} have both an unconditional edge and "
            "conditional edges. A node must be either deterministic (one `add_edge`) or "
            "a decision (`add_conditional_edges`), not both — split the logic into "
            "separate nodes."
        )

    # Conditional edges → decision branches keyed by path-map label.
    for src, branch_map in builder.branches.items():
        if src == START:
            # A conditional entry point: we cannot pick a single start node, so we
            # surface the unresolved case rather than guessing.
            raise FlowchartValidationError(
                "Conditional edge from START is not supported; add a deterministic "
                "entry node with `add_edge(START, <node>)`."
            )
        decision_sources.add(src)
        for branch_name, branch in branch_map.items():
            ends = branch.ends
            if not ends:
                raise FlowchartValidationError(
                    f"Conditional edge '{branch_name}' from node '{src}' has no path "
                    "map, so its branch targets cannot be determined statically. Pass "
                    "a path map, e.g. add_conditional_edges('"
                    f"{src}', router, {{'label': 'target_node'}})."
                )
            for label, target in ends.items():
                resolved = _terminal_id(target, END) if target == END else target
                if target == END:
                    terminal_ids.add(resolved)
                outgoing.setdefault(src, []).append(Edge(to=resolved, when=label))

    if start_node is None:
        raise FlowchartValidationError(
            "LangGraph graph has no entry edge from START; add `add_edge(START, <node>)`."
        )

    # Assemble IR nodes.
    nodes: dict[str, Node] = {}
    for nid in builder.nodes:
        role = "decision" if nid in decision_sources else "agent"
        prompt = None if role == "decision" else TODO_PROMPT
        nodes[nid] = Node(role=role, prompt=prompt, next=outgoing.get(nid, []))
    for tid in terminal_ids:
        nodes[tid] = Node(terminal="success")

    logger.debug(
        f"Converted StateGraph '{name}': {len(builder.nodes)} graph nodes, "
        f"{len(decision_sources)} decision nodes, {len(terminal_ids)} terminals."
    )

    # Structural conversion cannot recover prompts or user turns. Warn loudly so
    # the user edits the IR before spending money generating data: otherwise every
    # agent node carries a TODO placeholder (which would be sent verbatim to the
    # generator) and there are no `role: user` nodes (so generated conversations
    # would be agent monologue, not dialogue).
    n_todo = sum(1 for node in nodes.values() if node.prompt == TODO_PROMPT)
    if n_todo:
        logger.warning(
            f"{n_todo} agent node(s) have placeholder TODO prompts. Edit the compiled "
            "flowchart and replace every 'TODO:' prompt with real instructions before "
            "running `agent2model generate`."
        )
    logger.warning(
        "LangGraph graphs carry no user turns, so the converted flowchart has no "
        "`role: user` nodes. Add user nodes where the customer speaks, or generated "
        "conversations will be agent-only monologue. See docs/adapters.md."
    )

    # All sinks default to `terminal: success` — structure can't reveal intent,
    # and distinct logical endings (book / escalate / abandon) all collapse onto
    # the single END terminal. Scan both terminal ids AND the names of nodes that
    # lead to a terminal for escalation/abandonment cues, since the telltale name
    # is usually on the predecessor (e.g. an `escalate` node → END), and warn so
    # the user splits/retypes them (eval failure-rate depends on terminal kind).
    _SINK_CUES = ("escalat", "abandon", "fail", "reject", "deflect", "giveup", "give_up")
    terminal_predecessors = {
        nid for nid, edges in outgoing.items() if any(e.to in terminal_ids for e in edges)
    }
    suspect = sorted(
        {
            name
            for name in (terminal_ids | terminal_predecessors)
            if any(cue in name.lower() for cue in _SINK_CUES)
        }
    )
    if suspect:
        logger.warning(
            f"Node(s) {', '.join(suspect)} lead to a terminal typed `terminal: success` "
            "(LangGraph END carries no kind, and all sinks collapse onto one terminal). "
            "If these are escalation/abandonment outcomes, split them into distinct "
            "terminals with the right `terminal:` kind so evaluation failure rates are correct."
        )

    return Flowchart(name=name, description=description, start=start_node, nodes=nodes)


def load_stategraph_from_pyfile(path: Path) -> AnyStateGraph:
    """Import a ``.py`` file and return the ``StateGraph`` it defines.

    The module is imported under a unique name. The graph is located using the
    contract documented at the module level: a ``build_graph`` / ``make_graph`` /
    ``create_graph`` factory first, then a ``graph`` / ``app`` / ``workflow``
    module-level variable. Compiled graphs are unwrapped to their builder.

    Args:
        path: Path to a Python file defining a LangGraph graph.

    Returns:
        The discovered :class:`StateGraph`.

    Raises:
        FlowchartValidationError: If the file cannot be imported or no graph is
            found under the supported names.

    Example:
        >>> graph = load_stategraph_from_pyfile(Path("examples/langgraph_demo/graph.py"))
    """
    path = path.resolve()
    if not path.exists():
        raise FlowchartValidationError(f"No such file: {path}")

    # SECURITY: importing the module executes arbitrary Python from ``path``.
    # Only run LangGraph files you trust. Surfaced as a warning so it is never
    # silent. A unique module name avoids clobbering anything in ``sys.modules``.
    logger.warning(
        f"Importing and executing Python from {path} to load its LangGraph graph. "
        "Only run files you trust."
    )
    module_name = f"_agent2model_lg_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise FlowchartValidationError(f"Could not load Python module from {path}.")

    module = importlib.util.module_from_spec(spec)
    # The module MUST stay in ``sys.modules`` for the whole of discovery, not just
    # the exec: a factory like ``build_graph()`` that builds a StateGraph whose
    # state schema uses ``Annotated[...]`` under ``from __future__ import
    # annotations`` resolves those string annotations via the module's globals,
    # which ``typing`` looks up through ``sys.modules``. Popping it before calling
    # the factory caused a spurious ``NameError: Annotated is not defined``. We pop
    # only in ``finally``, after a graph is found or discovery gives up.
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise FlowchartValidationError(
                f"Failed to import LangGraph file {path}: {exc}"
            ) from exc

        # 1) Zero-arg factory functions.
        for factory_name in _FACTORY_NAMES:
            factory = getattr(module, factory_name, None)
            if callable(factory):
                try:
                    produced = factory()
                except Exception as exc:
                    raise FlowchartValidationError(
                        f"Calling {factory_name}() in {path} raised: {exc}"
                    ) from exc
                graph = _as_stategraph(produced)
                if graph is not None:
                    logger.debug(f"Found graph via {factory_name}() in {path}.")
                    return graph

        # 2) Module-level variables.
        for var_name in _VARIABLE_NAMES:
            candidate = getattr(module, var_name, None)
            if candidate is not None:
                graph = _as_stategraph(candidate)
                if graph is not None:
                    logger.debug(f"Found graph via module variable '{var_name}' in {path}.")
                    return graph
    finally:
        sys.modules.pop(module_name, None)

    raise FlowchartValidationError(
        f"No LangGraph StateGraph found in {path}. Expose it as a module-level "
        f"variable named one of {_VARIABLE_NAMES}, or a zero-argument factory named "
        f"one of {_FACTORY_NAMES}."
    )


def flowchart_to_yaml_text(flowchart: Flowchart) -> str:
    """Serialise a :class:`Flowchart` IR object to YAML text.

    Uses Pydantic's ``model_dump(mode="json", exclude_none=True)`` so the dump
    round-trips cleanly through :meth:`Flowchart.model_validate`. The output is
    the same shape the YAML loader accepts (``name`` / ``start`` / ``nodes`` /
    optional ``scenario_variables``), without dropping fields that Pydantic
    treats as defaults.

    Args:
        flowchart: The IR object to dump.

    Returns:
        A YAML string suitable for embedding in a :class:`Recipe.flowchart_yaml`
        or persisting to ``flowchart.yaml`` on a build volume.

    Example:
        >>> text = flowchart_to_yaml_text(fc)
        >>> Flowchart.model_validate(yaml.safe_load(text)).name == fc.name
        True
    """
    payload = flowchart.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def langgraph_to_yaml_text(path: Path, *, name: str | None = None) -> str:
    """Load a LangGraph ``.py`` file and dump the converted Flowchart to YAML.

    Convenience composition of :func:`load_stategraph_from_pyfile`,
    :func:`flowchart_from_stategraph`, and :func:`flowchart_to_yaml_text`. Used
    by the generic Modal ``run`` entrypoint and the ``agent2model cloud run``
    CLI so a user can point at either a ``.py`` LangGraph file or a YAML file.

    Args:
        path: Path to a ``.py`` file defining a LangGraph graph.
        name: Optional explicit name for the resulting flowchart. Defaults to
            ``path.stem``.

    Returns:
        YAML text representing the converted flowchart.

    Raises:
        FlowchartValidationError: If the file cannot be loaded or converted.
    """
    graph = load_stategraph_from_pyfile(path)
    flowchart = flowchart_from_stategraph(graph, name=name or path.stem)
    return flowchart_to_yaml_text(flowchart)
