"""A minimal LangGraph ``StateGraph`` for the compile-from-LangGraph demo.

This file is the *source* the LangGraph adapter discovers and compiles:

    subterranean compile examples/langgraph_demo/demo.py --out build/demo

It defines a small order-support procedure as a LangGraph graph. The adapter
recovers the *structure* (nodes, plain edges, and the conditional ``triage``
branch) into the Flowchart IR. It cannot recover natural-language prompts — a
LangGraph node is a Python callable, not an instruction — so every agent node
gets a ``TODO`` placeholder prompt you fill in before generating data.

The graph is exposed three ways the discovery contract accepts; we use a
module-level ``graph`` variable here (the most common pattern). A
``build_graph()`` factory or ``app`` / ``workflow`` variables would work too.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SupportState(TypedDict):
    """Conversation state carried between nodes (content is irrelevant here)."""

    category: str


def _node(state: SupportState) -> SupportState:
    """Identity node body; the demo only cares about graph *structure*."""
    return state


def _route(state: SupportState) -> str:
    """Branch label for the conditional ``triage`` edge."""
    return state["category"]


def build_graph() -> StateGraph:
    """Build the order-support ``StateGraph``.

    The procedure: greet, collect the order detail, triage into one of three
    handlers, attempt a resolution, and either close out or escalate. The
    ``handle_refund``/``handle_status``/``handle_other`` handlers loop back into
    a shared resolution check, giving the adapter a cycle with a terminal-reaching
    escape edge.

    Returns:
        The assembled (uncompiled) :class:`~langgraph.graph.StateGraph`.

    Example:
        >>> g = build_graph()
        >>> "triage" in g.nodes
        True
    """
    builder: StateGraph = StateGraph(SupportState)
    builder.add_node("greet", _node)
    builder.add_node("collect_order", _node)
    builder.add_node("triage", _node)
    builder.add_node("handle_refund", _node)
    builder.add_node("handle_status", _node)
    builder.add_node("handle_other", _node)
    builder.add_node("check_resolved", _node)
    builder.add_node("close_out", _node)
    builder.add_node("escalate", _node)

    builder.add_edge(START, "greet")
    builder.add_edge("greet", "collect_order")
    builder.add_edge("collect_order", "triage")
    builder.add_conditional_edges(
        "triage",
        _route,
        {
            "refund": "handle_refund",
            "status": "handle_status",
            "other": "handle_other",
        },
    )
    builder.add_edge("handle_refund", "check_resolved")
    builder.add_edge("handle_status", "check_resolved")
    builder.add_edge("handle_other", "check_resolved")
    builder.add_conditional_edges(
        "check_resolved",
        _route,
        {
            "resolved": "close_out",
            "retry": "triage",
            "stuck": "escalate",
        },
    )
    builder.add_edge("close_out", END)
    builder.add_edge("escalate", END)
    return builder


# Module-level variable discovered by `load_stategraph_from_pyfile`.
graph: StateGraph = build_graph()
