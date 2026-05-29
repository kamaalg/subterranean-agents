"""Unit tests for the workflow adapters (Phase 3).

Builds real LangGraph ``StateGraph`` objects, converts them to the Flowchart IR,
and checks structural fidelity. No network or GPU; fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from agent2model.adapters.crewai import flowchart_from_crew
from agent2model.adapters.langgraph import (
    TODO_PROMPT,
    flowchart_from_stategraph,
    load_stategraph_from_pyfile,
)
from agent2model.exceptions import FlowchartValidationError
from agent2model.ir.validator import validate


class _State(TypedDict):
    x: int


def _passthrough(state: _State) -> _State:
    return state


def _build_branching_graph() -> StateGraph:
    """A graph: START → triage → (decision) → reply | escalate → END."""

    def route(state: _State) -> str:
        return "simple" if state["x"] else "hard"

    graph = StateGraph(_State)
    graph.add_node("triage", _passthrough)
    graph.add_node("reply", _passthrough)
    graph.add_node("escalate", _passthrough)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route, {"simple": "reply", "hard": "escalate"})
    graph.add_edge("reply", END)
    graph.add_edge("escalate", END)
    return graph


def test_conversion_produces_valid_flowchart() -> None:
    fc = flowchart_from_stategraph(_build_branching_graph(), name="support")

    validate(fc)  # must not raise
    assert fc.name == "support"
    assert fc.start == "triage"

    # Terminals: a single 'end' terminal of kind success.
    assert set(fc.terminals) == {"end"}
    assert fc.terminals["end"].terminal == "success"


def test_conditional_edge_becomes_decision_node_with_branches() -> None:
    fc = flowchart_from_stategraph(_build_branching_graph(), name="support")

    triage = fc.nodes["triage"]
    assert triage.role == "decision"
    assert triage.prompt is None
    # Two guarded branches with the path-map keys as `when` labels.
    whens = {edge.when: edge.to for edge in triage.next}
    assert whens == {"simple": "reply", "hard": "escalate"}


def test_agent_nodes_get_todo_prompt() -> None:
    fc = flowchart_from_stategraph(_build_branching_graph(), name="support")
    reply = fc.nodes["reply"]
    assert reply.role == "agent"
    assert reply.prompt == TODO_PROMPT
    # Plain edge to terminal is unconditional.
    assert [e.to for e in reply.next] == ["end"]
    assert reply.next[0].when is None


def test_accepts_compiled_graph() -> None:
    compiled = _build_branching_graph().compile()
    fc = flowchart_from_stategraph(compiled, name="support")
    validate(fc)
    assert fc.start == "triage"


def test_rejects_non_stategraph() -> None:
    with pytest.raises(FlowchartValidationError, match="StateGraph"):
        flowchart_from_stategraph(object(), name="x")  # type: ignore[arg-type]


def test_rejects_empty_graph() -> None:
    graph = StateGraph(_State)
    with pytest.raises(FlowchartValidationError, match="no nodes"):
        flowchart_from_stategraph(graph, name="empty")


def test_rejects_missing_start_edge() -> None:
    graph = StateGraph(_State)
    graph.add_node("a", _passthrough)
    graph.add_edge("a", END)
    with pytest.raises(FlowchartValidationError, match="entry edge from START"):
        flowchart_from_stategraph(graph, name="nostart")


def test_conditional_branch_to_end_becomes_terminal() -> None:
    graph = StateGraph(_State)
    graph.add_node("a", _passthrough)
    graph.add_node("b", _passthrough)
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", lambda s: "done", {"more": "b", "done": END})
    graph.add_edge("b", "a")
    fc = flowchart_from_stategraph(graph, name="cond_end")
    validate(fc)
    assert fc.nodes["a"].role == "decision"
    whens = {e.when: e.to for e in fc.nodes["a"].next}
    assert whens == {"more": "b", "done": "end"}
    assert fc.terminals["end"].terminal == "success"


def test_rejects_conditional_edge_from_start() -> None:
    graph = StateGraph(_State)
    graph.add_node("a", _passthrough)
    graph.add_node("b", _passthrough)
    graph.add_conditional_edges(START, lambda s: "a", {"x": "a", "y": "b"})
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    with pytest.raises(FlowchartValidationError, match="from START is not supported"):
        flowchart_from_stategraph(graph, name="condstart")


def test_rejects_conditional_edge_without_path_map() -> None:
    graph = StateGraph(_State)
    graph.add_node("a", _passthrough)
    graph.add_node("b", _passthrough)
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", lambda s: "b")  # no path map
    graph.add_edge("b", END)
    with pytest.raises(FlowchartValidationError, match="no path"):
        flowchart_from_stategraph(graph, name="nomap")


# --- .py loader -------------------------------------------------------------

_GRAPH_FILE_TEMPLATE = """
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    x: int


def _node(state):
    return state


def route(state):
    return "yes" if state["x"] else "no"


{binding_block}
"""


def _write_graph_file(tmp_path: Path, binding_block: str) -> Path:
    path = tmp_path / "graph.py"
    path.write_text(_GRAPH_FILE_TEMPLATE.format(binding_block=binding_block))
    return path


def test_loader_finds_module_variable(tmp_path: Path) -> None:
    block = """
graph = StateGraph(State)
graph.add_node("triage", _node)
graph.add_node("done", _node)
graph.add_edge(START, "triage")
graph.add_edge("triage", "done")
graph.add_edge("done", END)
"""
    path = _write_graph_file(tmp_path, block)
    loaded = load_stategraph_from_pyfile(path)
    fc = flowchart_from_stategraph(loaded, name=path.stem)
    validate(fc)
    assert fc.start == "triage"


def test_loader_finds_factory(tmp_path: Path) -> None:
    block = """
def build_graph():
    g = StateGraph(State)
    g.add_node("a", _node)
    g.add_node("b", _node)
    g.add_edge(START, "a")
    g.add_conditional_edges("a", route, {"yes": "b", "no": "a"})
    g.add_edge("b", END)
    return g
"""
    path = _write_graph_file(tmp_path, block)
    loaded = load_stategraph_from_pyfile(path)
    fc = flowchart_from_stategraph(loaded, name="factory")
    validate(fc)
    assert fc.nodes["a"].role == "decision"


def test_loader_unwraps_compiled_app(tmp_path: Path) -> None:
    block = """
_g = StateGraph(State)
_g.add_node("a", _node)
_g.add_edge(START, "a")
_g.add_edge("a", END)
app = _g.compile()
"""
    path = _write_graph_file(tmp_path, block)
    loaded = load_stategraph_from_pyfile(path)
    assert isinstance(loaded, StateGraph)


def test_loader_no_graph_found(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("X = 1\n")
    with pytest.raises(FlowchartValidationError, match="No LangGraph StateGraph found"):
        load_stategraph_from_pyfile(path)


def test_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FlowchartValidationError, match="No such file"):
        load_stategraph_from_pyfile(tmp_path / "nope.py")


def test_loader_import_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("import nonexistent_module_xyz\n")
    with pytest.raises(FlowchartValidationError, match="Failed to import"):
        load_stategraph_from_pyfile(path)


def test_loader_factory_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_factory.py"
    path.write_text("def build_graph():\n    raise ValueError('boom')\n")
    with pytest.raises(FlowchartValidationError, match="raised"):
        load_stategraph_from_pyfile(path)


# --- crewai stub ------------------------------------------------------------


def test_crewai_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="v2"):
        flowchart_from_crew(object(), name="x")
