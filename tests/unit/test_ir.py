"""Unit tests for the Flowchart IR: schema, loader, and validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent2model.exceptions import FlowchartValidationError
from agent2model.ir.loader import load_flowchart, load_flowchart_from_string
from agent2model.ir.schema import Edge, Flowchart, Node
from agent2model.ir.validator import enumerate_paths, validate

# A minimal valid flowchart used by several tests.
MINIMAL = """
name: tiny
start: a
nodes:
  a:
    role: agent
    prompt: say hi
    next: [done]
  done:
    terminal: success
"""


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #


def test_next_normalizes_bare_string_to_edge() -> None:
    node = Node(role="agent", next=["b"])  # type: ignore[arg-type]
    assert node.next == [Edge(to="b")]


def test_next_normalizes_scalar_string() -> None:
    node = Node.model_validate({"role": "agent", "next": "b"})
    assert node.next == [Edge(to="b")]


def test_edge_mapping_with_condition() -> None:
    node = Node.model_validate(
        {"role": "decision", "next": [{"to": "b", "when": "ready"}, {"to": "c"}]}
    )
    assert node.next == [Edge(to="b", when="ready"), Edge(to="c")]


def test_terminal_node_rejects_role() -> None:
    with pytest.raises(ValueError, match="must not declare a `role`"):
        Node.model_validate({"terminal": "success", "role": "agent"})


def test_terminal_node_rejects_next() -> None:
    with pytest.raises(ValueError, match="must not declare outgoing `next`"):
        Node.model_validate({"terminal": "success", "next": ["a"]})


def test_non_terminal_requires_role() -> None:
    with pytest.raises(ValueError, match="must declare a `role`"):
        Node.model_validate({"prompt": "hi", "next": ["a"]})


def test_unknown_field_forbidden() -> None:
    with pytest.raises(ValueError):
        Node.model_validate({"role": "agent", "next": ["a"], "bogus": 1})


def test_is_terminal_and_terminals_property() -> None:
    fc = load_flowchart_from_string(MINIMAL)
    assert fc.nodes["done"].is_terminal is True
    assert fc.nodes["a"].is_terminal is False
    assert set(fc.terminals) == {"done"}


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #


def test_load_minimal_from_string() -> None:
    fc = load_flowchart_from_string(MINIMAL)
    assert isinstance(fc, Flowchart)
    assert fc.name == "tiny"
    assert fc.start == "a"


def test_load_missing_file_raises() -> None:
    with pytest.raises(FlowchartValidationError, match="not found"):
        load_flowchart("/nonexistent/path/flowchart.yaml")


def test_load_invalid_yaml_raises() -> None:
    with pytest.raises(FlowchartValidationError, match="not valid YAML"):
        load_flowchart_from_string("name: [unbalanced\n  : :")


def test_load_non_mapping_raises() -> None:
    with pytest.raises(FlowchartValidationError, match="mapping"):
        load_flowchart_from_string("- just\n- a\n- list")


def test_load_file_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unbalanced\n  : :", encoding="utf-8")
    with pytest.raises(FlowchartValidationError, match="not valid YAML"):
        load_flowchart(bad)


def test_load_file_non_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "list.yaml"
    f.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(FlowchartValidationError, match="mapping"):
        load_flowchart(f)


def test_load_file_schema_violation_reports_location(tmp_path: Path) -> None:
    f = tmp_path / "schema.yaml"
    f.write_text(
        "name: x\nstart: a\nnodes:\n  a:\n    terminal: success\n    role: agent\n",
        encoding="utf-8",
    )
    with pytest.raises(FlowchartValidationError) as exc:
        load_flowchart(f)
    assert any("nodes.a" in e for e in exc.value.errors)


def test_load_schema_violation_reports_location() -> None:
    bad = """
    name: x
    start: a
    nodes:
      a:
        terminal: success
        role: agent
    """
    with pytest.raises(FlowchartValidationError) as exc:
        load_flowchart_from_string(bad)
    assert any("nodes.a" in e for e in exc.value.errors)


# --------------------------------------------------------------------------- #
# Validator                                                                    #
# --------------------------------------------------------------------------- #


def test_travel_flowchart_is_valid(travel_flowchart: Flowchart) -> None:
    validate(travel_flowchart)  # must not raise


def test_unknown_start_node() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: missing
        nodes:
          a:
            role: agent
            next: [done]
          done:
            terminal: success
        """)
    with pytest.raises(FlowchartValidationError, match="unknown node 'missing'"):
        validate(fc)


def test_edge_to_unknown_node() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a:
            role: agent
            next: [ghost]
          done:
            terminal: success
        """)
    with pytest.raises(FlowchartValidationError, match="unknown node 'ghost'"):
        validate(fc)


def test_non_terminal_without_edges() -> None:
    fc = Flowchart(
        name="x",
        start="a",
        nodes={
            "a": Node(role="agent", next=[]),
            "done": Node(terminal="success"),
        },
    )
    with pytest.raises(FlowchartValidationError, match="no outgoing edges"):
        validate(fc)


def test_unreachable_terminal() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a:
            role: agent
            next: [done]
          done:
            terminal: success
          orphan:
            terminal: abandonment
        """)
    with pytest.raises(FlowchartValidationError, match=r"orphan.*not reachable"):
        validate(fc)


def test_cycle_without_escape_is_trapped() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a:
            role: agent
            next: [b]
          b:
            role: agent
            next: [a]
          done:
            terminal: success
        """)
    with pytest.raises(FlowchartValidationError, match="cannot reach any terminal"):
        validate(fc)


def test_cycle_with_escape_is_valid() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a:
            role: agent
            next: [b]
          b:
            role: decision
            next:
              - to: a
                when: keep looping
              - to: done
                when: escape
          done:
            terminal: success
        """)
    validate(fc)  # must not raise


# --------------------------------------------------------------------------- #
# Path enumeration                                                             #
# --------------------------------------------------------------------------- #


def test_enumerate_paths_all_end_at_terminal(travel_flowchart: Flowchart) -> None:
    paths = list(enumerate_paths(travel_flowchart, max_paths=50))
    assert paths, "expected at least one path"
    for path in paths:
        assert path[0] == travel_flowchart.start
        assert travel_flowchart.nodes[path[-1]].is_terminal


def test_enumerate_paths_respects_max_paths(travel_flowchart: Flowchart) -> None:
    assert len(list(enumerate_paths(travel_flowchart, max_paths=3))) <= 3


def test_enumerate_paths_terminates_on_cycle() -> None:
    fc = load_flowchart_from_string("""
        name: x
        start: a
        nodes:
          a:
            role: decision
            next:
              - to: a
                when: loop
              - to: done
                when: escape
          done:
            terminal: success
        """)
    # max_revisits bounds the loop, so enumeration must finish.
    paths = list(enumerate_paths(fc, max_paths=100, max_revisits=2))
    assert all(p[-1] == "done" for p in paths)
