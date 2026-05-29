"""Flowchart IR — the canonical procedure representation."""

from __future__ import annotations

from agent2model.ir.loader import load_flowchart, load_flowchart_from_string
from agent2model.ir.schema import Edge, Flowchart, Node, NodeRole, TerminalKind
from agent2model.ir.validator import enumerate_paths, validate

__all__ = [
    "Edge",
    "Flowchart",
    "Node",
    "NodeRole",
    "TerminalKind",
    "enumerate_paths",
    "load_flowchart",
    "load_flowchart_from_string",
    "validate",
]
