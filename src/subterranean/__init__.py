"""subterranean — compile agentic workflows into LLM weights.

Public surface for the IR is re-exported here for convenience; the CLI in
:mod:`subterranean.cli` is the primary user entry point.
"""

from __future__ import annotations

from subterranean.ir.loader import load_flowchart, load_flowchart_from_string
from subterranean.ir.schema import Edge, Flowchart, Node
from subterranean.ir.validator import enumerate_paths, validate

__version__ = "0.1.0.dev0"

__all__ = [
    "Edge",
    "Flowchart",
    "Node",
    "__version__",
    "enumerate_paths",
    "load_flowchart",
    "load_flowchart_from_string",
    "validate",
]
