"""CrewAI adapter — planned for v2.

CrewAI is a multi-agent framework; v1 of subterranean targets single-agent
procedural workflows only (see CLAUDE.md "What v1 ships"). This module is a
placeholder so the import path is stable for the v2 implementation.
"""

from __future__ import annotations

from typing import Any

from subterranean.ir.schema import Flowchart


def flowchart_from_crew(crew: Any, *, name: str, description: str = "") -> Flowchart:
    """Convert a CrewAI crew into a Flowchart IR. Not implemented in v1.

    Args:
        crew: A CrewAI crew/flow object.
        name: Stable identifier for the resulting procedure.
        description: Optional human-readable one-liner.

    Raises:
        NotImplementedError: Always — the CrewAI adapter is planned for v2.
    """
    raise NotImplementedError("The CrewAI adapter is planned for v2.")
