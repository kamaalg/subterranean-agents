"""Shared pytest fixtures.

The travel-booking flowchart is the canonical shared fixture used across tiers.
It is sourced from ``examples/travel_booking/flowchart.yaml`` so the example and
the tests can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent2model.ir.loader import load_flowchart
from agent2model.ir.schema import Flowchart

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAVEL_YAML = REPO_ROOT / "examples" / "travel_booking" / "flowchart.yaml"


@pytest.fixture
def travel_yaml_path() -> Path:
    """Filesystem path to the travel-booking flowchart YAML."""
    return TRAVEL_YAML


@pytest.fixture
def travel_flowchart() -> Flowchart:
    """The parsed (but not yet graph-validated) travel-booking flowchart."""
    return load_flowchart(TRAVEL_YAML)
