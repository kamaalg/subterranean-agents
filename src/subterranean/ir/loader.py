"""Load a Flowchart IR from YAML.

Structural errors (unknown fields, bad types) surface as
:class:`~subterranean.exceptions.FlowchartValidationError` with a readable message;
graph-level invariants are checked separately by :func:`subterranean.ir.validator.validate`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from subterranean.exceptions import FlowchartValidationError
from subterranean.ir.schema import Flowchart


def load_flowchart(path: str | Path) -> Flowchart:
    """Parse and structurally validate a flowchart YAML file.

    Args:
        path: Path to the ``.yaml`` flowchart spec.

    Returns:
        The parsed :class:`~subterranean.ir.schema.Flowchart`.

    Raises:
        FlowchartValidationError: If the file is missing, not valid YAML, not a
            mapping, or violates the IR schema.

    Example:
        >>> fc = load_flowchart("examples/travel_booking/flowchart.yaml")
        >>> fc.start
        'greet'
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FlowchartValidationError(f"Flowchart file not found: {p}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise FlowchartValidationError(f"{p} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise FlowchartValidationError(
            f"{p} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )

    try:
        return Flowchart.model_validate(data)
    except ValidationError as exc:
        errors = [_format_pydantic_error(e) for e in exc.errors()]
        raise FlowchartValidationError(
            f"{p} does not match the flowchart schema:\n  - " + "\n  - ".join(errors),
            errors=errors,
        ) from exc


def load_flowchart_from_string(text: str) -> Flowchart:
    """Parse a flowchart from an in-memory YAML string (used in tests).

    Args:
        text: YAML document.

    Returns:
        The parsed :class:`~subterranean.ir.schema.Flowchart`.

    Raises:
        FlowchartValidationError: On invalid YAML or schema violations.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FlowchartValidationError(f"Flowchart is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise FlowchartValidationError("Flowchart YAML must be a mapping at the top level")
    try:
        return Flowchart.model_validate(data)
    except ValidationError as exc:
        errors = [_format_pydantic_error(e) for e in exc.errors()]
        raise FlowchartValidationError(
            "Flowchart does not match the schema:\n  - " + "\n  - ".join(errors),
            errors=errors,
        ) from exc


def _format_pydantic_error(error: ErrorDetails) -> str:
    """Render one pydantic error as ``location: message``."""
    loc = ".".join(str(part) for part in error["loc"])
    msg = error["msg"]
    return f"{loc}: {msg}" if loc else str(msg)
