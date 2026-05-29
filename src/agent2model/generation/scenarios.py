"""Scenario-variable sampling from a flowchart's ``scenario_variables`` pools.

Each generated conversation is grounded in a sampled *scenario*: a concrete
destination, budget, user personality, and so on, drawn from the pools declared
in the flowchart YAML. These values are interpolated into node prompts and given
to the user simulator (which otherwise has no flowchart knowledge).

Pool shapes, by convention:

- A ``[lo, hi]`` pair of two numbers is treated as an inclusive numeric *range*
  and sampled uniformly. Two ints sample an int; any float bound samples a float.
- Any other list (``[Japan, Italy, ...]``, ``[decisive, skeptical, ...]``) is a
  *categorical* pool and one element is chosen.
- A scalar is passed through unchanged (a fixed value for every scenario).

Sampling is seedable via the supplied :class:`random.Random`, so a run is
reproducible end to end.
"""

from __future__ import annotations

import random
from typing import Any

from agent2model.ir.schema import Flowchart

Scenario = dict[str, Any]
"""A single sampled scenario: variable name -> concrete value."""


def _is_numeric_range(pool: Any) -> bool:
    """Whether ``pool`` is a ``[lo, hi]`` pair of two numbers (not bools)."""
    return (
        isinstance(pool, list)
        and len(pool) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in pool)
    )


def _sample_pool(name: str, pool: Any, rng: random.Random) -> Any:
    """Sample one value from a single pool according to its shape.

    Args:
        name: Variable name (used only for error messages).
        pool: The pool definition (range pair, categorical list, or scalar).
        rng: Seeded random source.

    Returns:
        The sampled value.

    Raises:
        ValueError: If ``pool`` is an empty list (nothing to sample).
    """
    if _is_numeric_range(pool):
        lo, hi = pool
        if lo > hi:
            lo, hi = hi, lo
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return rng.uniform(float(lo), float(hi))
    if isinstance(pool, list):
        if not pool:
            raise ValueError(f"scenario variable '{name}' is an empty pool")
        return rng.choice(pool)
    # Scalar (or mapping/other): a fixed value used verbatim for every scenario.
    return pool


def sample_scenario(flowchart: Flowchart, rng: random.Random) -> Scenario:
    """Sample one concrete scenario from the flowchart's variable pools.

    Args:
        flowchart: The flowchart whose ``scenario_variables`` define the pools.
        rng: Seeded random source. The same seed yields the same scenario.

    Returns:
        A mapping from each declared variable name to a sampled value. Empty when
        the flowchart declares no ``scenario_variables``.

    Raises:
        ValueError: If any categorical pool is an empty list.

    Example:
        >>> import random
        >>> s = sample_scenario(fc, random.Random(0))
        >>> s["destination_pool"] in fc.scenario_variables["destination_pool"]
        True
    """
    return {
        name: _sample_pool(name, pool, rng) for name, pool in flowchart.scenario_variables.items()
    }
