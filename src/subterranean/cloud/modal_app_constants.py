"""Modal app primitives shared between :mod:`modal_app` and :mod:`_modal_serve`.

Lives in its own module so :mod:`_modal_serve` (which must avoid
``from __future__ import annotations`` for Modal's parameter introspection)
can share the same :class:`modal.App`, images, volumes, and serve-knob
constants as the rest of the cloud package. Importing this module requires
``modal``.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "subterranean"
SERVE_APP = modal.App(APP_NAME)

# ----------------------------------------------------------------------------
# Images
#
# Until ``subterranean-agents`` is published to PyPI, ship the local source tree
# into the image and pip-install it editable with the relevant extras. This is
# the standard Modal dev pattern (see ``Image.add_local_dir``) and lets the same
# Modal apps run unchanged once the package is published — just swap each image
# to ``pip_install("subterranean-agents[<extra>]")``.
# ----------------------------------------------------------------------------

#: Repo root, derived from this file's location (``src/subterranean/cloud/...``).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REMOTE_SRC = "/root/subterranean"
_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/.mypy_cache",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/build",
    "**/dist",
    "**/site",
    "**/*.egg-info",
    "**/htmlcov",
    "**/.coverage",
]


def _local_install_image(extra: str) -> modal.Image:
    """Build an image that pip-installs the local source with ``[extra]`` extras."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .add_local_dir(
            _REPO_ROOT,
            remote_path=_REMOTE_SRC,
            copy=True,
            ignore=_IGNORE,
        )
        .run_commands(
            "pip install --upgrade pip",
            f"pip install -e '{_REMOTE_SRC}[{extra}]'",
        )
    )


#: CPU image for the API-bound generate/evaluate steps (core + anthropic + matplotlib).
CPU_IMAGE = _local_install_image("report")

#: Training image: the heavy ML stack (torch/trl/deepspeed/bitsandbytes).
TRAIN_IMAGE = _local_install_image("train")

#: Serving image: vLLM (CUDA/Linux only).
SERVE_IMAGE = _local_install_image("serve")

#: Persisted build artifacts (flowchart IR, dataset.jsonl, eval reports).
BUILD_VOLUME = modal.Volume.from_name("subterranean-build", create_if_missing=True)
#: Persisted fine-tuned model weights.
MODEL_VOLUME = modal.Volume.from_name("subterranean-models", create_if_missing=True)

BUILD_ROOT = "/build"
MODEL_ROOT = "/models"
VOLUMES = {BUILD_ROOT: BUILD_VOLUME, MODEL_ROOT: MODEL_VOLUME}

#: Anthropic API key, injected into the API-bound functions.
ANTHROPIC_SECRET = modal.Secret.from_name("anthropic-secret")

# Timeouts (seconds). Generation/eval are long API-bound jobs; the 3B run is the
# paper's ~3.5h; the 8B ZeRO-3 run is fast (~15-30 min) but gets head-room.
HOUR = 60 * 60
GENERATE_TIMEOUT = 6 * HOUR
TRAIN_3B_TIMEOUT = 6 * HOUR
TRAIN_8B_TIMEOUT = 3 * HOUR
EVALUATE_TIMEOUT = 4 * HOUR

# Serve knobs.
SERVE_GPU = "A100-80GB"
SERVE_TIMEOUT = HOUR
SERVE_SCALEDOWN_WINDOW = 300
SERVE_MIN_CONTAINERS = 0
SERVE_MAX_CONTAINERS = 4
SERVE_PORT = 8000
