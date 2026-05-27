"""Serve a compiled model behind an OpenAI-compatible HTTP API via vLLM.

Design
------
vLLM already ships a production-grade, OpenAI-compatible API server
(``vllm.entrypoints.openai.api_server``). Rather than re-implement request and
response shaping on top of ``AsyncLLMEngine`` + FastAPI, this module *drives that
server*: it resolves the servable checkpoint inside a build directory, builds the
exact ``argv`` the vLLM server's CLI parser expects, and hands it to vLLM's own
entrypoint. This keeps subterranean's surface tiny and inherits OpenAI-API
fidelity (``/v1/chat/completions``, ``/v1/models``, streaming) for free.

Import safety
-------------
vLLM only builds on CUDA/Linux, so it is **not** a core dependency and is **not**
installed on developer machines (incl. this macOS box). Every ``vllm`` import is
therefore performed lazily, inside the functions that need it. The module imports
cleanly with only the core dependencies present, which is what the unit tests and
``subterranean serve --help`` rely on.

The launch body (:func:`serve`) needs a GPU and a real vLLM install, so it runs
on a GPU host / Modal (Phase 7), not in the local test suite. Everything else —
model-path resolution (:func:`resolve_model_path`), argv construction
(:func:`build_vllm_server_args`), and the import guard (:func:`_require_vllm`) —
is pure Python and fully unit-tested without vLLM.
"""

from __future__ import annotations

from pathlib import Path

from subterranean.exceptions import ServingError
from subterranean.logging import logger

__all__ = [
    "BEST_SUBDIR",
    "build_vllm_server_args",
    "resolve_model_path",
    "serve",
]

BEST_SUBDIR = "best"
"""Name of the best-checkpoint subdirectory written by the training phase."""

# Relative locations (most-specific first) where a servable checkpoint may live
# under a build directory. ``subterranean train`` writes ``<build>/model/best``
# (the CLI sets the trainer ``output_dir`` to ``<build>/model``); ``<build>/best``
# is supported for builds that put the checkpoint directly under the build root.
_CANDIDATE_SUBDIRS = (
    ("model", BEST_SUBDIR),
    (BEST_SUBDIR,),
)

_INSTALL_HINT = (
    "Serving requires the optional vLLM stack, which is not installed. Install it "
    "with `pip install -e '.[serve]'` and run on a GPU host (vLLM requires "
    "CUDA/Linux and does not build on macOS or CPU-only machines)."
)

# Files that mark a directory as a Hugging Face / vLLM-loadable model checkpoint.
_MODEL_MARKERS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _looks_like_model_dir(path: Path) -> bool:
    """Return True if ``path`` is a directory containing model weights/config.

    A directory is considered servable when it holds a ``config.json`` (every HF
    checkpoint has one) or a recognised weights shard/index file.
    """
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _MODEL_MARKERS)


def resolve_model_path(build_dir: str | Path) -> Path:
    """Locate the servable model for a build directory.

    Resolution order:

    1. ``<build_dir>/model/best`` — the best checkpoint written by
       ``subterranean train`` — if it looks like a model directory.
    2. ``<build_dir>/best`` — a checkpoint placed directly under the build root.
    3. ``<build_dir>`` itself, if it directly looks like a model directory.

    Args:
        build_dir: A ``build/<name>`` directory, or a model directory itself.

    Returns:
        The resolved path to a directory containing a loadable checkpoint.

    Raises:
        ServingError: If ``build_dir`` does not exist, or no servable model can
            be found at any candidate location.

    Example:
        >>> resolve_model_path("build/travel")  # doctest: +SKIP
        PosixPath('build/travel/model/best')
    """
    root = Path(build_dir)
    if not root.exists():
        raise ServingError(
            f"Build directory not found: {root}. Run `subterranean train {root}` first."
        )

    for parts in _CANDIDATE_SUBDIRS:
        candidate = root.joinpath(*parts)
        if _looks_like_model_dir(candidate):
            return candidate
    if _looks_like_model_dir(root):
        return root

    expected = root.joinpath(*_CANDIDATE_SUBDIRS[0])
    raise ServingError(
        f"No servable model found under {root}. Expected a checkpoint at "
        f"{expected} (written by `subterranean train`) or a model directory at "
        f"{root} (containing config.json + weights). Run `subterranean train` "
        "first, or point at a model directory."
    )


def build_vllm_server_args(
    model_path: str | Path,
    *,
    port: int,
    host: str,
    served_model_name: str | None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the ``argv`` for vLLM's OpenAI-compatible API server.

    The result is the argument vector accepted by
    ``vllm.entrypoints.openai.api_server`` (the same flags as ``vllm serve``):
    the resolved ``--model`` path, the bind ``--host``/``--port``, an optional
    public ``--served-model-name``, then any caller-supplied ``extra`` flags
    appended verbatim (e.g. ``["--max-model-len", "8192"]``).

    This function is pure (no I/O, no vLLM import) so it can be unit-tested
    without a GPU or the serving stack installed.

    Args:
        model_path: Path to the loadable checkpoint directory.
        port: TCP port to bind.
        host: Interface to bind (e.g. ``"0.0.0.0"``).
        served_model_name: Public model id exposed via the API; if ``None`` the
            ``--served-model-name`` flag is omitted and vLLM defaults to the model
            path.
        extra: Additional raw vLLM CLI args, appended unchanged.

    Returns:
        The argv list, with ``--model`` first.

    Example:
        >>> build_vllm_server_args(
        ...     "build/travel/best", port=8000, host="0.0.0.0",
        ...     served_model_name="travel",
        ... )
        ['--model', 'build/travel/best', '--host', '0.0.0.0', '--port', '8000', \
'--served-model-name', 'travel']
    """
    args: list[str] = [
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if served_model_name is not None:
        args += ["--served-model-name", served_model_name]
    if extra:
        args += list(extra)
    return args


def _require_vllm() -> None:
    """Verify vLLM is importable, else raise an actionable :class:`ServingError`.

    Raises:
        ServingError: If ``vllm`` cannot be imported, with an install hint and a
            reminder that vLLM is GPU/CUDA-only.
    """
    try:
        import vllm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ServingError(_INSTALL_HINT) from exc


def serve(
    model_path: str | Path,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    served_model_name: str | None = None,
    extra: list[str] | None = None,
) -> None:
    """Launch a blocking, OpenAI-compatible vLLM server for a compiled model.

    Lazily imports vLLM (raising an actionable :class:`ServingError` if it is not
    installed), builds the server argv via :func:`build_vllm_server_args`, and
    runs vLLM's own OpenAI API server. This call blocks until the server is shut
    down. It needs a GPU and the ``[serve]`` extra, so it is exercised on a GPU
    host / Modal rather than in the local unit suite.

    Args:
        model_path: Path to a loadable checkpoint directory (use
            :func:`resolve_model_path` to derive one from a build dir).
        port: TCP port to bind. Defaults to ``8000``.
        host: Interface to bind. Defaults to ``"0.0.0.0"``.
        served_model_name: Public model id exposed via the API. Defaults to the
            model path when ``None``.
        extra: Additional raw vLLM CLI args appended to the server argv.

    Raises:
        ServingError: If vLLM is not installed on this host.

    Example:
        >>> serve("build/travel/best", port=8000)  # doctest: +SKIP
    """
    _require_vllm()

    args = build_vllm_server_args(
        model_path,
        port=port,
        host=host,
        served_model_name=served_model_name,
        extra=extra,
    )

    logger.info(
        f"Serving '{served_model_name or model_path}' on http://{host}:{port} "
        "(OpenAI-compatible: /v1/chat/completions, /v1/models)."
    )
    _run_vllm_server(args)


def _run_vllm_server(args: list[str]) -> None:  # pragma: no cover - GPU host only
    """Parse ``args`` with vLLM's CLI parser and run its OpenAI API server.

    Split out from :func:`serve` so the GPU-only vLLM imports live in one place;
    this body only executes on a GPU host with the ``[serve]`` extra installed.

    Args:
        args: Argv built by :func:`build_vllm_server_args`.
    """
    import asyncio

    # Lazy imports: vLLM is GPU/CUDA-only and absent on dev machines.
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(description="subterranean vLLM OpenAI server"))
    parsed = parser.parse_args(args)
    asyncio.run(run_server(parsed))
