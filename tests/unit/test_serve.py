"""Unit tests for the vLLM serving layer.

These tests never import vllm (it is GPU/CUDA-only and not installed here). They
exercise the pure, testable surface: argv construction, model-path resolution,
and the import-guard error path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from subterranean.exceptions import ServingError
from subterranean.serve import vllm_server


def _make_model_dir(path: Path) -> Path:
    """Create a directory that looks like a HF checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# build_vllm_server_args
# --------------------------------------------------------------------------- #


def test_build_args_basic() -> None:
    args = vllm_server.build_vllm_server_args(
        "build/travel/best",
        port=8000,
        host="0.0.0.0",
        served_model_name="travel",
    )
    assert args == [
        "--model",
        "build/travel/best",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        "travel",
    ]


def test_build_args_omits_served_model_name_when_none() -> None:
    args = vllm_server.build_vllm_server_args(
        "build/travel/best", port=1234, host="127.0.0.1", served_model_name=None
    )
    assert "--served-model-name" not in args
    assert args[:2] == ["--model", "build/travel/best"]
    assert args[args.index("--port") + 1] == "1234"
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_args_appends_extra_verbatim() -> None:
    args = vllm_server.build_vllm_server_args(
        "m",
        port=8000,
        host="0.0.0.0",
        served_model_name=None,
        extra=["--max-model-len", "8192", "--dtype", "bfloat16"],
    )
    assert args[-4:] == ["--max-model-len", "8192", "--dtype", "bfloat16"]


def test_build_args_accepts_path_object(tmp_path: Path) -> None:
    args = vllm_server.build_vllm_server_args(
        tmp_path / "best", port=8000, host="0.0.0.0", served_model_name=None
    )
    assert args[1] == str(tmp_path / "best")


# --------------------------------------------------------------------------- #
# resolve_model_path
# --------------------------------------------------------------------------- #


def test_resolve_prefers_model_best_from_training(tmp_path: Path) -> None:
    # `subterranean train` writes <build>/model/best — the canonical layout.
    best = _make_model_dir(tmp_path / "model" / "best")
    assert vllm_server.resolve_model_path(tmp_path) == best


def test_resolve_prefers_model_best_over_root_best(tmp_path: Path) -> None:
    model_best = _make_model_dir(tmp_path / "model" / "best")
    _make_model_dir(tmp_path / "best")  # also present; model/best must win
    assert vllm_server.resolve_model_path(tmp_path) == model_best


def test_resolve_uses_root_best_subdir(tmp_path: Path) -> None:
    best = _make_model_dir(tmp_path / "best")
    # Build root also looks like a model dir; best/ must win over the root.
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert vllm_server.resolve_model_path(tmp_path) == best


def test_resolve_falls_back_to_build_dir(tmp_path: Path) -> None:
    _make_model_dir(tmp_path)  # no best/ subdir
    assert vllm_server.resolve_model_path(tmp_path) == tmp_path


def test_resolve_recognises_safetensors_weights(tmp_path: Path) -> None:
    best = tmp_path / "best"
    best.mkdir()
    (best / "model.safetensors").write_text("", encoding="utf-8")
    assert vllm_server.resolve_model_path(tmp_path) == best


def test_resolve_raises_when_nothing_servable(tmp_path: Path) -> None:
    (tmp_path / "dataset.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ServingError, match="No servable model found"):
        vllm_server.resolve_model_path(tmp_path)


def test_resolve_raises_when_build_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(ServingError, match="Build directory not found"):
        vllm_server.resolve_model_path(tmp_path / "does-not-exist")


def test_resolve_ignores_best_that_is_a_file(tmp_path: Path) -> None:
    # A file named "best" is not a model dir; fall back to the build root.
    (tmp_path / "best").write_text("not a dir", encoding="utf-8")
    _make_model_dir(tmp_path)
    assert vllm_server.resolve_model_path(tmp_path) == tmp_path


# --------------------------------------------------------------------------- #
# import guard
# --------------------------------------------------------------------------- #


def test_require_vllm_raises_install_hint_when_absent() -> None:
    # vllm is not installed in this environment; the guard must fire.
    with pytest.raises(ServingError) as excinfo:
        vllm_server._require_vllm()
    msg = str(excinfo.value)
    assert "pip install -e '.[serve]'" in msg
    assert "GPU host" in msg


def test_serve_raises_serving_error_without_vllm(tmp_path: Path) -> None:
    model = _make_model_dir(tmp_path / "best")
    with pytest.raises(ServingError, match=r"pip install -e '\.\[serve\]'"):
        vllm_server.serve(model, port=8000)


def test_module_imports_without_vllm() -> None:
    # Re-importing must not pull in vllm at module scope.
    mod = importlib.reload(vllm_server)
    assert hasattr(mod, "build_vllm_server_args")
