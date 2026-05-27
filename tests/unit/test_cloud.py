"""Unit tests for the cloud recipes (Phase 7).

These tests are network-/cloud-/GPU-free. They exercise the pure recipe helpers
in :mod:`subterranean.cloud._recipes` (image/GPU/config selection, step
sequencing, that the reproduce configs match the paper's per-example sizes and
epochs), validate the RunPod JSON specs, and confirm the core + ``cloud`` package
import **without** modal installed. ``modal_app`` itself is only touched behind a
``pytest.importorskip("modal")`` guard, so it is skipped cleanly here.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from subterranean.cloud import _recipes
from subterranean.cloud._recipes import (
    EXAMPLES,
    GPU_3B,
    GPU_8B,
    build_training_config,
    get_recipe,
    gpu_for_size,
    image_kind_for_step,
    n_gpus_for_size,
    pipeline_steps,
    training_function_for_size,
)

RUNPOD_DIR = Path(_recipes.__file__).parent / "runpod"


# --------------------------------------------------------------------------- #
# Core + cloud import without modal                                            #
# --------------------------------------------------------------------------- #


def test_core_imports_without_modal() -> None:
    # modal is not installed here; these must import regardless.
    assert importlib.import_module("subterranean") is not None
    assert importlib.import_module("subterranean.cloud") is not None
    assert importlib.util.find_spec("modal") is None


def test_cloud_init_does_not_import_modal_app() -> None:
    import subterranean.cloud as cloud_pkg

    assert not hasattr(cloud_pkg, "modal_app")


def test_recipes_module_has_no_modal_dependency() -> None:
    # Reloading the pure recipe module must not require modal.
    mod = importlib.reload(_recipes)
    assert hasattr(mod, "get_recipe")


# --------------------------------------------------------------------------- #
# Example recipes match the paper                                              #
# --------------------------------------------------------------------------- #


def test_known_examples() -> None:
    assert set(EXAMPLES) == {"travel", "zoom", "insurance"}


def test_get_recipe_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown example"):
        get_recipe("nope")


def test_travel_recipe_matches_paper() -> None:
    r = get_recipe("travel")
    assert r.size == "3b"
    assert r.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert r.n_convos == 2000
    assert r.epochs == 20


def test_zoom_recipe_matches_paper() -> None:
    r = get_recipe("zoom")
    assert r.size == "8b"
    assert r.base_model == "Qwen/Qwen3-8B"
    assert r.n_convos == 6000
    assert r.epochs == 10


def test_insurance_recipe_matches_paper() -> None:
    r = get_recipe("insurance")
    assert r.size == "8b"
    assert r.base_model == "Qwen/Qwen3-8B"
    assert r.n_convos == 3000
    assert r.epochs == 20


def test_recipe_is_frozen() -> None:
    from pydantic import ValidationError

    r = get_recipe("travel")
    with pytest.raises(ValidationError):
        r.epochs = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# GPU / size selection                                                         #
# --------------------------------------------------------------------------- #


def test_gpu_for_size() -> None:
    assert gpu_for_size("3b") == GPU_3B == "A10G"
    assert gpu_for_size("8b") == GPU_8B == "A100-80GB:8"


def test_gpu_for_size_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported model size"):
        gpu_for_size("70b")  # type: ignore[arg-type]


def test_n_gpus_for_size() -> None:
    assert n_gpus_for_size("3b") == 1
    assert n_gpus_for_size("8b") == 8


def test_training_function_for_size() -> None:
    assert training_function_for_size("3b") == "train_3b"
    assert training_function_for_size("8b") == "train_8b"


# --------------------------------------------------------------------------- #
# Training config construction                                                 #
# --------------------------------------------------------------------------- #


def test_build_training_config_3b() -> None:
    cfg = build_training_config(get_recipe("travel"), "/models/travel")
    assert cfg.size == "3b"
    assert cfg.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.epochs == 20
    assert cfg.num_gpus == 1
    assert cfg.effective_batch_size == 16
    assert cfg.output_dir == "/models/travel"


def test_build_training_config_8b_zero3() -> None:
    cfg = build_training_config(get_recipe("zoom"), "/models/zoom")
    assert cfg.size == "8b"
    assert cfg.base_model == "Qwen/Qwen3-8B"
    assert cfg.epochs == 10
    assert cfg.num_gpus == 8
    assert cfg.effective_batch_size == 32


def test_build_training_config_insurance_8b() -> None:
    cfg = build_training_config(get_recipe("insurance"), "/models/insurance")
    assert cfg.size == "8b"
    assert cfg.num_gpus == 8
    assert cfg.epochs == 20


# --------------------------------------------------------------------------- #
# Pipeline step sequencing + image selection                                  #
# --------------------------------------------------------------------------- #


def test_pipeline_steps_order() -> None:
    assert pipeline_steps() == ("generate", "train", "evaluate")


def test_image_kind_per_step() -> None:
    assert image_kind_for_step("generate") == "cpu"
    assert image_kind_for_step("train") == "train"
    assert image_kind_for_step("evaluate") == "cpu"


def test_image_kind_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline step"):
        image_kind_for_step("deploy")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# RunPod specs                                                                 #
# --------------------------------------------------------------------------- #


def test_runpod_specs_exist() -> None:
    names = {p.name for p in RUNPOD_DIR.glob("*.json")}
    assert names == {"train_3b.json", "train_8b.json", "serve.json"}


@pytest.mark.parametrize("spec_name", ["train_3b.json", "train_8b.json", "serve.json"])
def test_runpod_spec_is_valid_json_with_required_keys(spec_name: str) -> None:
    data = json.loads((RUNPOD_DIR / spec_name).read_text(encoding="utf-8"))
    for key in ("name", "imageName", "gpuTypeId", "gpuCount", "ports", "env", "dockerArgs"):
        assert key in data, f"{spec_name} missing required key {key!r}"
    assert isinstance(data["env"], dict)
    assert isinstance(data["gpuCount"], int)


def test_runpod_8b_spec_requests_eight_gpus() -> None:
    data = json.loads((RUNPOD_DIR / "train_8b.json").read_text(encoding="utf-8"))
    assert data["gpuCount"] == 8


def test_runpod_serve_spec_exposes_http_port() -> None:
    data = json.loads((RUNPOD_DIR / "serve.json").read_text(encoding="utf-8"))
    assert "8000" in data["ports"]


def test_runpod_setup_script_present_and_covers_all_stages() -> None:
    setup = RUNPOD_DIR / "setup.sh"
    assert setup.exists()
    text = setup.read_text(encoding="utf-8")
    for stage in ("generate", "train", "evaluate", "serve"):
        assert stage in text


# --------------------------------------------------------------------------- #
# modal_app: skipped here, runs where modal is installed                       #
# --------------------------------------------------------------------------- #


def test_modal_app_importable_where_modal_present() -> None:
    pytest.importorskip("modal")
    from subterranean.cloud import modal_app

    assert modal_app.APP_NAME == "subterranean"
    for fn in ("generate_data", "train_3b", "train_8b", "evaluate", "serve"):
        assert hasattr(modal_app, fn)
    for ep in ("reproduce_travel", "reproduce_zoom", "reproduce_insurance"):
        assert hasattr(modal_app, ep)
