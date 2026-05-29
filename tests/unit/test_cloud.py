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
    DEFAULT_BASE_FOR_SIZE,
    EXAMPLES,
    GPU_3B,
    GPU_8B,
    ExampleRecipe,
    Recipe,
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
    # The core and cloud packages must import even when modal is absent. Here we
    # only assert the import contract holds (modal may or may not be installed
    # depending on the host); the modal-free guarantee is independently covered
    # by ``test_cloud_init_does_not_import_modal_app`` and
    # ``test_recipes_module_has_no_modal_dependency``.
    assert importlib.import_module("subterranean") is not None
    assert importlib.import_module("subterranean.cloud") is not None


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
    # 3B: A100-40GB — A10G's 22 GB is too tight for Qwen 2.5 3B full FT bf16.
    assert gpu_for_size("3b") == GPU_3B == "A100-40GB"
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
    for ep in ("reproduce_travel", "reproduce_zoom", "reproduce_insurance", "run"):
        assert hasattr(modal_app, ep)


def test_modal_app_exposes_build_recipe_helper() -> None:
    pytest.importorskip("modal")
    from subterranean.cloud import modal_app

    assert callable(modal_app.build_recipe_from_path)


# --------------------------------------------------------------------------- #
# New Recipe shape (carries flowchart_yaml inline)                              #
# --------------------------------------------------------------------------- #


_MINIMAL_YAML = (
    "name: tiny\nstart: greet\nnodes:\n"
    "  greet:\n    role: agent\n    prompt: hi\n    next: [end]\n"
    "  end:\n    terminal: success\n"
)


def test_recipe_class_replaces_example_recipe_alias() -> None:
    # Backward-compat alias must point at the same class.
    assert ExampleRecipe is Recipe


def test_recipe_requires_flowchart_yaml() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Recipe(  # type: ignore[call-arg]
            name="x",
            size="3b",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            n_convos=10,
            epochs=1,
        )


def test_recipe_round_trips_through_model_dump() -> None:
    r = Recipe(
        name="x",
        flowchart_yaml=_MINIMAL_YAML,
        size="3b",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        n_convos=10,
        epochs=1,
    )
    blob = r.model_dump()
    again = Recipe.model_validate(blob)
    assert again == r
    assert again.flowchart_yaml == _MINIMAL_YAML


def test_default_base_for_size_table() -> None:
    assert DEFAULT_BASE_FOR_SIZE["3b"] == "Qwen/Qwen2.5-3B-Instruct"
    assert DEFAULT_BASE_FOR_SIZE["8b"] == "Qwen/Qwen3-8B"


# --------------------------------------------------------------------------- #
# Paper EXAMPLES carry their YAML inline and still build the right config      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("example_id", ["travel", "zoom", "insurance"])
def test_example_recipe_has_inline_flowchart_yaml(example_id: str) -> None:
    recipe = get_recipe(example_id)
    assert isinstance(recipe.flowchart_yaml, str)
    assert recipe.flowchart_yaml.strip()
    import yaml as _yaml

    parsed = _yaml.safe_load(recipe.flowchart_yaml)
    assert isinstance(parsed, dict)
    assert "name" in parsed and "start" in parsed and "nodes" in parsed


# --------------------------------------------------------------------------- #
# LangGraph → YAML helper round-trip                                            #
# --------------------------------------------------------------------------- #


def test_flowchart_to_yaml_text_round_trips() -> None:
    pytest.importorskip("langgraph")
    import yaml as _yaml

    from subterranean.adapters.langgraph import flowchart_to_yaml_text
    from subterranean.ir.schema import Flowchart

    source = Flowchart.model_validate(_yaml.safe_load(_MINIMAL_YAML))
    text = flowchart_to_yaml_text(source)
    again = Flowchart.model_validate(_yaml.safe_load(text))
    assert again.name == source.name
    assert again.start == source.start
    assert set(again.nodes) == set(source.nodes)


def test_langgraph_to_yaml_text_produces_validatable_flowchart(tmp_path: Path) -> None:
    pytest.importorskip("langgraph")
    import yaml as _yaml

    from subterranean.adapters.langgraph import langgraph_to_yaml_text
    from subterranean.ir.schema import Flowchart
    from subterranean.ir.validator import validate

    src = tmp_path / "g.py"
    src.write_text(
        "from typing import TypedDict\n"
        "from langgraph.graph import StateGraph, START, END\n"
        "class S(TypedDict):\n    x: int\n"
        "def _node(s: S) -> S:\n    return s\n"
        "def build_graph():\n"
        "    g = StateGraph(S)\n"
        "    g.add_node('greet', _node)\n"
        "    g.add_edge(START, 'greet')\n"
        "    g.add_edge('greet', END)\n"
        "    return g\n",
        encoding="utf-8",
    )

    text = langgraph_to_yaml_text(src)
    fc = Flowchart.model_validate(_yaml.safe_load(text))
    validate(fc)
    assert fc.start == "greet"


# --------------------------------------------------------------------------- #
# build_recipe_from_path: the pure helper the `run` entrypoint dispatches to   #
# --------------------------------------------------------------------------- #


def test_build_recipe_from_path_reads_yaml(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from subterranean.cloud.modal_app import build_recipe_from_path

    yaml_path = tmp_path / "wf.yaml"
    yaml_path.write_text(_MINIMAL_YAML, encoding="utf-8")
    recipe = build_recipe_from_path(yaml_path, size="3b", n=42, epochs=3)
    # Class identity check is brittle because earlier tests reload the
    # _recipes module; compare by qualified name instead.
    assert type(recipe).__qualname__ == "Recipe"
    assert recipe.name == "tiny"  # picked up from YAML's `name` field
    assert recipe.size == "3b"
    assert recipe.n_convos == 42
    assert recipe.epochs == 3
    assert recipe.base_model == DEFAULT_BASE_FOR_SIZE["3b"]
    assert recipe.flowchart_yaml == _MINIMAL_YAML


def test_build_recipe_from_path_name_override(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from subterranean.cloud.modal_app import build_recipe_from_path

    yaml_path = tmp_path / "wf.yaml"
    yaml_path.write_text(_MINIMAL_YAML, encoding="utf-8")
    recipe = build_recipe_from_path(yaml_path, name="custom", size="8b")
    assert recipe.name == "custom"
    assert recipe.size == "8b"
    assert recipe.base_model == DEFAULT_BASE_FOR_SIZE["8b"]


def test_build_recipe_from_path_rejects_unknown_suffix(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from subterranean.cloud.modal_app import build_recipe_from_path

    bogus = tmp_path / "wf.txt"
    bogus.write_text("oops", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported flowchart suffix"):
        build_recipe_from_path(bogus)


def test_build_recipe_from_path_rejects_unknown_size(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from subterranean.cloud.modal_app import build_recipe_from_path

    yaml_path = tmp_path / "wf.yaml"
    yaml_path.write_text(_MINIMAL_YAML, encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported size"):
        build_recipe_from_path(yaml_path, size="70b")


def test_build_recipe_from_path_handles_langgraph_pyfile(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    pytest.importorskip("langgraph")
    from subterranean.cloud.modal_app import build_recipe_from_path

    src = tmp_path / "lg.py"
    src.write_text(
        "from typing import TypedDict\n"
        "from langgraph.graph import StateGraph, START, END\n"
        "class S(TypedDict):\n    x: int\n"
        "def _n(s: S) -> S:\n    return s\n"
        "def build_graph():\n"
        "    g = StateGraph(S)\n"
        "    g.add_node('greet', _n)\n"
        "    g.add_edge(START, 'greet')\n"
        "    g.add_edge('greet', END)\n"
        "    return g\n",
        encoding="utf-8",
    )
    recipe = build_recipe_from_path(src, size="3b")
    assert recipe.size == "3b"
    # The flowchart YAML must come back as valid YAML embedded in the recipe.
    import yaml as _yaml

    parsed = _yaml.safe_load(recipe.flowchart_yaml)
    assert isinstance(parsed, dict)
    assert parsed["start"] == "greet"
