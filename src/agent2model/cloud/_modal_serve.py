"""Modal class for the autoscaling vLLM serve endpoint.

Lives in its own module because :func:`modal.parameter` requires Modal to
introspect real type objects on the class annotations, which means this file
**cannot** carry ``from __future__ import annotations``. The rest of
:mod:`agent2model.cloud.modal_app` does carry it (for the worker-function
signatures), so we isolate the class here and re-export it.

Modal also requires :func:`modal.App.cls`-decorated classes to be defined at
module top level (they cannot be created inside a factory function), which is
why :data:`ServeCls` is constructed declaratively here using shared images and
volumes imported from :mod:`agent2model.cloud.modal_app`.
"""

import modal

from agent2model.cloud.modal_app_constants import (
    MODEL_ROOT,
    SERVE_APP,
    SERVE_GPU,
    SERVE_IMAGE,
    SERVE_MAX_CONTAINERS,
    SERVE_MIN_CONTAINERS,
    SERVE_PORT,
    SERVE_SCALEDOWN_WINDOW,
    SERVE_TIMEOUT,
    VOLUMES,
)


@SERVE_APP.cls(
    image=SERVE_IMAGE,
    gpu=SERVE_GPU,
    volumes=VOLUMES,
    timeout=SERVE_TIMEOUT,
    scaledown_window=SERVE_SCALEDOWN_WINDOW,
    min_containers=SERVE_MIN_CONTAINERS,
    max_containers=SERVE_MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=32)
class ServeCls:
    """Per-recipe vLLM endpoint, parameterised by ``recipe_name`` / ``model_path``.

    Modal's ``@modal.web_server`` requires a nullary method, so the recipe name
    and optional checkpoint override are passed via :func:`modal.parameter`
    rather than method arguments. Callers usually go through
    :data:`agent2model.cloud.modal_app.serve` which presents a natural
    ``serve.remote(recipe, model_path=None)`` API.
    """

    recipe_name: str = modal.parameter()
    model_path: str = modal.parameter(default="")

    @modal.web_server(port=SERVE_PORT, startup_timeout=600)  # type: ignore[untyped-decorator]
    def run(self) -> None:
        """Launch the OpenAI-compatible vLLM server inside the container."""
        from agent2model.serve import vllm_server

        base = self.model_path or f"{MODEL_ROOT}/{self.recipe_name}"
        resolved = vllm_server.resolve_model_path(base)
        vllm_server.serve(
            resolved,
            port=SERVE_PORT,
            host="0.0.0.0",
            served_model_name=self.recipe_name,
        )
