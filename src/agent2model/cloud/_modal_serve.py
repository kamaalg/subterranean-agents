"""Modal class for the autoscaling vLLM serve endpoint.

The class is intentionally **non-parameterised** — ``@modal.parameter`` +
``@modal.web_server`` conflict at the URL routing layer (parameter dispatch
uses the 303-poll function-call pattern, but ``web_server`` needs Modal to
proxy raw HTTP to a long-running server). Empirically this manifests as
infinite 303 redirects on every incoming request.

To serve a different recipe, set the env var ``AGENT2MODEL_SERVE_RECIPE`` on
the Modal Secret ``agent2model-serve-config`` (or in the deployment) and
redeploy. One deployment = one served recipe. Trades flexibility for an HTTP
path that actually works.

Lives in its own module because the rest of :mod:`agent2model.cloud.modal_app`
carries ``from __future__ import annotations`` (needed for the worker function
signatures), and Modal's class-discovery has historically been fragile around
stringified annotations.
"""

import os

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

#: Env var read at container start to pick which recipe to serve.
SERVE_RECIPE_ENV = "AGENT2MODEL_SERVE_RECIPE"
#: Optional env var to override the on-volume model path.
SERVE_MODEL_PATH_ENV = "AGENT2MODEL_SERVE_MODEL_PATH"
#: Fallback recipe when neither env var nor secret is set. Aligns with the
#: included travel-booking example so a fresh deploy "just serves" something.
SERVE_DEFAULT_RECIPE = "travel_booking"


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
    """Single-recipe vLLM endpoint exposed via a stable Modal HTTP URL.

    Reads the recipe from ``$AGENT2MODEL_SERVE_RECIPE`` (default:
    ``travel_booking``) at container startup. To serve a different recipe,
    update the deployment's env / Modal Secret and redeploy.
    """

    @modal.web_server(port=SERVE_PORT, startup_timeout=600)  # type: ignore[untyped-decorator]
    def run(self) -> None:
        """Launch the OpenAI-compatible vLLM server inside the container."""
        from agent2model.serve import vllm_server

        recipe_name = os.environ.get(SERVE_RECIPE_ENV, SERVE_DEFAULT_RECIPE)
        model_path = os.environ.get(SERVE_MODEL_PATH_ENV, "")
        base = model_path or f"{MODEL_ROOT}/{recipe_name}"
        resolved = vllm_server.resolve_model_path(base)
        vllm_server.serve(
            resolved,
            port=SERVE_PORT,
            host="0.0.0.0",
            served_model_name=recipe_name,
        )
