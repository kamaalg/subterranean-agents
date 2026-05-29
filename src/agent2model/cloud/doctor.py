"""Pure preflight checks for the cloud-UX layer (``agent2model cloud doctor``).

This module is deliberately split from the Typer command so each check can be
unit-tested in isolation with monkeypatched dependencies. Every check returns a
:class:`CheckResult` carrying its name, a green/red outcome, a one-line message,
and an actionable fix command. The CLI in :mod:`agent2model.cli` just calls
these and renders the result with Rich.

The five checks, in order:

1. :func:`check_modal_installed` — ``import modal`` succeeds.
2. :func:`check_modal_token` — ``~/.modal.toml`` exists and is non-empty.
3. :func:`check_anthropic_secret` — the ``anthropic-secret`` Modal Secret
   resolves in the user's workspace (we hydrate the handle; we do not read the
   value).
4. :func:`check_anthropic_key` — informational. If ``ANTHROPIC_API_KEY`` is set
   *locally*, we make a 1-token ping to ``claude-haiku-4-5`` to confirm the key
   bills. Skipped cleanly if unset; this hits the local env var, not the secret
   stored on Modal (those are checked separately).
5. :func:`check_hf_token` — informational; only runs if an HF token is
   discoverable.

Only the first three are *critical* (their failure flips :attr:`CheckResult.ok`
and the CLI exits non-zero). The Anthropic / HF probes are informational —
they help users diagnose but never block.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ANTHROPIC_PING_MODEL",
    "ANTHROPIC_SECRET_NAME",
    "MODAL_TOKEN_PATH",
    "CheckResult",
    "CheckSeverity",
    "check_anthropic_key",
    "check_anthropic_secret",
    "check_hf_token",
    "check_modal_installed",
    "check_modal_token",
    "run_all_checks",
]

CheckSeverity = Literal["critical", "informational"]
"""How a check contributes to the overall ``doctor`` exit code.

* ``critical`` — a red result flips ``doctor`` to exit 1.
* ``informational`` — a red result is rendered but does not change the exit code.
"""

#: Path to the Modal CLI's local token file. Modal writes this when the user
#: runs ``modal token new`` and it contains workspace + ``ak-…`` / ``as-…``
#: tokens. We only check it for existence and non-emptiness; we never parse the
#: secret material.
MODAL_TOKEN_PATH = Path.home() / ".modal.toml"

#: The Modal Secret name we expect to find in the user's workspace. The Modal
#: workers in :mod:`agent2model.cloud.modal_app` look this up under the same
#: name; if it is missing the API-bound steps fail at runtime.
ANTHROPIC_SECRET_NAME = "anthropic-secret"

#: A cheap, current model used purely to validate the local ``ANTHROPIC_API_KEY``
#: bills. ``claude-haiku-4-5`` is the cheapest in :data:`generator._PRICING`.
ANTHROPIC_PING_MODEL = "claude-haiku-4-5"


class CheckResult(BaseModel):
    """The outcome of a single preflight check.

    Attributes:
        name: A short, human-readable label printed verbatim by the CLI.
        ok: ``True`` if the check passed (green); ``False`` for a failure (red).
        severity: ``"critical"`` failures flip the ``doctor`` exit code to 1;
            ``"informational"`` results never do.
        message: A one-line status string, shown on the same row as the mark.
        fix_command: An actionable shell command (or short hint) the user can
            copy-paste to address a failure. Empty when ``ok`` is True or the
            check has no obvious fix.

    Example:
        >>> CheckResult(name="modal installed", ok=True, severity="critical",
        ...             message="modal 1.4.3").ok
        True
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ok: bool
    severity: CheckSeverity = "critical"
    message: str = ""
    fix_command: str = ""


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #


def check_modal_installed() -> CheckResult:
    """Confirm the ``modal`` Python package is importable.

    Returns:
        A critical :class:`CheckResult`. On failure the ``fix_command`` is the
        ``pip install`` extra.

    Example:
        >>> result = check_modal_installed()
        >>> result.name
        'modal package installed'
    """
    try:
        import modal  # noqa: F401
    except ImportError:
        return CheckResult(
            name="modal package installed",
            ok=False,
            severity="critical",
            message="`import modal` failed.",
            fix_command='pip install "agent2model[cloud]"',
        )
    try:
        version = getattr(__import__("modal"), "__version__", "unknown")
    except Exception:  # pragma: no cover - defensive
        version = "unknown"
    return CheckResult(
        name="modal package installed",
        ok=True,
        severity="critical",
        message=f"modal {version}",
    )


def check_modal_token(token_path: Path | None = None) -> CheckResult:
    """Confirm a Modal token file exists at ``~/.modal.toml``.

    The check is intentionally cheap — it does not contact Modal; it only
    verifies the local token file is present and non-empty. The workspace name
    (the first TOML section header) is surfaced when readable.

    Args:
        token_path: Override the token file path. Used by tests; production code
            should leave it unset to read :data:`MODAL_TOKEN_PATH`.

    Returns:
        A critical :class:`CheckResult`.

    Example:
        >>> isinstance(check_modal_token(Path("/nonexistent")).ok, bool)
        True
    """
    path = token_path or MODAL_TOKEN_PATH
    if not path.exists():
        return CheckResult(
            name="modal token configured",
            ok=False,
            severity="critical",
            message=f"No token file at {path}.",
            fix_command="modal token new",
        )
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return CheckResult(
            name="modal token configured",
            ok=False,
            severity="critical",
            message=f"Could not read {path}: {exc}.",
            fix_command="modal token new",
        )
    if not contents:
        return CheckResult(
            name="modal token configured",
            ok=False,
            severity="critical",
            message=f"{path} is empty.",
            fix_command="modal token new",
        )
    workspace = _parse_workspace(contents)
    msg = f"workspace: {workspace}" if workspace else f"present at {path}"
    return CheckResult(
        name="modal token configured",
        ok=True,
        severity="critical",
        message=msg,
    )


def _parse_workspace(contents: str) -> str | None:
    """Return the active workspace name from a ``~/.modal.toml`` file.

    The Modal CLI writes one ``[workspace_name]`` section per profile, with an
    ``active = true`` line marking the current one. We do a tiny line-based
    scan rather than depending on ``tomllib`` to keep the check minimal.

    Args:
        contents: The full token-file contents.

    Returns:
        The active workspace name, or the first section's name as a fallback,
        or ``None`` if no section header was found.
    """
    sections: list[str] = []
    active: str | None = None
    current: str | None = None
    for raw in contents.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.append(current)
            continue
        if current and line.startswith("active") and "true" in line.lower():
            active = current
    return active or (sections[0] if sections else None)


def check_anthropic_secret(secret_name: str = ANTHROPIC_SECRET_NAME) -> CheckResult:
    """Confirm a Modal Secret exists in the user's workspace.

    We construct a handle via :func:`modal.Secret.from_name` and call
    :meth:`hydrate` — this contacts Modal and resolves the secret's id without
    exposing its plaintext. Any failure (not found, wrong env, auth) is folded
    into a single red result.

    Args:
        secret_name: The Modal Secret to look up. Defaults to
            :data:`ANTHROPIC_SECRET_NAME`.

    Returns:
        A critical :class:`CheckResult`.

    Example:
        >>> result = check_anthropic_secret("anthropic-secret")
        >>> result.name
        "modal secret 'anthropic-secret' exists"
    """
    label = f"modal secret '{secret_name}' exists"
    try:
        import modal
    except ImportError:
        return CheckResult(
            name=label,
            ok=False,
            severity="critical",
            message="modal not installed; cannot look up secrets.",
            fix_command='pip install "agent2model[cloud]"',
        )
    try:
        handle = modal.Secret.from_name(secret_name)
        handle.hydrate()
    except Exception as exc:  # broad: Modal raises NotFoundError, AuthError, etc.
        return CheckResult(
            name=label,
            ok=False,
            severity="critical",
            message=f"could not resolve secret: {exc}",
            fix_command=(
                "agent2model cloud setup  "
                "# or: modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-..."
            ),
        )
    return CheckResult(
        name=label,
        ok=True,
        severity="critical",
        message="resolved on Modal.",
    )


def check_anthropic_key(model: str = ANTHROPIC_PING_MODEL) -> CheckResult:
    """Confirm the **local** ``ANTHROPIC_API_KEY`` bills against Anthropic.

    This is intentionally informational: a local key is unrelated to the Modal
    Secret used by remote workers (that is :func:`check_anthropic_secret`). The
    distinction matters because the Modal Secret can be valid while the
    developer's local key is missing or revoked, and vice versa.

    Skips cleanly if ``ANTHROPIC_API_KEY`` is unset.

    Args:
        model: Anthropic model id used for the 1-token ping. Default is the
            cheapest model in our pricing table.

    Returns:
        An informational :class:`CheckResult`. Never causes a non-zero exit.

    Example:
        >>> check_anthropic_key().severity
        'informational'
    """
    label = "local ANTHROPIC_API_KEY bills"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult(
            name=label,
            ok=True,
            severity="informational",
            message="skipped: ANTHROPIC_API_KEY not set in local env.",
        )
    try:
        from anthropic import Anthropic
    except ImportError:
        return CheckResult(
            name=label,
            ok=False,
            severity="informational",
            message="anthropic SDK not installed.",
            fix_command='pip install "anthropic>=0.40"',
        )
    try:
        client = Anthropic()
        client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:
        return CheckResult(
            name=label,
            ok=False,
            severity="informational",
            message=(
                f"ping to {model} failed: {exc}. "
                "Note this checks the LOCAL env var, not the Modal secret."
            ),
            fix_command="export ANTHROPIC_API_KEY=sk-ant-...   # then re-run",
        )
    return CheckResult(
        name=label,
        ok=True,
        severity="informational",
        message=f"1-token ping to {model} succeeded.",
    )


def check_hf_token() -> CheckResult:
    """Validate a Hugging Face token *if* one is discoverable.

    Most users do not need an HF token — Qwen 2.5 and Qwen3 are ungated. We only
    bother running ``HfApi().whoami()`` when an env var or cached token is
    present. The check is informational; a missing or invalid token never blocks
    ``doctor``.

    Returns:
        An informational :class:`CheckResult`.

    Example:
        >>> check_hf_token().severity
        'informational'
    """
    label = "HF token valid (if any)"
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cached_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not env_token and not cached_token_path.exists():
        return CheckResult(
            name=label,
            ok=True,
            severity="informational",
            message="skipped: no HF token in env or cache (fine; Qwen models are ungated).",
        )
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError:
        return CheckResult(
            name=label,
            ok=True,
            severity="informational",
            message="skipped: `huggingface_hub` not installed.",
        )
    try:
        info = HfApi().whoami(token=env_token) if env_token else HfApi().whoami()
        name = info.get("name") if isinstance(info, dict) else None
        return CheckResult(
            name=label,
            ok=True,
            severity="informational",
            message=f"authenticated as {name}." if name else "authenticated.",
        )
    except Exception as exc:
        return CheckResult(
            name=label,
            ok=False,
            severity="informational",
            message=f"whoami failed: {exc}.",
            fix_command="huggingface-cli login",
        )


def run_all_checks() -> list[CheckResult]:
    """Run every preflight check in the documented order.

    Returns:
        Five :class:`CheckResult` instances in spec order: modal install,
        modal token, anthropic secret, local Anthropic key, HF token.

    Example:
        >>> results = run_all_checks()
        >>> [r.name for r in results][:2]
        ['modal package installed', 'modal token configured']
    """
    return [
        check_modal_installed(),
        check_modal_token(),
        check_anthropic_secret(),
        check_anthropic_key(),
        check_hf_token(),
    ]


def overall_exit_code(results: Sequence[CheckResult]) -> int:
    """Return the ``doctor`` exit code for a sequence of check results.

    Returns ``0`` if every *critical* check passed, ``1`` otherwise.
    Informational failures are surfaced visually but never flip the code.

    Args:
        results: The :class:`CheckResult` sequence.

    Returns:
        ``0`` on overall pass, ``1`` on any critical failure.

    Example:
        >>> overall_exit_code([CheckResult(name="x", ok=True, severity="critical")])
        0
        >>> overall_exit_code([CheckResult(name="x", ok=False, severity="critical")])
        1
        >>> overall_exit_code([CheckResult(name="x", ok=False, severity="informational")])
        0
    """
    for r in results:
        if r.severity == "critical" and not r.ok:
            return 1
    return 0
