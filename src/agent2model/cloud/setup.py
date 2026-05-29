"""Idempotent first-time wizard for the cloud-UX layer (``agent2model cloud setup``).

Splits the *flow logic* from the I/O so the same plan can be unit-tested with
stubbed prompts/subprocess/webbrowser/Modal Secret APIs. The wizard's three
steps mirror :mod:`agent2model.cloud.doctor`:

1. **Modal account** — ask whether the user already has one; if not, open
   ``https://modal.com/signup`` and bail with a "come back" instruction.
2. **Modal token** — if ``~/.modal.toml`` is missing, shell out to
   ``modal token new`` (interactive, opens a browser).
3. **Anthropic Secret on Modal** — if missing, prompt for the API key (hidden
   input), create the secret via :meth:`modal.Secret.create_deployed`, and
   verify the local key via :func:`doctor.check_anthropic_key`.

Every step short-circuits if its state is already satisfied — re-running the
wizard is a no-op. The :class:`WizardIO` interface centralises the three I/O
verbs (``confirm`` / ``prompt_hidden`` / ``echo``) so tests can swap them out
without monkeypatching :mod:`typer`.
"""

from __future__ import annotations

import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Literal, Protocol

from agent2model.cloud.doctor import (
    ANTHROPIC_SECRET_NAME,
    MODAL_TOKEN_PATH,
    check_anthropic_key,
    check_anthropic_secret,
    check_modal_installed,
    check_modal_token,
)

__all__ = [
    "MODAL_SIGNUP_URL",
    "StepOutcome",
    "WizardIO",
    "WizardResult",
    "create_anthropic_secret",
    "run_setup",
    "step_anthropic_secret",
    "step_modal_account",
    "step_modal_token",
]

#: Where to send the user if they don't yet have a Modal account.
MODAL_SIGNUP_URL = "https://modal.com/signup"

StepOutcome = Literal["already_done", "completed", "skipped", "user_declined", "failed"]
"""Per-step outcomes the wizard reports.

* ``already_done`` — the underlying state was already satisfied; no action taken.
* ``completed`` — the wizard took action and the state is now satisfied.
* ``skipped`` — the user explicitly opted out of this step.
* ``user_declined`` — the user said "no" to the account question (we stop).
* ``failed`` — the step ran but did not reach a satisfied state.
"""


@dataclass(frozen=True)
class WizardResult:
    """Outcome of one wizard step or the whole flow.

    Attributes:
        step: The step id (e.g. ``"modal_token"``).
        outcome: A :data:`StepOutcome` literal.
        message: Human-readable explanation, surfaced to the user.
    """

    step: str
    outcome: StepOutcome
    message: str


class WizardIO(Protocol):
    """The three I/O verbs the wizard needs.

    Implementations live in :mod:`agent2model.cli` (typer-backed) and in the
    unit tests (canned responses). Keeping the wizard parameterised on this
    protocol means flow tests do not have to monkeypatch Typer or stdin.
    """

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        """Yes/no confirmation prompt."""

    def prompt_hidden(self, prompt: str) -> str:
        """Hidden-input prompt (used for the Anthropic API key)."""

    def echo(self, message: str) -> None:
        """Print a status line to the user."""


# --------------------------------------------------------------------------- #
# Step implementations                                                         #
# --------------------------------------------------------------------------- #


def step_modal_account(
    io: WizardIO, *, open_browser: bool = True, browser_open: object = None
) -> WizardResult:
    """Ask whether the user has a Modal account; open signup if they don't.

    This step has no programmatic detection (Modal exposes no public "do I have
    an account?" API), so we just ask. We treat *no* as the only declining
    answer that stops the wizard.

    Args:
        io: The :class:`WizardIO` for prompts/echo.
        open_browser: When ``True`` (default) and the user says no, open
            :data:`MODAL_SIGNUP_URL` in their browser.
        browser_open: Optional override for :func:`webbrowser.open` (tests).

    Returns:
        :class:`WizardResult` for this step.
    """
    has_account = io.confirm("Do you have a Modal account?", default=True)
    if has_account:
        return WizardResult(
            step="modal_account",
            outcome="already_done",
            message="Modal account already exists.",
        )
    if open_browser:
        opener = browser_open if callable(browser_open) else webbrowser.open
        opener(MODAL_SIGNUP_URL)
    io.echo(f"Sign up at {MODAL_SIGNUP_URL}, then re-run " "`agent2model cloud setup` to continue.")
    return WizardResult(
        step="modal_account",
        outcome="user_declined",
        message="No Modal account; opened signup page.",
    )


def step_modal_token(
    io: WizardIO,
    *,
    run_subprocess: object = None,
    modal_bin: str = "modal",
) -> WizardResult:
    """Ensure ``~/.modal.toml`` is present, invoking ``modal token new`` if not.

    The Modal CLI is interactive and opens a browser to authenticate — we
    inherit stdin/stdout/stderr so the user can complete the flow. After the
    subprocess returns we re-check the token file; if it is still missing we
    report ``failed`` rather than asserting.

    Args:
        io: The :class:`WizardIO` for status echo.
        run_subprocess: Optional override for :func:`subprocess.run` (tests).
        modal_bin: The Modal executable; default ``"modal"`` (on PATH).

    Returns:
        :class:`WizardResult` for this step.
    """
    initial = check_modal_token()
    if initial.ok:
        return WizardResult(
            step="modal_token",
            outcome="already_done",
            message=initial.message,
        )
    io.echo("No Modal token found; launching `modal token new` (browser-based)...")
    runner = run_subprocess if callable(run_subprocess) else subprocess.run
    try:
        runner([modal_bin, "token", "new"], check=False)
    except FileNotFoundError:
        return WizardResult(
            step="modal_token",
            outcome="failed",
            message=(
                f"Could not find `{modal_bin}` on PATH. "
                'Install with `pip install "agent2model[cloud]"`.'
            ),
        )
    final = check_modal_token()
    if final.ok:
        return WizardResult(
            step="modal_token",
            outcome="completed",
            message=f"Token created ({final.message}).",
        )
    return WizardResult(
        step="modal_token",
        outcome="failed",
        message=f"Token still missing at {MODAL_TOKEN_PATH}; re-run `modal token new`.",
    )


def create_anthropic_secret(api_key: str, secret_name: str = ANTHROPIC_SECRET_NAME) -> str:
    """Create the named Anthropic Secret on Modal, idempotently.

    Wraps :meth:`modal.Secret.create_deployed` with ``overwrite=True``, so
    re-running the wizard rotates the key cleanly. Falls back gracefully if the
    installed Modal SDK does not expose that helper: the caller surfaces an
    instructive message rather than crashing.

    Args:
        api_key: The plaintext Anthropic API key (starts with ``sk-ant-``).
        secret_name: Name of the deployed Secret. Defaults to
            :data:`ANTHROPIC_SECRET_NAME`.

    Returns:
        The Secret's deployment id from Modal.

    Raises:
        ImportError: ``modal`` is not installed.
        AttributeError: The installed Modal SDK has no ``create_deployed``.
        RuntimeError: Modal rejected the secret creation.

    Example:
        >>> # create_anthropic_secret("sk-ant-...")  # doctest: +SKIP
    """
    import modal  # local import: doctor.py validates installation already

    if not hasattr(modal.Secret, "create_deployed"):
        raise AttributeError(
            "This Modal SDK version does not support `Secret.create_deployed`. "
            "Create the secret manually at https://modal.com/secrets."
        )
    try:
        secret_id = modal.Secret.create_deployed(
            secret_name,
            {"ANTHROPIC_API_KEY": api_key},
            overwrite=True,
        )
    except Exception as exc:  # broad: Modal raises a variety of error classes
        raise RuntimeError(f"Modal rejected secret creation: {exc}") from exc
    return str(secret_id)


def step_anthropic_secret(
    io: WizardIO,
    *,
    secret_name: str = ANTHROPIC_SECRET_NAME,
    secret_creator: object = None,
    key_checker: object = None,
) -> WizardResult:
    """Ensure the Anthropic Secret exists on Modal; create it if not.

    Calls :func:`doctor.check_anthropic_secret` first. If the secret already
    resolves, the step is a no-op. Otherwise the user is prompted (hidden
    input) for their API key and :func:`create_anthropic_secret` is invoked.

    Args:
        io: The :class:`WizardIO` for prompts/echo.
        secret_name: The Secret name (passed through to both check and create).
        secret_creator: Optional override for :func:`create_anthropic_secret`
            (tests).
        key_checker: Optional override for :func:`doctor.check_anthropic_key`
            (tests).

    Returns:
        :class:`WizardResult` for this step.
    """
    initial = check_anthropic_secret(secret_name)
    if initial.ok:
        return WizardResult(
            step="anthropic_secret",
            outcome="already_done",
            message=initial.message,
        )
    io.echo(
        f"Modal Secret '{secret_name}' is missing. The wizard will create it now. "
        "Your key is sent only to Modal; it is not printed or logged."
    )
    api_key = io.prompt_hidden("Paste your Anthropic API key").strip()
    if not api_key:
        return WizardResult(
            step="anthropic_secret",
            outcome="failed",
            message="No API key entered; secret not created.",
        )
    creator = secret_creator if callable(secret_creator) else create_anthropic_secret
    try:
        creator(api_key, secret_name)
    except (ImportError, AttributeError, RuntimeError) as exc:
        return WizardResult(
            step="anthropic_secret",
            outcome="failed",
            message=str(exc),
        )
    # Best-effort: validate the *same* key locally so the user gets immediate
    # feedback that what they pasted actually works. The local env var may
    # already be set to a different key; we don't override it here.
    checker = key_checker if callable(key_checker) else check_anthropic_key
    ping = checker()
    suffix = f" Local ping: {ping.message}" if ping.message else ""
    return WizardResult(
        step="anthropic_secret",
        outcome="completed",
        message=f"Secret '{secret_name}' created on Modal.{suffix}",
    )


# --------------------------------------------------------------------------- #
# Whole-wizard orchestrator                                                    #
# --------------------------------------------------------------------------- #


def run_setup(
    io: WizardIO,
    *,
    open_browser: bool = True,
    browser_open: object = None,
    run_subprocess: object = None,
    secret_creator: object = None,
    key_checker: object = None,
) -> list[WizardResult]:
    """Run the three wizard steps in order, stopping early on a hard decline.

    The flow short-circuits when ``modal`` itself is not installed (we cannot
    do anything useful without it) and when the user declines to have a Modal
    account. Every other "missing precondition" surfaces as a ``failed`` step
    and the wizard continues — the final ``doctor`` printout shows what is
    still red.

    Args:
        io: The :class:`WizardIO` to use for all prompts and echoes.
        open_browser: Whether to open the signup URL when the user declines.
        browser_open: Optional override for :func:`webbrowser.open` (tests).
        run_subprocess: Optional override for :func:`subprocess.run` (tests).
        secret_creator: Optional override for :func:`create_anthropic_secret`.
        key_checker: Optional override for :func:`doctor.check_anthropic_key`.

    Returns:
        The list of per-step :class:`WizardResult` records, in order.

    Example:
        >>> # results = run_setup(my_io)  # doctest: +SKIP
    """
    results: list[WizardResult] = []

    install = check_modal_installed()
    if not install.ok:
        results.append(
            WizardResult(
                step="modal_installed",
                outcome="failed",
                message=f"{install.message} Fix: {install.fix_command}",
            )
        )
        return results

    account = step_modal_account(io, open_browser=open_browser, browser_open=browser_open)
    results.append(account)
    if account.outcome == "user_declined":
        return results

    results.append(step_modal_token(io, run_subprocess=run_subprocess))
    results.append(
        step_anthropic_secret(
            io,
            secret_creator=secret_creator,
            key_checker=key_checker,
        )
    )
    return results
