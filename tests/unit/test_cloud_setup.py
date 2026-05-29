"""Unit tests for :mod:`agent2model.cloud.setup`.

Each step is exercised through the pure :class:`WizardIO` protocol with canned
responses, plus mocks for the Modal/subprocess/webbrowser side effects. The
idempotency contract (steps short-circuit when state is already satisfied) is
the central guarantee.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent2model.cloud import setup as setup_mod
from agent2model.cloud.doctor import CheckResult
from agent2model.cloud.setup import (
    WizardResult,
    create_anthropic_secret,
    run_setup,
    step_anthropic_secret,
    step_modal_account,
    step_modal_token,
)

# --------------------------------------------------------------------------- #
# Fake IO                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class FakeIO:
    """Test double for the :class:`WizardIO` protocol."""

    confirm_answers: list[bool] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    echoes: list[str] = field(default_factory=list)
    hidden_value: str = ""

    def _next_confirm(self) -> bool:
        return self.confirm_answers.pop(0) if self.confirm_answers else False

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        self.prompts.append(prompt)
        return self._next_confirm()

    def prompt_hidden(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.hidden_value

    def echo(self, message: str) -> None:
        self.echoes.append(message)


# --------------------------------------------------------------------------- #
# step_modal_account                                                            #
# --------------------------------------------------------------------------- #


def test_step_modal_account_user_has_account() -> None:
    io = FakeIO(confirm_answers=[True])
    result = step_modal_account(io, open_browser=False)
    assert result.outcome == "already_done"
    assert "exists" in result.message


def test_step_modal_account_user_declines_opens_browser() -> None:
    io = FakeIO(confirm_answers=[False])
    opens: list[str] = []
    result = step_modal_account(
        io, open_browser=True, browser_open=lambda url: opens.append(url) or True
    )
    assert result.outcome == "user_declined"
    assert opens == [setup_mod.MODAL_SIGNUP_URL]
    assert any("modal.com/signup" in e for e in io.echoes)


def test_step_modal_account_skip_browser_when_disabled() -> None:
    io = FakeIO(confirm_answers=[False])
    opened: list[str] = []
    result = step_modal_account(
        io, open_browser=False, browser_open=lambda url: opened.append(url) or True
    )
    assert result.outcome == "user_declined"
    # ``open_browser=False`` short-circuits the opener entirely.
    assert opened == []


# --------------------------------------------------------------------------- #
# step_modal_token                                                              #
# --------------------------------------------------------------------------- #


def test_step_modal_token_idempotent_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = tmp_path / ".modal.toml"
    token.write_text("[ws]\ntoken_id = 'ak-x'\nactive = true\n", encoding="utf-8")
    monkeypatch.setattr(setup_mod, "MODAL_TOKEN_PATH", token)
    monkeypatch.setattr(
        setup_mod,
        "check_modal_token",
        lambda: (
            setup_mod.check_modal_token.__wrapped__(token)
            if hasattr(setup_mod.check_modal_token, "__wrapped__")
            else CheckResult(name="t", ok=True, message="workspace: ws")
        ),
    )
    io = FakeIO()
    spawned: list[list[str]] = []
    result = step_modal_token(
        io, run_subprocess=lambda argv, check=False: spawned.append(argv) or None
    )
    assert result.outcome == "already_done"
    assert spawned == []  # No subprocess call when already present.


def test_step_modal_token_runs_modal_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # First check returns red, second returns green to simulate token creation.
    states: Iterator[CheckResult] = iter(
        [
            CheckResult(name="t", ok=False, message="missing", fix_command="modal token new"),
            CheckResult(name="t", ok=True, message="workspace: foo"),
        ]
    )
    monkeypatch.setattr(setup_mod, "check_modal_token", lambda: next(states))
    spawned: list[list[str]] = []
    io = FakeIO()
    result = step_modal_token(
        io,
        run_subprocess=lambda argv, check=False: spawned.append(argv) or None,
    )
    assert result.outcome == "completed"
    assert spawned == [["modal", "token", "new"]]


def test_step_modal_token_reports_failure_when_still_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_modal_token",
        lambda: CheckResult(name="t", ok=False, message="missing"),
    )
    io = FakeIO()
    result = step_modal_token(io, run_subprocess=lambda argv, check=False: None)
    assert result.outcome == "failed"


def test_step_modal_token_handles_missing_modal_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_modal_token",
        lambda: CheckResult(name="t", ok=False, message="missing"),
    )

    def _boom(_argv: list[str], check: bool = False) -> None:
        raise FileNotFoundError("modal")

    io = FakeIO()
    result = step_modal_token(io, run_subprocess=_boom)
    assert result.outcome == "failed"
    assert "PATH" in result.message


# --------------------------------------------------------------------------- #
# step_anthropic_secret                                                         #
# --------------------------------------------------------------------------- #


def test_step_anthropic_secret_idempotent_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_anthropic_secret",
        lambda name: CheckResult(name="s", ok=True, message="resolved"),
    )
    io = FakeIO()
    called: list[Any] = []
    result = step_anthropic_secret(io, secret_creator=lambda *a, **k: called.append(a) or "id")
    assert result.outcome == "already_done"
    assert called == []  # Secret already exists; we never call the creator.


def test_step_anthropic_secret_creates_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_anthropic_secret",
        lambda name: CheckResult(name="s", ok=False, message="missing"),
    )
    io = FakeIO(hidden_value="sk-ant-test")
    created: list[tuple[str, str]] = []
    result = step_anthropic_secret(
        io,
        secret_creator=lambda key, name: created.append((key, name)) or "secret-id",
        key_checker=lambda: CheckResult(
            name="k", ok=True, severity="informational", message="ping ok"
        ),
    )
    assert result.outcome == "completed"
    assert created == [("sk-ant-test", "anthropic-secret")]
    assert "ping ok" in result.message


def test_step_anthropic_secret_empty_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_anthropic_secret",
        lambda name: CheckResult(name="s", ok=False, message="missing"),
    )
    io = FakeIO(hidden_value="   ")
    result = step_anthropic_secret(
        io,
        secret_creator=lambda *a, **k: pytest.fail("should not be called"),
    )
    assert result.outcome == "failed"
    assert "No API key" in result.message


def test_step_anthropic_secret_creator_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_anthropic_secret",
        lambda name: CheckResult(name="s", ok=False, message="missing"),
    )

    def _boom(_key: str, _name: str) -> str:
        raise RuntimeError("rejected by modal")

    io = FakeIO(hidden_value="sk-ant-test")
    result = step_anthropic_secret(io, secret_creator=_boom)
    assert result.outcome == "failed"
    assert "rejected" in result.message


# --------------------------------------------------------------------------- #
# create_anthropic_secret                                                       #
# --------------------------------------------------------------------------- #


def test_create_anthropic_secret_uses_modal_api(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    import modal

    calls: list[dict[str, Any]] = []

    def _fake_create_deployed(name: str, env: dict[str, str], overwrite: bool = False) -> str:
        calls.append({"name": name, "env": env, "overwrite": overwrite})
        return "se-fake"

    monkeypatch.setattr(modal.Secret, "create_deployed", staticmethod(_fake_create_deployed))
    out = create_anthropic_secret("sk-ant-zzz")
    assert out == "se-fake"
    assert calls == [
        {"name": "anthropic-secret", "env": {"ANTHROPIC_API_KEY": "sk-ant-zzz"}, "overwrite": True}
    ]


def test_create_anthropic_secret_falls_back_when_api_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("modal")
    import modal

    monkeypatch.delattr(modal.Secret, "create_deployed", raising=False)
    with pytest.raises(AttributeError, match="create_deployed"):
        create_anthropic_secret("sk-ant-zzz")


# --------------------------------------------------------------------------- #
# run_setup                                                                     #
# --------------------------------------------------------------------------- #


def test_run_setup_short_circuits_when_modal_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_mod,
        "check_modal_installed",
        lambda: CheckResult(
            name="m", ok=False, message="not installed", fix_command="pip install ..."
        ),
    )
    io = FakeIO(confirm_answers=[True])
    results = run_setup(io)
    assert len(results) == 1
    assert results[0].outcome == "failed"
    assert results[0].step == "modal_installed"


def test_run_setup_stops_when_user_declines_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_mod, "check_modal_installed", lambda: CheckResult(name="m", ok=True))
    io = FakeIO(confirm_answers=[False])  # "no" to "do you have a Modal account?"
    results = run_setup(io, open_browser=False)
    steps = [r.step for r in results]
    assert steps == ["modal_account"]
    assert results[0].outcome == "user_declined"


def test_run_setup_runs_all_three_when_state_is_satisfiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_mod, "check_modal_installed", lambda: CheckResult(name="m", ok=True))
    # token already present, secret already present -> two no-ops after the
    # account question.
    monkeypatch.setattr(
        setup_mod,
        "check_modal_token",
        lambda: CheckResult(name="t", ok=True, message="workspace: ws"),
    )
    monkeypatch.setattr(
        setup_mod, "check_anthropic_secret", lambda name="x": CheckResult(name="s", ok=True)
    )
    io = FakeIO(confirm_answers=[True])
    results = run_setup(io, open_browser=False)
    assert [r.step for r in results] == ["modal_account", "modal_token", "anthropic_secret"]
    assert all(r.outcome == "already_done" for r in results)


def test_wizard_result_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    r = WizardResult(step="x", outcome="completed", message="ok")
    with pytest.raises(FrozenInstanceError):
        r.message = "oops"  # type: ignore[misc]
