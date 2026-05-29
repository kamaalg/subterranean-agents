"""Unit tests for :mod:`agent2model.cloud.doctor`.

Every check is tested in isolation by monkeypatching the underlying
dependency (env vars, filesystem, Modal SDK, Anthropic SDK, ``huggingface_hub``).
Both the green path and each red path are exercised so the CLI's exit-code
contract is locked in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from agent2model.cloud import doctor
from agent2model.cloud.doctor import (
    CheckResult,
    check_anthropic_key,
    check_anthropic_secret,
    check_hf_token,
    check_modal_installed,
    check_modal_token,
    overall_exit_code,
    run_all_checks,
)

# --------------------------------------------------------------------------- #
# CheckResult model                                                            #
# --------------------------------------------------------------------------- #


def test_check_result_is_frozen() -> None:
    from pydantic import ValidationError

    r = CheckResult(name="x", ok=True, severity="critical")
    with pytest.raises(ValidationError):
        r.ok = False  # type: ignore[misc]


def test_check_result_defaults_to_critical() -> None:
    r = CheckResult(name="x", ok=True)
    assert r.severity == "critical"
    assert r.message == ""
    assert r.fix_command == ""


# --------------------------------------------------------------------------- #
# check_modal_installed                                                        #
# --------------------------------------------------------------------------- #


def test_check_modal_installed_green_when_importable() -> None:
    pytest.importorskip("modal")
    result = check_modal_installed()
    assert result.ok is True
    assert result.severity == "critical"
    assert "modal" in result.message


def test_check_modal_installed_red_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "modal":
            raise ImportError("no modal here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "modal", None)
    # Wipe any cached modal so the `import modal` inside the check actually runs.
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    monkeypatch.setattr("builtins.__import__", _fake_import)
    result = check_modal_installed()
    assert result.ok is False
    assert "cloud" in result.fix_command  # pip install "...[cloud]"
    assert "pip install" in result.fix_command
    assert result.severity == "critical"


# --------------------------------------------------------------------------- #
# check_modal_token                                                            #
# --------------------------------------------------------------------------- #


def test_check_modal_token_missing(tmp_path: Path) -> None:
    result = check_modal_token(tmp_path / "ghost.toml")
    assert result.ok is False
    assert "modal token new" in result.fix_command


def test_check_modal_token_empty(tmp_path: Path) -> None:
    token = tmp_path / ".modal.toml"
    token.write_text("", encoding="utf-8")
    result = check_modal_token(token)
    assert result.ok is False
    assert "empty" in result.message


def test_check_modal_token_workspace_parsed(tmp_path: Path) -> None:
    token = tmp_path / ".modal.toml"
    token.write_text(
        '[my-workspace]\ntoken_id = "ak-xx"\ntoken_secret = "as-yy"\nactive = true\n',
        encoding="utf-8",
    )
    result = check_modal_token(token)
    assert result.ok is True
    assert "my-workspace" in result.message


def test_check_modal_token_no_active_falls_back_to_first(tmp_path: Path) -> None:
    token = tmp_path / ".modal.toml"
    token.write_text('[first-ws]\ntoken_id = "ak"\n', encoding="utf-8")
    result = check_modal_token(token)
    assert result.ok is True
    assert "first-ws" in result.message


# --------------------------------------------------------------------------- #
# check_anthropic_secret                                                       #
# --------------------------------------------------------------------------- #


def test_check_anthropic_secret_green(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    import modal

    class _FakeHandle:
        def hydrate(self) -> None:
            return None

    monkeypatch.setattr(modal.Secret, "from_name", lambda _name: _FakeHandle())
    result = check_anthropic_secret("anthropic-secret")
    assert result.ok is True
    assert "Modal" in result.message or "resolved" in result.message


def test_check_anthropic_secret_red_when_hydrate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    import modal

    class _Boom:
        def hydrate(self) -> None:
            raise RuntimeError("NotFound: anthropic-secret")

    monkeypatch.setattr(modal.Secret, "from_name", lambda _name: _Boom())
    result = check_anthropic_secret()
    assert result.ok is False
    assert "anthropic-secret" in result.fix_command
    assert "NotFound" in result.message


# --------------------------------------------------------------------------- #
# check_anthropic_key                                                          #
# --------------------------------------------------------------------------- #


def test_check_anthropic_key_skipped_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = check_anthropic_key()
    assert result.ok is True
    assert result.severity == "informational"
    assert "skipped" in result.message


def test_check_anthropic_key_green_on_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class _FakeMessages:
        def create(self, **kwargs: Any) -> object:
            return object()

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: _FakeClient())
    result = check_anthropic_key(model="claude-haiku-4-5")
    assert result.ok is True
    assert result.severity == "informational"
    assert "claude-haiku-4-5" in result.message


def test_check_anthropic_key_red_when_ping_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class _FakeMessages:
        def create(self, **kwargs: Any) -> object:
            raise RuntimeError("billing error")

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: _FakeClient())
    result = check_anthropic_key()
    assert result.ok is False
    assert result.severity == "informational"
    assert "billing" in result.message
    # The local-vs-Modal nuance is surfaced in the message.
    assert "LOCAL" in result.message or "local" in result.message


# --------------------------------------------------------------------------- #
# check_hf_token                                                               #
# --------------------------------------------------------------------------- #


def test_check_hf_token_skipped_when_no_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # Re-point HOME to a temp dir so the cache file is guaranteed absent.
    monkeypatch.setenv("HOME", str(tmp_path))
    # The doctor uses Path.home() captured at module import; patch the function.
    monkeypatch.setattr("agent2model.cloud.doctor.Path.home", lambda: tmp_path)
    result = check_hf_token()
    assert result.ok is True
    assert result.severity == "informational"
    assert "skipped" in result.message


def test_check_hf_token_green_when_whoami_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    class _FakeApi:
        def whoami(self, token: str | None = None) -> dict[str, str]:
            return {"name": "test-user"}

    import types

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.HfApi = _FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    result = check_hf_token()
    assert result.ok is True
    assert "test-user" in result.message


def test_check_hf_token_red_when_whoami_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_bad")

    class _BadApi:
        def whoami(self, token: str | None = None) -> dict[str, str]:
            raise RuntimeError("401")

    import types

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.HfApi = _BadApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    result = check_hf_token()
    assert result.ok is False
    assert result.severity == "informational"


# --------------------------------------------------------------------------- #
# overall_exit_code + run_all_checks                                           #
# --------------------------------------------------------------------------- #


def test_overall_exit_code_zero_on_all_green() -> None:
    assert overall_exit_code([CheckResult(name="x", ok=True)]) == 0


def test_overall_exit_code_one_on_critical_failure() -> None:
    assert overall_exit_code([CheckResult(name="x", ok=False, severity="critical")]) == 1


def test_overall_exit_code_unaffected_by_informational_failure() -> None:
    results = [
        CheckResult(name="a", ok=True, severity="critical"),
        CheckResult(name="b", ok=False, severity="informational"),
    ]
    assert overall_exit_code(results) == 0


def test_run_all_checks_returns_five_results(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub each underlying check to keep this fully offline.
    monkeypatch.setattr(doctor, "check_modal_installed", lambda: CheckResult(name="m", ok=True))
    monkeypatch.setattr(doctor, "check_modal_token", lambda: CheckResult(name="t", ok=True))
    monkeypatch.setattr(doctor, "check_anthropic_secret", lambda: CheckResult(name="s", ok=True))
    monkeypatch.setattr(
        doctor,
        "check_anthropic_key",
        lambda: CheckResult(name="k", ok=True, severity="informational"),
    )
    monkeypatch.setattr(
        doctor,
        "check_hf_token",
        lambda: CheckResult(name="h", ok=True, severity="informational"),
    )
    results = run_all_checks()
    assert len(results) == 5
    assert [r.name for r in results] == ["m", "t", "s", "k", "h"]
