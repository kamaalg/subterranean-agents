"""Unit tests for the Typer CLI surface — focused on the ``cloud run`` group.

The ``cloud run`` subcommand wraps ``modal run -m agent2model.cloud.modal_app::run``
in a subprocess. We test:

- The argv it builds for the typed flags (pure helper :func:`_build_modal_run_argv`).
- That ``--help`` shows the documented flags.
- That ``--dry-run`` prints the constructed command and exits 0 without invoking
  ``modal``.
- That the subprocess is actually invoked with the expected argv (mocked).

No real ``modal`` calls happen. The harness is fast, deterministic, and offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent2model.cli import MODAL_RUN_TARGET, _build_modal_run_argv, app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cloud_run_help_lists_documented_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["cloud", "run", "--help"])
    assert result.exit_code == 0, result.output
    # Help text wraps so we just check the flag tokens individually.
    for flag in (
        "--name",
        "--size",
        "--n",
        "--epochs",
        "--eval-n",
        "--base-model",
        "--skip-eval",
        "--serve-after",
        "--dry-run",
    ):
        assert flag in result.output, f"missing {flag} in help output:\n{result.output}"


def test_cloud_subcommand_listed_in_main_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "cloud" in result.output


def test_build_modal_run_argv_minimal(tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    argv = _build_modal_run_argv(
        path,
        name=None,
        size="3b",
        n=2000,
        epochs=20,
        eval_n=200,
        base_model=None,
        skip_eval=False,
        serve_after=False,
    )
    assert argv[:4] == ["modal", "run", "-m", MODAL_RUN_TARGET]
    assert argv[4] == "--"
    assert "--flowchart-path" in argv
    assert str(path) in argv
    assert "--size" in argv and "3b" in argv
    assert "--n" in argv and "2000" in argv
    assert "--epochs" in argv and "20" in argv
    assert "--eval-n" in argv and "200" in argv
    # Optional flags must NOT appear when defaulted.
    for absent in ("--name", "--base-model", "--skip-eval", "--serve-after"):
        assert absent not in argv


def test_build_modal_run_argv_all_overrides(tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    argv = _build_modal_run_argv(
        path,
        name="my_flow",
        size="8b",
        n=500,
        epochs=5,
        eval_n=50,
        base_model="meta-llama/Llama-3-8b",
        skip_eval=True,
        serve_after=True,
    )
    assert "--name" in argv and "my_flow" in argv
    assert "--base-model" in argv and "meta-llama/Llama-3-8b" in argv
    assert "--skip-eval" in argv
    assert "--serve-after" in argv
    assert "--size" in argv and "8b" in argv


def test_cloud_run_dry_run_prints_command(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["cloud", "run", str(path), "--size", "3b", "--n", "10", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    out = result.output.strip()
    assert out.startswith("modal run -m agent2model.cloud.modal_app::run --")
    assert "--flowchart-path" in out
    assert str(path.resolve()) in out
    assert "--size 3b" in out
    assert "--n 10" in out


def test_cloud_run_invokes_subprocess(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    class _FakeCompleted:
        def __init__(self, code: int) -> None:
            self.returncode = code

    def _fake_run(argv: list[str], check: bool = False) -> _FakeCompleted:
        captured["argv"] = argv
        captured["check"] = check
        return _FakeCompleted(0)

    monkeypatch.setattr("agent2model.cli.subprocess.run", _fake_run)
    result = runner.invoke(
        app,
        ["cloud", "run", str(path), "--size", "8b", "--epochs", "3"],
    )
    assert result.exit_code == 0, result.output
    argv = captured["argv"]
    assert argv[:4] == ["modal", "run", "-m", MODAL_RUN_TARGET]
    assert "--size" in argv and "8b" in argv
    assert "--epochs" in argv and "3" in argv


def test_cloud_run_propagates_modal_failure(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")

    class _Failed:
        returncode = 7

    monkeypatch.setattr("agent2model.cli.subprocess.run", lambda *a, **k: _Failed())
    result = runner.invoke(app, ["cloud", "run", str(path)])
    assert result.exit_code == 7


def test_cloud_run_missing_flowchart(runner: CliRunner, tmp_path: Path) -> None:
    ghost = tmp_path / "does_not_exist.yaml"
    result = runner.invoke(app, ["cloud", "run", str(ghost)])
    assert result.exit_code == 1


def test_cloud_run_handles_missing_modal_executable(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("modal")

    monkeypatch.setattr("agent2model.cli.subprocess.run", _boom)
    result = runner.invoke(app, ["cloud", "run", str(path)])
    assert result.exit_code == 1
    assert "modal" in result.output.lower() or "cloud" in result.output.lower()


# --------------------------------------------------------------------------- #
# --yes flag plumbing                                                          #
# --------------------------------------------------------------------------- #


def test_cloud_run_help_advertises_yes(runner: CliRunner) -> None:
    result = runner.invoke(app, ["cloud", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--yes" in result.output
    assert "--no-yes" in result.output


def test_build_modal_run_argv_with_yes(tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    argv = _build_modal_run_argv(
        path,
        name=None,
        size="3b",
        n=10,
        epochs=1,
        eval_n=10,
        base_model=None,
        skip_eval=False,
        serve_after=False,
        yes=True,
    )
    assert "--yes" in argv


def test_build_modal_run_argv_without_yes_omits_it(tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    argv = _build_modal_run_argv(
        path,
        name=None,
        size="3b",
        n=10,
        epochs=1,
        eval_n=10,
        base_model=None,
        skip_eval=False,
        serve_after=False,
        yes=False,
    )
    assert "--yes" not in argv


def test_cloud_run_dry_run_includes_yes_when_flag_set(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    result = runner.invoke(app, ["cloud", "run", str(path), "--yes", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "--yes" in result.output


# --------------------------------------------------------------------------- #
# cloud doctor                                                                 #
# --------------------------------------------------------------------------- #


def test_cloud_doctor_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["cloud", "doctor", "--help"])
    assert result.exit_code == 0, result.output
    assert "preflight" in result.output.lower() or "doctor" in result.output.lower()


def test_cloud_doctor_runs_and_renders_checks(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent2model.cloud import doctor as doctor_mod

    def _all_green() -> list[doctor_mod.CheckResult]:
        return [
            doctor_mod.CheckResult(name="modal install", ok=True, message="modal 1.4.3"),
            doctor_mod.CheckResult(name="modal token", ok=True, message="workspace: ws"),
            doctor_mod.CheckResult(name="anthropic secret", ok=True, message="resolved"),
            doctor_mod.CheckResult(
                name="local key", ok=True, severity="informational", message="skipped"
            ),
            doctor_mod.CheckResult(
                name="hf token", ok=True, severity="informational", message="skipped"
            ),
        ]

    monkeypatch.setattr("agent2model.cli.run_all_checks", _all_green, raising=False)
    # The doctor command imports run_all_checks inside its body; patch the
    # source module so the imported reference resolves to ours.
    monkeypatch.setattr(doctor_mod, "run_all_checks", _all_green)
    result = runner.invoke(app, ["cloud", "doctor"])
    assert result.exit_code == 0
    assert "modal install" in result.output
    assert "anthropic secret" in result.output


def test_cloud_doctor_exits_one_on_critical_failure(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent2model.cloud import doctor as doctor_mod

    def _critical_red() -> list[doctor_mod.CheckResult]:
        return [
            doctor_mod.CheckResult(name="modal install", ok=True),
            doctor_mod.CheckResult(
                name="modal token",
                ok=False,
                severity="critical",
                message="missing",
                fix_command="modal token new",
            ),
        ]

    monkeypatch.setattr(doctor_mod, "run_all_checks", _critical_red)
    result = runner.invoke(app, ["cloud", "doctor"])
    assert result.exit_code == 1
    assert "modal token new" in result.output


# --------------------------------------------------------------------------- #
# cloud setup                                                                  #
# --------------------------------------------------------------------------- #


def test_cloud_setup_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["cloud", "setup", "--help"])
    assert result.exit_code == 0, result.output
    assert "setup" in result.output.lower() or "wizard" in result.output.lower()


def test_cloud_setup_runs_to_completion(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent2model.cloud import doctor as doctor_mod
    from agent2model.cloud import setup as setup_mod_

    def _fake_run_setup(io: Any, **_kw: Any) -> list[setup_mod_.WizardResult]:
        return [
            setup_mod_.WizardResult(step="modal_account", outcome="already_done", message="ok"),
            setup_mod_.WizardResult(step="modal_token", outcome="already_done", message="present"),
            setup_mod_.WizardResult(
                step="anthropic_secret",
                outcome="completed",
                message="created",
            ),
        ]

    monkeypatch.setattr(setup_mod_, "run_setup", _fake_run_setup)
    monkeypatch.setattr(
        doctor_mod,
        "run_all_checks",
        lambda: [doctor_mod.CheckResult(name="all", ok=True, message="green")],
    )

    result = runner.invoke(app, ["cloud", "setup"], input="\n")
    assert result.exit_code == 0
    assert "anthropic_secret" in result.output
    assert "Ready" in result.output or "Setup" in result.output
