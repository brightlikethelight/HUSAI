from __future__ import annotations

import sys
from pathlib import Path

from scripts.experiments.run_official_external_benchmarks import run_command


def test_run_command_executes_without_shell(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    result = run_command(
        name="echo_ok",
        command=f"{sys.executable} -c \"print('ok')\"",
        cwd=tmp_path,
        logs_dir=logs_dir,
        execute=True,
    )

    assert result.attempted is True
    assert result.success is True
    assert result.returncode == 0
    assert result.stdout_log is not None

    stdout_path = Path(result.stdout_log)
    if not stdout_path.is_absolute():
        stdout_path = Path(__file__).resolve().parents[2] / stdout_path
    assert "ok" in stdout_path.read_text()


def test_run_command_rejects_shell_operators(tmp_path: Path) -> None:
    result = run_command(
        name="unsafe",
        command="echo ok | cat",
        cwd=tmp_path,
        logs_dir=tmp_path / "logs",
        execute=True,
    )

    assert result.attempted is False
    assert result.success is False
    assert "shell operators" in result.note
