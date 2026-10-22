"""Unit tests for unified production CLI interface (src/iam/cli.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import scripts.run_dashboard as rd

from iam.cli import build_parser, main


class TestIAMCommandLineInterface:
    """Tests for the unified iam command-line dispatcher and subcommands."""

    def test_parser_construction(self) -> None:
        parser = build_parser()
        assert parser.prog == "iam"
        subcommands = [
            "generate",
            "train",
            "evaluate",
            "repair",
            "dashboard",
            "export",
            "benchmark",
        ]
        for sub in subcommands:
            assert sub in parser.format_help()

    def test_main_no_arguments_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "usage: iam" in captured.out or "usage: iam" in captured.err

    def test_cli_generate_command(self, tmp_path: Path) -> None:
        out_file = tmp_path / "test_graph.json"
        exit_code = main(
            ["generate", "--num-nodes", "60", "--seed", "42", "--output", str(out_file)]
        )
        assert exit_code == 0
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "num_nodes" in data
        assert data["num_nodes"] >= 40

    def test_cli_train_command(self) -> None:
        exit_code = main(
            ["train", "--epochs", "1", "--lr", "0.01", "--hidden-dim", "16", "--num-bases", "2"]
        )
        assert exit_code == 0

    def test_cli_repair_command(self, tmp_path: Path) -> None:
        out_patch = tmp_path / "test_patch.json"
        exit_code = main(["repair", "--num-nodes", "50", "--output-patch", str(out_patch)])
        assert exit_code == 0
        assert out_patch.exists()

        patch_data = json.loads(out_patch.read_text(encoding="utf-8"))
        assert "applied_transformations" in patch_data

    def test_cli_export_command(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "export_out"
        exit_code = main(
            ["export", "--num-nodes", "50", "--top-k", "2", "--output-dir", str(out_dir)]
        )
        assert exit_code == 0
        assert (out_dir / "triage_summary.md").exists()

    def test_cli_benchmark_fast_command(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "benchmarks_out"
        exit_code = main(["benchmark", "--fast", "--output-dir", str(out_dir)])
        assert exit_code == 0
        assert (out_dir / "benchmark_report.md").exists()

    def test_cli_dashboard_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called_cmd: list[list[str]] = []

        def mock_call(cmd: list[str]) -> int:
            called_cmd.append(cmd)
            return 0

        monkeypatch.setattr("subprocess.call", mock_call)
        exit_code = main(["dashboard", "--port", "8502"])
        assert exit_code == 0
        assert len(called_cmd) == 1
        assert "run_dashboard.py" in called_cmd[0][1]
        assert "--port" in called_cmd[0]
        assert "8502" in called_cmd[0]


class TestRunDashboardScript:
    """Tests for scripts/run_dashboard.py argument parsing and streamlit invocation."""

    def test_run_dashboard_translates_port_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        executed_cmds: list[list[str]] = []

        def mock_run(
            cmd: list[str], *_args: Any, **_kwargs: Any
        ) -> subprocess.CompletedProcess[bytes]:
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr(sys, "argv", ["run_dashboard.py", "--port", "8503"])

        with pytest.raises(SystemExit) as exc_info:
            rd.main()
        assert exc_info.value.code == 0
        assert len(executed_cmds) == 1
        cmd = executed_cmds[0]
        assert "--server.port" in cmd
        port_idx = cmd.index("--server.port")
        assert cmd[port_idx + 1] == "8503"
        assert "--port" not in cmd

    def test_run_dashboard_translates_host_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        executed_cmds: list[list[str]] = []

        def mock_run(
            cmd: list[str], *_args: Any, **_kwargs: Any
        ) -> subprocess.CompletedProcess[bytes]:
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr(sys, "argv", ["run_dashboard.py", "--host", "0.0.0.0"])

        with pytest.raises(SystemExit) as exc_info:
            rd.main()
        assert exc_info.value.code == 0
        assert len(executed_cmds) == 1
        cmd = executed_cmds[0]
        assert "--server.port" in cmd
        assert "--server.address" in cmd
        addr_idx = cmd.index("--server.address")
        assert cmd[addr_idx + 1] == "0.0.0.0"
