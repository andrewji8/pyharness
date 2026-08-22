"""Tests for module B — the CLI plugin (`dsh-py`).

Uses ``typer.testing.CliRunner`` to invoke the real command tree in-process
(deterministic via the ``dummy`` provider). The interactive ``chat`` REPL cannot
be driven headlessly, so it's validated at the plumbing level (`--help`).
"""

from __future__ import annotations

from typer.testing import CliRunner

from pyharness.plugins.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("PyHarness")


def test_run_dummy_single_turn() -> None:
    result = runner.invoke(
        app,
        ["run", "--provider", "dummy", "--model", "dummy", "--initial", "hello there"],
    )
    assert result.exit_code == 0, result.output
    # the user turn and the (echoed) reply both appear in the transcript
    assert result.output.count("hello there") >= 2  # user line + rendered reply
    assert "you" in result.output
    assert "Session" in result.output  # rich panel title


def test_run_unknown_provider_is_usage_error() -> None:
    result = runner.invoke(app, ["run", "--provider", "bogus", "--model", "dummy"])
    assert result.exit_code != 0
    assert "unknown provider" in result.output.lower()


def test_chat_command_plumbing() -> None:
    # definition is valid and documented; the REPL itself needs a tty.
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0, result.output
    assert "--model" in result.output