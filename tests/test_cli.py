"""Tests for module B — the CLI plugin (`dsh-py`).

Uses ``typer.testing.CliRunner`` to invoke the real command tree in-process
(deterministic via the ``dummy`` provider). The interactive ``chat`` REPL cannot
be driven headlessly, so it's validated at the plumbing level (`--help`).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
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


def test_plugin_command_bypasses_system_proxy() -> None:
    """The CLI only talks to a local `pyharness serve`; it must build the
    httpx client with ``trust_env=False`` so the OS/system proxy is ignored."""
    fake_client = MagicMock()
    # Simulate a refused local connection so the command handles it gracefully.
    fake_client.__enter__.return_value.get.side_effect = httpx.ConnectError("refused")

    with patch("httpx.Client", return_value=fake_client) as mock_cls:
        result = runner.invoke(app, ["plugin", "list", "--url", "http://127.0.0.1:1"])

    # The client was constructed with trust_env=False (no system proxy).
    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs.get("trust_env") is False
    # Connection failure is surfaced as a non-zero exit, not a crash.
    assert result.exit_code != 0