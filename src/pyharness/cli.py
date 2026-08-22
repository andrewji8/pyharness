"""PyHarness top-level CLI entry point.

Exposes the ``pyharness`` command as defined in ``pyproject.toml``
``[project.scripts]``.

This thin wrapper delegates to the real Typer app implemented in
:mod:`pyharness.plugins.cli.app`.
"""

from pyharness.plugins.cli.app import app

__all__ = ["app"]
