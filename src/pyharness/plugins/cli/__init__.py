"""PyHarness CLI plugin (`dsh-py`).

A Typer + Rich interface that consumes the public Harness API. It is itself a
plugin: when ``--verbose`` it registers a ``CliObserver`` ``@hookimpl`` into the
harness it owns, rendering the event stream to the terminal.
"""

from pyharness.plugins.cli.app import app, main

__all__ = ["app", "main"]