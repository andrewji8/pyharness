"""Run the Web/API plugin: python -m pyharness.plugins.web [--host --port]."""

from __future__ import annotations

import typer  # reuse typer for trivial arg parsing (already a dependency)
import uvicorn


def main(host: str = "127.0.0.1", port: int = 8070, reload: bool = False) -> None:
    """Launch the FastAPI + WebSocket server."""
    uvicorn.run("pyharness.plugins.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    typer.run(main)