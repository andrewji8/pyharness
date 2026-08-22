"""Web/API plugin for PyHarness (module C).

Serves the same agent engine over:
* REST ``POST /chat``   — one-shot turn (returns the full schema transcript),
* WebSocket ``/ws/chat`` — streaming deltas + tool round-trips in real time,
* ``GET /health``       — liveness + the configured provider models.

Like every PyHarness surface it is a **plugin/consumer**: it owns a ``Harness``
(auto-loading the ``builtin``/``llm`` plugins), never imports engine internals,
and keeps multi-turn state as reversible ``SessionContext`` snapshots in an
in-memory :class:`SessionStore` (single-process demo store).

Start with ``dsh-py serve`` (or ``python -m pyharness.plugins.web``).
"""

from pyharness.plugins.web.app import app

__all__ = ["app"]