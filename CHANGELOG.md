# Changelog

All notable changes to PyHarness are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.8.0] — 2026-08-28

### Added
- **Docker sandbox (fail-closed)** — `python_exec` can run inside a
  `--rm --network none --memory 256m --cpus 0.5 --pids-limit 64 --read-only`
  `python:3.11-slim` container. On any misconfiguration or missing image the
  executor fails **closed** (refuses to run) instead of falling back to host
  execution. `pyharness sandbox init` pulls the image.
- **Built-in evaluation framework (`pyharness eval`)** — YAML task suites,
  programmatic checks (`contains`, `contains_any`, `tool_called`,
  `tool_not_called`, `safety_pass`), LLM-as-judge scoring (correctness /
  tool_usage / safety), and JSON + Markdown reports with per-run diff
  (↑/↓/=) and total-score delta.
- **Hot-swappable plugins** — `pyharness plugin load|reload|unload <path>`
  plus `GET /api/plugins`, so tools can be added/updated/removed at runtime
  without restarting the server.
- **Session Management UI** — list / inspect / resume / delete persisted
  sessions from the Web UI.

### Changed
- **Core single-loop refactor** — the agent loop now runs as one unified
  `run_session` coroutine (no nested loops), improving async isolation and
  making sub-agent fan-out deterministic.
- `build_harness()` factory is the single source of truth for wiring the
  full plugin set + LLM provider; the Web UI, CLI, and eval runner all use it.
- `pyharness eval` accepts `--model` / `--judge` so the report reflects the
  real model under test; eval runs with `auto_approve=True` to avoid blocking
  on interactive confirmation.

### Security
- **SSRF protection** — `tool-web` now validates host resolution and blocks
  requests to link-local / loopback / private ranges unless explicitly allowed.
- **WebSocket authorization** — the Web UI WS endpoint requires a shared token
  (`PYHARNESS_WS_TOKEN`) and rejects unauthenticated upgrades.
- **Unicast fan-out** — observer events are delivered only to subscribed
  clients (no cross-session broadcast).
- **Backpressure** — outbound event/topic queues are bounded; fast producers
  cannot exhaust memory when a slow client falls behind.
- **Human-in-the-loop Guard** — high-risk tools (`python_exec`, `shell_exec`,
  `fs_write`, `fs_delete`) require confirmation through the `ask_user_confirmation`
  hook.

---

## [0.7.0]
- Previous stable release. See git history for details.
