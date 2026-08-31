# Changelog

All notable changes to PyHarness are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-31

First stable release. Six pillar capabilities are now complete and covered by an
expanded test suite (214 passed, 4 skipped).

### 🐳 Docker Sandbox
- Sandboxed code execution tool (`tool_python_exec`) with process-level isolation and security guards.
- Security-focused execution tests for untrusted code paths.

### 📊 Evaluation Framework
- Built-in evaluation harness (`eval_runner`) with LLM-as-judge grading.
- `pyharness eval` CLI entry point and YAML-driven eval scenarios (`evals/`).
- Report generation into `eval_reports/`.

### 🔌 Hot-Swappable Plugins
- Load/unload plugins at runtime without restart via the pluggy-based plugin registry.
- Plugin hot-reload tests (`test_plugin_hotload.py`).

### 🖼️ Multimodal
- Multimodal support for vision and audio inputs.
- Audio tool plugin (`tool_audio`) plus multimodal test coverage.

### 🔗 MCP Streaming
- Streaming SSE support for MCP tools alongside the existing stdio transport.
- Test coverage for streaming tool events.

### 🌐 Distributed Agent Cluster (Redis)
- **Session store**: `RedisSessionStorePlugin` for shared conversation persistence (`session_store_redis`).
- **Event bus**: `RedisEventBusPlugin` Pub/Sub fan-out over the `pyharness:events` channel (`event_bus_redis`).
- **Task queue**: distributed tool execution via Redis List (LPUSH/BRPOP) with
  deterministic one-shot `process_one()` and an infinite `run_worker()` loop (`distributed.py`).
- **Worker process**: `pyharness worker` CLI command (`worker.py`) that executes
  tools off the queue and streams `tool.stream` events back to the web node.
- Graceful degradation to in-process execution when `REDIS_URL` or
  `PYHARNESS_DISTRIBUTED_EXEC` is absent.

### 🔧 Other
- Kernel/engine hardening and plugin routing via `factory.py`.
- Expanded security and streaming test suites.

## [0.7.0] - 2026-08-31

- Tavily search integration.
- Workflow execution fixes.
- Live PLAN panel in the Web UI.

