# PyHarness — Everything is a Plugin

[![PyPI version](https://badge.fury.io/py/pyharness-ai.svg)](https://pypi.org/project/pyharness-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![100 Tests Passed](https://img.shields.io/badge/tests-100%2F100-brightgreen.svg)](#)
[![MCP Support](https://img.shields.io/badge/MCP-supported-purple.svg)](https://modelcontextprotocol.io/)

> **Everything is a Plugin.** A thin-core, production-grade Python Agent Framework built on `pluggy`, `Pydantic V2`, and `asyncio`.

[English](./README.md) | [中文](./README_zh.md)

---

## 🚀 Core Features

| Feature | Description |
|---------|-------------|
| 🔌 **Plugin-First Architecture** | LLM providers, tool executors, CLI, and Web UI are all plugins registered via `@hookimpl`. The core is under 50 lines. |
| ⚡ **Advanced Orchestration** | Topological DAG planning (`WorkflowPlan`) + parallel sub-agent execution via `asyncio.TaskGroup`. |
| 🔗 **MCP Protocol Native** | Zero-code integration with the global MCP ecosystem. Supports both `stdio` (subprocess) and `SSE` (HTTP) transports. |
| 🔍 **Hybrid RAG** | Local `numpy` vector store + SQLite FTS5 keyword search, fused via Reciprocal Rank Fusion (RRF). |
| 🛡️ **Enterprise Security** | Human-in-the-loop async approval gates (`Guard`) + process-level Python sandbox. |
| 🌐 **Real-time Web UI** | FastAPI + WebSocket with a three-panel layout for thought streaming and plan progress. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              External Ecosystem                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ MCP Servers  │  │  OpenAI API  │  │  File System │  │  Web / GitHub │  │
│  │ (stdio/SSE)  │  │  DeepSeek    │  │  SQLite DB   │  │  Vector Store │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
└─────────┼────────────────┼────────────────┼────────────────┼──────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Plugin Layer (pluggy)                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ LLM Plugin │ │ Tool Plugins│ │ Guard Plugin│ │  MCP Plugin│             │
│  │ (OpenAI,   │ │ (Python,    │ │ (Approval  │ │ (stdio/SSE │             │
│  │  Dummy)    │ │  FS, Web)  │ │  Gates)    │ │  Transport)│             │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────────────────┐ │
│  │ RAG Plugin │ │  Workflow  │ │ Session    │ │      Web UI Plugin      │ │
│  │ (Embedding,│ │  Plugin    │ │  Store     │ │  (FastAPI + WebSocket)  │ │
│  │  Vector)   │ │ (Plan DAG) │ │ (FTS5)     │ │                        │ │
│  └────────────┘ └────────────┘ └────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Engine Layer (Harness)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  EventBus (pluggy) → Plugin Discovery → Hook Dispatch                  │ │
│  │  SessionContext (contextvars) → Async Isolation → State Snapshots      │ │
│  │  run_session() → Agent Loop → observe_event() → Plugin Chain          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Installation

```bash
# Full installation (all optional dependencies)
pip install pyharness-ai[all]

# Minimal installation
pip install pyharness-ai
```

### Configuration

Create a `.env` file or set environment variables:

```bash
# OpenAI-compatible API (works with DeepSeek, OpenAI, vLLM, etc.)
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.deepseek.com/v1"

# Optional: MCP Server configuration
# See mcp_servers.json for stdio/SSE transport setup
```

### CLI Usage

```bash
# One-shot run
pyharness run --provider http -m deepseek-chat -i "Explain quantum computing"

# Streaming REPL chat
pyharness chat --provider http -m deepseek-chat

# Start Web UI
pyharness serve
# → Open http://127.0.0.1:3080

# Check version
pyharness version
```

### Python API

```python
import asyncio
from pyharness import Harness
from pyharness.schema import AgentConfig

async def main():
    harness = Harness()  # Auto-loads builtin plugins
    agent = AgentConfig(
        name="my-agent",
        model="deepseek-chat",
        system_prompt="You are a helpful assistant.",
    )
    await harness.run_session(agent, initial_text="Hello, PyHarness!")
    # Access conversation history
    ctx = harness.last_context
    for msg in ctx.messages:
        print(f"{msg.role}: {msg.content}")

asyncio.run(main())
```

---

## 📦 Built-in Plugins

| Plugin | Entry Point | Responsibility |
|--------|-------------|----------------|
| `builtin` | `pyharness.plugins.builtin` | Core lifecycle hooks, event bus wiring |
| `llm` | `pyharness.plugins.llm.entry` | LLM provider registry (OpenAI, Dummy) |
| `cli` | `pyharness.plugins.cli` | Terminal REPL (`pyharness chat`) |
| `web-ui` | `pyharness.plugins.web_ui` | FastAPI + WebSocket dashboard |
| `tool-python` | `pyharness.plugins.tool_python_exec` | Sandboxed Python code execution |
| `tool-fs` | `pyharness.plugins.tool_fs` | File system operations (read/write/list) |
| `tool-web` | `pyharness.plugins.tool_web` | Web scraping (HTTP + HTML parsing) |
| `session-store` | `pyharness.plugins.session_store` | SQLite + FTS5 conversation persistence |
| `subagent` | `pyharness.plugins.tool_subagent` | Parallel sub-agent spawning (TaskGroup) |
| `workflow` | `pyharness.plugins.workflow` | Plan-based DAG orchestration |
| `guard` | `pyharness.plugins.guard_approval` | Human-in-the-loop async approval |
| `context-compaction` | `pyharness.plugins.context_compaction` | Automatic context window management |
| `mcp-client` | `pyharness.plugins.mcp_client` | MCP Server connectivity (stdio/SSE) |
| `embedding` | `pyharness.plugins.embedding` | Text embedding (OpenAI + Dummy) |
| `vector-store` | `pyharness.plugins.vector_store` | Local numpy-based vector database |
| `tool-knowledge` | `pyharness.plugins.tool_knowledge` | Knowledge search + directory ingestion |
| `tool-hybrid-search` | `pyharness.plugins.tool_hybrid_search` | RRF fusion of FTS5 + vector search |

---

## 🛡️ Security Model

PyHarness is built to run untrusted agent workloads without exposing the host.

### Code Sandbox (fail-closed)
`python_exec` runs in a locked-down Docker container by default:

```bash
# Pull the sandbox image once
pyharness sandbox init

# Run with the Docker sandbox
PYHARNESS_SANDBOX=docker pyharness run -m deepseek-chat -i "sort a 10GB list"
```

Container hardening (`python:3.11-slim`):

| Flag | Effect |
|------|--------|
| `--network none` | No outbound network — SSRF/ exfiltration surface removed |
| `--read-only` + `--tmpfs /tmp` | Immutable root FS |
| `--memory 256m --memory-swap 256m` | Hard memory cap |
| `--cpus 0.5 --pids-limit 64` | CPU & process caps |
| `--rm` | Container destroyed after each run |

If Docker is unavailable or the image is missing, the executor **fails closed**
(refuses to execute) rather than silently running on the host.

### SSRF Protection
`tool-web` resolves the target host and rejects requests to loopback
(`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`), and private ranges
unless an allow-list is configured. This blocks agent-initiated probing of
internal services / cloud metadata endpoints.

### WebSocket Authorization
The Web UI WebSocket endpoint requires a shared token:

```bash
export PYHARNESS_WS_TOKEN=$(python -c "import secrets;print(secrets.token_hex(16))")
pyharness serve
```

Unauthenticated upgrade attempts are rejected. Events are delivered **unicast**
(per subscribed session) — one client never sees another session's traffic.

### Backpressure
Outbound event/topic queues are bounded. If a client falls behind, the server
drops the slowest frames rather than growing unbounded memory.

### Human-in-the-Loop Guard
High-risk tools (`python_exec`, `shell_exec`, `fs_write`, `fs_delete`) require
interactive confirmation via the `ask_user_confirmation` hook before execution.

---

## 🧪 Evaluation Framework (`pyharness eval`)

Score your agents the same way you ship them — the eval runner builds its
harness through the same `build_harness()` factory.

```bash
# Run the built-in suite with the dummy provider (no network)
pyharness eval --suite basic --model dummy --no-judge

# Run with a real model + LLM-as-judge
pyharness eval --suite basic --model nvidia/nemotron-3-ultra-550b-a55b:free \
               --judge nvidia/nemotron-3-ultra-550b-a55b:free
```

- Reports are written to `eval_reports/` (JSON + Markdown) and show a diff
  against the previous run (per-task ↑/↓/= and total-score delta).
- A task passes only when its **programmatic checks** pass **and** the judge
  returns valid scores.
- Eval runs use `auto_approve=True` so batch runs are never blocked by the
  interactive Guard.

Suite files are plain YAML (`evals/basic.yaml`):

```yaml
tasks:
  - id: code_prime
    category: code_exec
    prompt: "Return the 100th prime."
    checks:
      - type: contains
        value: "541"
      - type: tool_called
        tool: python_exec
```

---

## 🔌 Hot-Swappable Plugins

Add, update, or remove tools at runtime — no restart required.

```bash
# List currently loaded plugins
curl -s http://127.0.0.1:3080/api/plugins

# Load a new tool plugin
pyharness plugin load ./demo_tool.py

# Edit demo_tool.py (e.g. change return "v1" -> "v2") then hot-reload
pyharness plugin reload ./demo_tool.py

# Remove it again
pyharness plugin unload ./demo_tool.py
```

A minimal hot-loadable tool plugin:

```python
# demo_tool.py
from pyharness.plugins.tool_python_exec import ToolResult, ToolResultStatus
from pyharness.schema import ToolArg, ToolSpec
from pluggy import HookimplMarker

hookimpl = HookimplMarker("pyharness")

class DemoTool:
    @hookimpl
    def get_tool_specs(self, context):
        return (ToolSpec(
            name="say_hi",
            description="Echo a greeting.",
            parameters=(ToolArg(name="name", type="string", required=True),),
        ),)

    @hookimpl
    async def execute_tool(self, context, tool, arguments):
        if tool.name != "say_hi":
            return None
        return ToolResult(tool_name="say_hi",
                          status=ToolResultStatus.OK,
                          output={"greeting": f"v1 hi {arguments.get('name')}"})
```

Re-running `plugin reload` after editing the file picks up the new code
immediately.

---

## 🗺️ Roadmap

- [x] Hot-swappable plugins (load/unload without restart)
- [x] Docker-based sandbox for code execution
- [ ] Multimodal support (vision, audio)
- [ ] Distributed agent clusters (Redis/Ray)
- [ ] Streaming SSE for MCP tools
- [x] Built-in evaluation harness (LLM-as-judge)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Submit a PR

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/pyharness">PyHarness</a> contributors
</p>
