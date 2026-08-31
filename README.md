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

## 🌐 Distributed Deployment

PyHarness ships a distributed agent cluster built on Redis. Tool execution is
offloaded from the web node to independent `pyharness worker` processes via a
Redis task queue, with events fanned out over a Redis Pub/Sub event bus and
sessions shared through a Redis store. If Redis is unavailable, PyHarness
degrades gracefully to single-node in-process execution.

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│         Web Node(s)          │        │       Worker Node(s)         │
│  ┌────────────────────────┐  │        │  ┌────────────────────────┐  │
│  │  Harness (agent loop)  │  │        │  │ process_one() / loop   │  │
│  │  _exec_tool()          │  │        │  │  execute_tool() hook   │  │
│  └───────────┬────────────┘  │        │  └───────────┬────────────┘  │
└──────────────┼───────────────┘        └──────────────┼───────────────┘
               │                                       │
   enqueue (LPUSH) / await (BRPOP)       consume (BRPOP) → result (LPUSH)
               │                                       │
               └───────────────┬───────────────────────┘
                               ▼
              ┌──────────────────────────────────────────┐
              │                  Redis                   │
              │  task queue : pyharness:tool_tasks       │
              │  result key : pyharness:tool_result:<id> │
              │  event bus  : pyharness:events (Pub/Sub) │
              │  session    : RedisSessionStorePlugin    │
              └──────────────────────────────────────────┘
```

### Configuration

Set these in `.env` (see `.env.example`) or the environment:

```bash
# Enable distributed tool execution (web node + worker)
PYHARNESS_DISTRIBUTED_EXEC="1"

# Redis connection URL
REDIS_URL="redis://127.0.0.1:6379/0"
```

### Quick start

```bash
# 1. Start Redis (or use a hosted instance) and export REDIS_URL

# 2. Start one or more workers — each consumes tool tasks off the queue
pyharness worker

# 3. Run the web node with distributed execution enabled
pyharness serve
# → Tools now execute on the worker process(es); stream events flow back live
```

Without `REDIS_URL` or with `PYHARNESS_DISTRIBUTED_EXEC` unset, everything runs
in-process as before — no config change is needed for single-node use.

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

## 🗺️ Roadmap

- [ ] Hot-swappable plugins (load/unload without restart)
- [ ] Docker-based sandbox for code execution
- [ ] Multimodal support (vision, audio)
- [ ] Distributed agent clusters (Redis/Ray)
- [ ] Streaming SSE for MCP tools
- [ ] Built-in evaluation harness (LLM-as-judge)

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
