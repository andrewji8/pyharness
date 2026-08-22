# PyHarness — 万物皆插件

[![PyPI version](https://badge.fury.io/py/pyharness-ai.svg)](https://pypi.org/project/pyharness-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![100 Tests Passed](https://img.shields.io/badge/tests-100%2F100-brightgreen.svg)](#)
[![MCP Support](https://img.shields.io/badge/MCP-supported-purple.svg)](https://modelcontextprotocol.io/)

> **万物皆插件。** 基于 `pluggy` + `Pydantic V2` + `asyncio` 构建的生产级 Python Agent 框架。

[English](./README.md) | [中文](./README_zh.md)

---

## 🚀 核心特性

| 特性 | 说明 |
|------|------|
| 🔌 **插件优先架构** | LLM 提供商、工具执行器、CLI、Web UI 全部通过 `@hookimpl` 注册。核心引擎不到 50 行代码。 |
| ⚡ **高级编排能力** | 拓扑 DAG 计划 (`WorkflowPlan`) + 基于 `asyncio.TaskGroup` 的并行子 Agent 执行。 |
| 🔗 **原生 MCP 协议支持** | 零代码接入全球 MCP 工具生态，同时支持 `stdio`（子进程）和 `SSE`（HTTP）传输。 |
| 🔍 **混合 RAG 检索** | 本地 `numpy` 向量库 + SQLite FTS5 关键词搜索，通过 Reciprocal Rank Fusion (RRF) 融合排序。 |
| 🛡️ **企业级安全** | 异步人工审批门控 (`Guard`) + 进程级 Python 代码沙盒。 |
| 🌐 **实时 Web UI** | FastAPI + WebSocket，三栏布局实时展示思考流与 Plan 进度。 |

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              外部生态层                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ MCP Servers  │  │  OpenAI API  │  │  File System │  │  Web / GitHub │  │
│  │ (stdio/SSE)  │  │  DeepSeek    │  │  SQLite DB   │  │  Vector Store │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
└─────────┼────────────────┼────────────────┼────────────────┼──────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           插件层 (pluggy)                                    │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ LLM 插件   │ │ 工具插件    │ │ Guard 插件 │ │ MCP 插件   │             │
│  │ (OpenAI,   │ │ (Python,    │ │ (异步审批  │ │ (stdio/SSE │             │
│  │  Dummy)    │ │  FS, Web)  │ │  门控)     │ │  传输)     │             │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────────────────┐ │
│  │ RAG 插件   │ │ Workflow   │ │ 会话存储    │ │      Web UI 插件        │ │
│  │ (Embedding,│ │ 插件       │ │ (FTS5)     │ │  (FastAPI + WebSocket)  │ │
│  │  Vector)   │ │ (Plan DAG) │ │            │ │                        │ │
│  └────────────┘ └────────────┘ └────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           引擎层 (Harness)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  EventBus (pluggy) → 插件发现 → Hook 分发                               │ │
│  │  SessionContext (contextvars) → 异步隔离 → 状态快照                      │ │
│  │  run_session() → Agent Loop → observe_event() → 插件链式执行             │ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 快速上手

### 安装

```bash
# 完整安装（含所有可选依赖）
pip install pyharness-ai[all]

# 最小安装
pip install pyharness-ai
```

### 配置

创建 `.env` 文件或设置环境变量：

```bash
# OpenAI 兼容 API（支持 DeepSeek、OpenAI、vLLM 等）
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.deepseek.com/v1"

# 可选：MCP Server 配置
# 参见 mcp_servers.json 了解 stdio/SSE 传输设置
```

### CLI 使用

```bash
# 单次运行
pyharness run --provider http -m deepseek-chat -i "解释量子计算"

# 流式 REPL 对话
pyharness chat --provider http -m deepseek-chat

# 启动 Web UI
pyharness serve
# → 打开 http://127.0.0.1:3080

# 查看版本
pyharness version
```

### Python API

```python
import asyncio
from pyharness import Harness
from pyharness.schema import AgentConfig

async def main():
    harness = Harness()  # 自动加载内置插件
    agent = AgentConfig(
        name="my-agent",
        model="deepseek-chat",
        system_prompt="你是一个有用的助手。",
    )
    await harness.run_session(agent, initial_text="你好，PyHarness！")
    # 访问对话历史
    ctx = harness.last_context
    for msg in ctx.messages:
        print(f"{msg.role}: {msg.content}")

asyncio.run(main())
```

---

## 📦 内置插件矩阵

| 插件 | 入口点 | 职责 |
|------|--------|------|
| `builtin` | `pyharness.plugins.builtin` | 核心生命周期钩子、事件总线 wiring |
| `llm` | `pyharness.plugins.llm.entry` | LLM 提供商注册表（OpenAI、Dummy） |
| `cli` | `pyharness.plugins.cli` | 终端 REPL（`pyharness chat`） |
| `web-ui` | `pyharness.plugins.web_ui` | FastAPI + WebSocket 仪表盘 |
| `tool-python` | `pyharness.plugins.tool_python_exec` | 沙盒化 Python 代码执行 |
| `tool-fs` | `pyharness.plugins.tool_fs` | 文件系统操作（读/写/列表） |
| `tool-web` | `pyharness.plugins.tool_web` | 网页抓取（HTTP + HTML 解析） |
| `session-store` | `pyharness.plugins.session_store` | SQLite + FTS5 会话持久化 |
| `subagent` | `pyharness.plugins.tool_subagent` | 并行子 Agent 生成（TaskGroup） |
| `workflow` | `pyharness.plugins.workflow` | Plan 驱动的 DAG 编排 |
| `guard` | `pyharness.plugins.guard_approval` | Human-in-the-loop 异步审批 |
| `context-compaction` | `pyharness.plugins.context_compaction` | 自动上下文窗口管理 |
| `mcp-client` | `pyharness.plugins.mcp_client` | MCP Server 连接（stdio/SSE） |
| `embedding` | `pyharness.plugins.embedding` | 文本嵌入（OpenAI + Dummy） |
| `vector-store` | `pyharness.plugins.vector_store` | 本地 numpy 向量数据库 |
| `tool-knowledge` | `pyharness.plugins.tool_knowledge` | 知识搜索 + 目录摄入 |
| `tool-hybrid-search` | `pyharness.plugins.tool_hybrid_search` | RRF 融合 FTS5 + 向量搜索 |

---

## 🗺️ 未来路线图

- [ ] 热插拔插件（无需重启即可加载/卸载）
- [ ] Docker 代码执行沙箱
- [ ] 多模态支持（视觉、音频）
- [ ] 分布式 Agent 集群（Redis / Ray）
- [ ] MCP 工具流式 SSE 输出
- [ ] 内置评估框架（LLM-as-judge）

---

## 🤝 贡献指南

欢迎贡献！请：

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feat/amazing-feature`）
3. 运行测试（`pytest tests/`）
4. 提交 PR

## 📄 协议

MIT — 详见 [LICENSE](LICENSE) 文件。

---

<p align="center">
  由 <a href="https://github.com/pyharness">PyHarness</a>  contributors 用 ❤️ 构建
</p>
