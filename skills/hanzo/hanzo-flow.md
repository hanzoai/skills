# Hanzo Flow - Visual AI Workflow Builder

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-studio.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Flow is a **visual workflow builder** for AI-powered agents and pipelines. It is a fork of [Langflow](https://github.com/langflow-ai/langflow) maintained at `github.com/hanzoai/flow`, providing a drag-and-drop visual interface for designing LLM chains, multi-agent orchestrations, and data processing pipelines. Every workflow can be deployed as a REST API or MCP server with zero additional code.

The project is a **Python-first** monorepo with a React/TypeScript frontend. It ships three packages: `hanzoflow` (the main application), `langflow-base` (the backend framework layer), and `lfx` (a lightweight CLI executor for running flows headlessly).

## Quick Reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/flow` |
| Upstream | `langflow-ai/langflow` (remote: `upstream`) |
| Language | Python 3.12+ (backend), TypeScript (frontend) |
| Package | `hanzoflow` on PyPI |
| CLI Executor | `lfx` on PyPI |
| Version | 1.8.0 (`hanzoflow`), 0.8.0 (`langflow-base`), 0.3.0 (`lfx`) |
| Default Port | 7860 (hanzoflow), 3006 (Docker), 8000 (lfx serve) |
| Docker Image | `hanzoai/flow:latest` |
| License | MIT |
| Python Tooling | `uv` (package manager), `ruff` (linter/formatter), `mypy` (type checker) |
| Frontend Tooling | Node.js 20+, npm, Biome |
| Testing | `pytest` (backend), `jest` + Playwright (frontend) |

## Install and Run

```bash
# Install from PyPI
uv pip install hanzoflow -U

# Run the visual builder
uv run hanzoflow run
# --> http://localhost:7860

# Docker (production)
docker run -p 7860:7860 hanzoai/flow:latest

# From source (development)
git clone ssh://github.com/hanzoai/flow
cd flow
make run_cli
```

## Project Structure

```
flow/
├── pyproject.toml              # Root workspace: hanzoflow v1.8.0
├── uv.lock                     # Locked dependencies
├── Dockerfile                  # Multi-stage production build (Python 3.12-slim)
├── Makefile                    # 40+ automation targets
├── Makefile.frontend           # Frontend build targets
├── src/
│   ├── backend/
│   │   ├── base/
│   │   │   ├── pyproject.toml  # langflow-base v0.8.0
│   │   │   ├── langflow/       # Core framework (API, services, graph engine)
│   │   │   │   ├── api/        # REST API (v1, v2)
│   │   │   │   ├── base/       # Base classes
│   │   │   │   ├── components/ # Built-in component definitions
│   │   │   │   ├── custom/     # Custom component loader
│   │   │   │   ├── graph/      # DAG execution engine
│   │   │   │   ├── interface/  # Type interfaces
│   │   │   │   ├── services/   # Service layer (storage only)
│   │   │   │   └── template/   # Component templates
│   │   │   └── hanzoflow/      # Hanzo-branded entry point
│   │   ├── src/                # Additional backend source
│   │   └── tests/              # Backend test suite
│   ├── lfx/                    # Hanzo Flow Executor (lightweight CLI)
│   │   ├── pyproject.toml      # lfx v0.3.0
│   │   ├── src/lfx/
│   │   │   ├── __main__.py     # CLI entry point
│   │   │   ├── cli/            # Typer CLI (serve, run commands)
│   │   │   ├── components/     # 109 component integrations
│   │   │   ├── custom/         # Custom component support
│   │   │   ├── graph/          # Graph execution engine
│   │   │   ├── events/         # Event system
│   │   │   ├── services/       # Pluggable services (auth, cache, chat, database,
│   │   │   │                   #   mcp_composer, session, settings, storage,
│   │   │   │                   #   telemetry, tracing, transaction, variable)
│   │   │   ├── inputs/         # Input type definitions
│   │   │   ├── io/             # I/O handling
│   │   │   ├── load/           # Flow file loaders
│   │   │   ├── log/            # Structured logging (loguru + structlog)
│   │   │   ├── memory/         # Conversation memory
│   │   │   ├── processing/     # Pipeline processing
│   │   │   ├── run/            # Flow execution runtime
│   │   │   ├── schema/         # Pydantic schemas
│   │   │   └── serialization/  # Flow import/export
│   │   └── tests/              # lfx test suite
│   └── frontend/               # React/TypeScript UI
│       ├── package.json        # langflow v1.8.0 (private)
│       ├── src/                # React components, state, API client
│       ├── tests/              # Jest unit + Playwright E2E tests
│       └── tailwind.config.mjs # Tailwind CSS config
├── docker/                     # Docker variants (dev, CDK, render, build-and-push)
├── deploy/                     # Docker Compose + Prometheus config
├── docs/                       # Documentation site
└── scripts/                    # Automation scripts
```

## Core Concepts

### Visual Flow Builder
The frontend provides a canvas where users drag components from a library, connect them via edges, and configure parameters visually. Flows are serialized as JSON DAGs with nodes (components) and edges (data connections).

### Component Library (109 Integrations)
The `lfx` package includes 109 component directories covering major AI/ML providers and tools:

- **LLM Providers**: OpenAI, Anthropic, Google, Azure, Cohere, NVIDIA, Groq, Fireworks, Hugging Face, Ollama, Mistral, Perplexity, AI/ML API
- **Vector Stores**: Chroma, Pinecone, Qdrant, Weaviate, Milvus, pgvector, Elasticsearch, MongoDB Atlas, Astra DB, ClickHouse, Couchbase, Cassandra
- **Data Sources**: URL, file, API, Google Drive, Confluence, Notion, GitHub, Git, S3, Slack, Discord, Telegram, WhatsApp
- **Agents**: LangChain agents, CrewAI, custom agent components
- **Tools**: Python code execution, SearXNG search, Composio, MCP tools, Tavily, SerpAPI, Wolfram Alpha
- **Processing**: Text splitters, embeddings, parsers, transformers, chains
- **Output**: Chat output, text output, webhook, email

### lfx CLI Executor
The `lfx` package is a standalone CLI for running flows without the web UI:

```bash
# Run a flow directly
uv run lfx run my_flow.json "What is AI?"

# Serve a flow as an API
export LANGFLOW_API_KEY=your-secret-key
uv run lfx serve my_flow.json --port 8000

# Programmatic usage
from lfx import components as cp
from lfx.graph import Graph

chat_input = cp.ChatInput()
agent = cp.AgentComponent()
chat_output = cp.ChatOutput()
graph = Graph(chat_input, chat_output)
```

### MCP Server Deployment
Flows can be exposed as MCP (Model Context Protocol) tools, making them callable by any MCP-compatible AI client. The `mcp_composer` service handles tool registration and request routing.

### Pluggable Service Architecture
lfx supports replacing built-in services with custom implementations. Services include auth, cache, chat history, database, MCP composer, session, settings, storage, telemetry, tracing, transaction, and variable management. Services are registered via config files, decorators, or Python entry points.

## Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| Framework | FastAPI (REST API v1 + v2) |
| Server | Uvicorn (dev), Gunicorn (production) |
| ORM | SQLModel (SQLAlchemy + Pydantic) |
| Graph Engine | Custom DAG executor with networkx |
| AI Framework | LangChain 0.3.x, LangChain Community, LangChain Experimental |
| Serialization | orjson, Pydantic 2.11 |
| Auth | PyJWT, passlib + bcrypt |
| Logging | loguru + structlog |
| Telemetry | OpenTelemetry (API, SDK, FastAPI instrumentation, Prometheus exporter) |
| Migrations | Alembic |
| Task Execution | multiprocess, asyncer |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 18+ |
| UI Libraries | Radix UI, Chakra UI, Headless UI |
| State | React hooks |
| Styling | Tailwind CSS |
| Canvas | React Flow (node-based graph editor) |
| Testing | Jest, Playwright |
| Linting | Biome |

### Key Dependencies (langflow-base)
- `langchain~=0.3.27`, `langchain-core>=0.3.81`, `langchain-community>=0.3.28`
- `fastapi>=0.115.2`, `uvicorn>=0.30.0`, `gunicorn>=22.0.0`
- `sqlmodel==0.0.22`, `pydantic~=2.11.0`, `alembic>=1.13.0`
- `mcp>=1.17.0` (Model Context Protocol)
- `sentry-sdk[fastapi,loguru]>=2.5.1`
- `opentelemetry-api>=1.25.0`, `prometheus-client>=0.20.0`
- `duckdb>=1.0.0`, `clickhouse-connect==0.7.19`
- `cryptography>=43.0.1`, `bcrypt==4.0.1`

### Key Dependencies (lfx)
- `langchain-core>=0.3.81`, `langchain~=0.3.23`
- `fastapi>=0.115.13`, `uvicorn>=0.34.3`
- `pydantic>=2.0.0`, `pydantic-settings>=2.10.1`
- `ag-ui-protocol>=0.1.10` (Agent UI Protocol)
- `httpx[http2]>=0.24.0`, `networkx>=3.4.2`
- `structlog>=25.4.0`, `loguru>=0.7.3`

## Observability

Hanzo Flow integrates with multiple observability backends:

- **LangSmith**: LangChain-native tracing and evaluation
- **LangFuse**: Open-source LLM observability (traces, scores, costs)
- **OpenTelemetry**: Distributed tracing via OTEL SDK
- **Prometheus**: Metrics exposure via FastAPI instrumentation
- **Sentry**: Error tracking and performance monitoring

## Docker

The production Dockerfile uses a multi-stage build:

1. **Builder stage**: Python 3.12-slim, installs `uv` from `ghcr.io/astral-sh/uv:latest`, copies workspace manifests, runs `uv sync --frozen --no-dev`
2. **Production stage**: Python 3.12-slim, copies `.venv` from builder, runs as non-root `hanzo` user, exposes port 3006, health check via `curl`

```bash
# Build
docker build -t hanzoai/flow:latest .

# Run
docker run -p 3006:3006 hanzoai/flow:latest
```

## Development

```bash
# Clone
git clone ssh://github.com/hanzoai/flow
cd flow

# Backend development
make install_backend   # Install Python deps with uv
make run_cli          # Start backend dev server

# Frontend development
make install_frontend  # npm install
make run_frontend     # Start frontend dev server

# Testing
make test             # Run all tests
make lint             # Ruff lint
make format           # Ruff format

# Workspace (uv)
uv sync --all-extras  # Install all workspace packages
uv run pytest -v      # Run tests
uv run ruff format .  # Format code
uv run mypy .         # Type check
```

## Related Skills

- `hanzo/hanzo-agent.md` -- Agent SDK for multi-agent orchestration
- `hanzo/hanzo-studio.md` -- Visual AI engine (ComfyUI fork, image/video generation)
- `hanzo/hanzo-console.md` -- Observability, traces, and cost tracking
- `hanzo/hanzo-mcp.md` -- MCP tools and protocol implementation
- `hanzo/hanzo-llm.md` -- LLM Gateway (upstream model routing)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: workflow, langflow, visual-builder, agents, mcp, pipelines
**Prerequisites**: Python 3.12+, uv, Node.js 20+ (for frontend development)
