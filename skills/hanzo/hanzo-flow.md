# Hanzo Flow - Visual AI Workflow Builder

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-agent.md`, `hanzo/hanzo-studio.md`

## Overview

Hanzo Flow is a **visual workflow builder** for LLM chains, agent pipelines, and AI workflows. Fork of **Langflow**, providing drag-and-drop pipeline design with MCP server deployment capabilities.

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/flow` |
| Language | Python (NOT TypeScript) |
| Package | `hanzoflow` (PyPI) |
| Python | >=3.12 |
| Port | 7860 |
| Image | `hanzoai/flow:latest` |

## Install and Run

```bash
# Install
uv pip install hanzoflow -U

# Run
uv run hanzoflow run
# → http://localhost:7860

# Docker
docker run -p 7860:7860 hanzoai/flow:latest
```

## Features

- **Visual builder**: Drag-and-drop LLM pipeline design
- **MCP server deployment**: Expose flows as MCP tools
- **LangSmith/LangFuse observability**: Trace and debug pipelines
- **Component library**: Pre-built LLM, embedding, vector store, and agent components
- **Python-native**: Custom components in Python

## Observability Integration

Supports LangSmith and LangFuse for trace visualization and debugging.

## Related Skills

- `hanzo/hanzo-console.md` — Observability and cost tracking
- `hanzo/hanzo-agent.md` — Agent SDK (produces traces)
- `hanzo/hanzo-studio.md` — Visual AI engine (ComfyUI fork, different purpose)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: workflow, langflow, visual, debugging
**Prerequisites**: Python, LLM chain concepts
