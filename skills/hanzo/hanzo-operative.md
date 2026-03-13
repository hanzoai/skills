# Hanzo Operative - Computer Use Agent

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Operative (`hanzo-operative`) is an **autonomous computer use agent** that enables AI to control desktop environments via screenshots, mouse, keyboard, and bash commands. It uses Anthropic's Claude API with computer use tool definitions to analyze screenshots and execute actions. The entry point is a Streamlit web UI. Fork of Anthropic's computer-use demo.

### What it actually is

- Python package `hanzo-operative` v0.1.1
- **Requires Python >= 3.14**
- Streamlit UI (`operative/operative.py`) as the primary entry point
- Tool-based architecture in `operative/tools/`:
  - `computer.py` -- Screenshot, mouse, keyboard via xdotool/scrot
  - `bash.py` -- Shell command execution
  - `edit.py` -- File editing
  - `base.py` -- Base tool class
  - `collection.py` -- Tool collection registry
  - `groups.py` -- Tool grouping
  - `run.py` -- Tool execution runner
- Agent loop (`operative/loop.py`) -- orchestrates Claude API calls with tool results
- System prompt (`operative/prompt.py`)
- Three Dockerfiles in `docker/`:
  - `Dockerfile` -- Main image (builds on xvfb image)
  - `Dockerfile.desktop` -- Desktop environment image
  - `Dockerfile.xvfb` -- Xvfb virtual display image
- Docker images: `ghcr.io/hanzoai/operative:latest`, `ghcr.io/hanzoai/xvfb:latest`, `ghcr.io/hanzoai/desktop:latest`
- No compose.yml in the repo

### What it is NOT

- No programmatic Python API like `from operative import Operative` with `op.click()`, `op.type()` etc.
- No `op.execute_task()` high-level task runner
- Tools are invoked by Claude via the Anthropic API tool use protocol, not called directly by user code
- VNC port 6080 is exposed in the Makefile's docker run commands

## When to use

- Automating desktop workflows via AI vision
- AI-driven UI testing across any desktop application
- Screen scraping with AI understanding
- RPA (Robotic Process Automation) with Claude reasoning
- Remote desktop automation via VNC

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/operative` |
| Package | `hanzo-operative` (PyPI) |
| Version | 0.1.1 |
| Upstream | Anthropic computer-use demo |
| Python | >= 3.14 |
| UI | Streamlit (port 8501) |
| VNC | Port 5900 (native), 6080 (noVNC web) |
| Docker images | `ghcr.io/hanzoai/operative`, `ghcr.io/hanzoai/xvfb`, `ghcr.io/hanzoai/desktop` |
| Dev (native) | `make run` (runs `uv run -- python3 -m streamlit run operative/operative.py`) |
| Dev (Docker) | `make dev` |
| Setup | `make setup` (creates venv with uv, Python 3.14) |
| Test | `make test` (pytest) |
| Test + coverage | `make test-cov` |
| Lint | `make lint` (ruff) |
| Format | `make format` (ruff) |
| License | MIT |

## Project structure

```
hanzoai/operative/
  pyproject.toml          # Package config (hanzo-operative 0.1.1, python >= 3.14)
  requirements.txt        # Pinned deps
  Makefile                # Build, run, test, Docker targets
  operative/
    __init__.py
    operative.py          # Streamlit UI entry point
    loop.py               # Agent loop (Claude API + tool orchestration)
    prompt.py             # System prompt
    requirements.txt      # Operative-specific deps
    tools/
      __init__.py         # Tool exports
      base.py             # Base tool class
      computer.py         # Computer use (screenshot, mouse, keyboard)
      bash.py             # Bash command execution
      edit.py             # File editing tool
      collection.py       # Tool collection registry
      groups.py           # Tool grouping definitions
      run.py              # Tool execution runner
  docker/
    Dockerfile            # Main operative image
    Dockerfile.desktop    # Desktop environment image
    Dockerfile.xvfb       # Xvfb virtual display base image
    image/                # Docker image assets
  tests/                  # Test suite
  docs/                   # Documentation
```

## Dependencies

From `pyproject.toml`:
- `anthropic >= 0.84.0` -- Claude API client
- `streamlit >= 1.43.0` -- Web UI framework
- `httpx >= 0.28.0` -- HTTP client
- `pillow >= 12.1.1` -- Image processing
- `certifi >= 2025.1.31` -- TLS certificates

Dev/test extras: pytest, pytest-cov, pytest-mock, pytest-asyncio, ruff, black

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/hanzoai/operative.git
cd operative

# Run with Docker (includes Xvfb, xdotool, VNC, Streamlit)
make dev
# Requires: ANTHROPIC_API_KEY env var

# Access Streamlit UI
open http://localhost:8501

# Access VNC (view desktop)
open http://localhost:6080    # noVNC web client
# or connect VNC client to localhost:5900
```

### Native (requires Python 3.14)

```bash
git clone https://github.com/hanzoai/operative.git
cd operative

# Setup with uv (creates .venv with Python 3.14)
make setup

# Install system deps (Ubuntu/Debian)
sudo apt install xdotool xvfb scrot

# Start Xvfb if headless
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Run
make run
# Equivalent to: uv run -- python3 -m streamlit run operative/operative.py
```

## Docker ports

The `make dev` and `make run-docker` commands expose:

| Port | Service |
|------|---------|
| 5900 | VNC (native) |
| 6080 | noVNC (web) |
| 8080 | HTTP |
| 8501 | Streamlit UI |

## How it works

1. User enters a task in the Streamlit UI
2. The agent loop (`loop.py`) sends the task to Claude with computer use tool definitions
3. Claude responds with tool calls (screenshot, click, type, bash, edit)
4. Tools execute against the X11 display (via xdotool/scrot) or shell
5. Results (screenshots, output) are sent back to Claude
6. Loop continues until task is complete or user stops it

The tools are structured as Anthropic API tool definitions -- they are NOT a user-facing Python SDK. Claude decides when and how to use them based on visual analysis of screenshots.

## Makefile targets

| Target | Purpose |
|--------|---------|
| `make setup` | Create venv with uv + Python 3.14, install deps |
| `make run` | Run Streamlit locally |
| `make dev` | Run Docker container with all ports |
| `make build` | Build operative Docker image |
| `make build-desktop` | Build desktop Docker image |
| `make build-xvfb` | Build xvfb Docker image |
| `make test` | Run pytest |
| `make test-cov` | Run pytest with coverage |
| `make lint` | Run ruff check |
| `make format` | Run ruff format |
| `make install-dev` | Install with dev extras |
| `make install-test` | Install with test extras |

## Related Skills

- `hanzo/hanzo-agent.md` -- Agent framework (can use operative for computer use)
- `hanzo/hanzo-mcp.md` -- MCP integration

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: computer-use, automation, desktop, x11, xdotool, vision, streamlit
**Prerequisites**: Python >= 3.14, Docker (recommended), Anthropic API key
