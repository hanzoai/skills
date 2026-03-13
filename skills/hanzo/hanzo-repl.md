# Hanzo REPL - Interactive MCP Testing Environment

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-aci.md`, `hanzo/hanzo-chat.md`

## Overview

Hanzo REPL is a **Python interactive environment** for testing Hanzo MCP tools and AI integration. Three interfaces: a Textual TUI with command palette, an IPython REPL with magic commands, and a basic prompt-toolkit REPL. All MCP tools are available as Python functions. Chat with any LLM provider via litellm, and the LLM can call MCP tools in a loop.

### Why Hanzo REPL?

- **Direct MCP access**: All MCP tools available as Python functions with tab completion
- **Integrated chat**: Converse with AI that can invoke MCP tools autonomously
- **Three interfaces**: Textual TUI (default), IPython with magic commands, basic CLI
- **Multi-provider**: Any LLM via litellm (OpenAI, Anthropic, Groq, Together, Ollama, etc.)
- **Voice mode**: Optional speech recognition + text-to-speech (SpeechRecognition, pyttsx3)
- **Self-editing**: Edit REPL source code from within the REPL

### Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: uv (with hatchling build backend)
- **TUI**: Textual + textual-dev
- **Classic REPL**: IPython, prompt-toolkit, pygments
- **LLM**: litellm (multi-provider)
- **MCP**: hanzo-mcp (local path dependency)
- **Output**: Rich (tables, syntax highlighting, markdown, panels)
- **Voice**: SpeechRecognition, pyttsx3, pyaudio, sounddevice (optional)
- **Testing**: pytest, pytest-asyncio
- **Linting**: ruff, black, mypy

Repo: `github.com/hanzoai/repl`

## When to use

- Testing MCP tools interactively before integrating them
- Rapid prototyping with AI that has tool access
- Debugging MCP tool behavior in a live environment
- Building and testing agentic workflows with immediate feedback

## Quick reference

| Item | Value |
|------|-------|
| Package | `hanzo-repl` |
| Repo | `github.com/hanzoai/repl` |
| Version | 0.1.0 |
| License | MIT |
| Python | 3.12+ |
| Entry points | `hanzo-repl` (Textual), `hanzo-repl-ipython`, `hanzo-repl-basic` |

## Installation & Setup

```bash
# Clone and setup
git clone https://github.com/hanzoai/repl
cd repl
make setup    # Creates venv via uv, installs deps

# Or manually
uv venv && uv pip install -e .
```

Requires at least one LLM API key:
```bash
export OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, etc.
```

## Usage

### Start the REPL

```bash
make dev              # Textual TUI (default, recommended)
make dev-ipython      # IPython with magic commands
make chat             # Basic prompt-toolkit REPL
```

Or directly:
```bash
uv run hanzo-repl               # Textual TUI
uv run hanzo-repl-ipython       # IPython
uv run hanzo-repl-basic         # Basic CLI
```

### Direct Tool Access

```python
>>> read_file(file_path="/etc/hosts")
>>> write_file(file_path="test.txt", content="Hello, World!")
>>> search(query="def main", path=".")
>>> run_command(command="ls -la")
```

### Chat with AI + MCP Tools

```python
>>> chat("What files are in the current directory?")
# AI calls MCP tools to answer

>>> chat("Create a Python script that prints the current time")
# AI generates and writes the file via MCP

>>> chat("Find all Python files with 'test' in the name")
# AI uses search + read tools
```

### IPython Magic Commands

```python
%chat What is the weather today?

%%ai
Can you help me create a web scraper?
I need it to extract titles from a list of URLs.

%tools                                # List available MCP tools
%tool read_file {"file_path": "README.md"}  # Execute specific tool
%model claude-3-opus-20240229         # Change model
%edit_self ipython_repl.py            # Edit REPL source code
```

### REPL Commands (all interfaces)

```
/help        Show help
/tools       List available MCP tools
/providers   List detected LLM providers
/model       Show or set current model
/context     Show conversation context
/reset       Reset conversation context
/clear       Clear screen
/exit        Exit REPL
```

## Makefile Targets

```bash
make dev              # Start Textual TUI
make dev-ipython      # Start IPython REPL
make chat             # Start basic chat
make test             # Run pytest
make test-integration # Run integration tests (needs API keys)
make lint             # Lint with ruff
make format           # Format with black + ruff
make type-check       # mypy
make clean            # Remove build artifacts
make build            # Build distribution (swaps pyproject.toml for PyPI)
make publish          # Publish to PyPI
```

## Key Files

| File | Purpose |
|------|---------|
| `hanzo_repl/textual_repl.py` | Textual TUI with status bar, command palette |
| `hanzo_repl/ipython_repl.py` | IPython REPL with magic commands |
| `hanzo_repl/repl.py` | Basic prompt-toolkit REPL |
| `hanzo_repl/cli.py` | Click CLI entry point (routes to ipython or basic) |
| `hanzo_repl/llm_client.py` | Multi-provider LLM client (litellm wrapper) |
| `hanzo_repl/tool_executor.py` | Executes MCP tools from LLM responses |
| `hanzo_repl/command_palette.py` | Textual command palette widget |
| `hanzo_repl/command_suggestions.py` | Autocomplete suggestions |
| `hanzo_repl/voice_mode.py` | Speech recognition + TTS |
| `hanzo_repl/backends.py` | Backend manager for LLM providers |
| `hanzo_repl/tests.py` | Built-in test suite for MCP tools |
| `mcp_repl.py` | Minimal direct-access REPL script |

## LLM Provider Detection

The REPL auto-detects available providers by checking environment variables:

| Provider | Env Vars |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY` |
| Google | `GOOGLE_API_KEY`, `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| Ollama | (no key needed) |
| Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

## Related Skills

- `hanzo/hanzo-mcp.md` - MCP server that provides the tools
- `hanzo/hanzo-aci.md` - Agent Computer Interface (file editing, linting, shell)
- `hanzo/hanzo-chat.md` - Chat UI (REPL can serve as MCP backend for Chat)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: repl, mcp, interactive, testing, python, textual, ipython, llm
**Prerequisites**: Python 3.12+, uv, at least one LLM API key
