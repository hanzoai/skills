# Hanzo ACI - Agent Computer Interface

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-code.md`, `hanzo/hanzo-agent.md`

## Overview

Hanzo ACI (Agent Computer Interface) is a **Python toolkit** that gives AI agents the ability to edit files, lint code, index codebases, execute shell commands, and convert documents. It provides the backend "hands" for Hanzo's AI coding agents. Integrates with `hanzo-mcp` for Model Context Protocol server capabilities. Installable as `hanzo-aci` from PyPI.

### Why Hanzo ACI?

- **File editing for agents**: Programmatic file create/modify/view with line-number precision, undo history, and file caching
- **Code linting**: Tree-sitter based multi-language linting (syntax validation before applying changes)
- **Codebase indexing**: LocAgent-based code search and indexing with optional LlamaIndex integration
- **Document conversion**: Convert PDF, DOCX, PPTX, XLSX, HTML, YouTube transcripts to Markdown
- **Shell execution**: Safe subprocess execution utilities for agent tool calls
- **MCP integration**: CLI (`hanzo-dev`) unifies ACI editor tools with MCP server in a single process

### Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: Poetry (with `poetry.lock`)
- **Code Analysis**: tree-sitter 0.24+, tree-sitter-language-pack 0.7.3, grep-ast, libcst 1.5.0
- **Document Processing**: pdfminer-six, pypdf, mammoth, python-pptx, beautifulsoup4, markdownify, openpyxl, xlrd
- **Data/Viz**: pandas, matplotlib, networkx
- **Validation**: pydantic 2.11+
- **Fuzzy Matching**: rapidfuzz 3.13+
- **MCP**: hanzo-mcp (local path dep), mcp >=1.9.4, fastmcp >=2.9.2
- **Optional**: llama-index 0.12+ (for advanced code search/retrieval)
- **Testing**: pytest 8, pytest-forked
- **Linting**: ruff, pre-commit
- **CI**: GitHub Actions (unit tests, integration tests, lint, PyPI release)

### OSS Base

Original work by Hanzo. The LocAgent indexing module draws on code intelligence patterns. The editor module is purpose-built for agent-driven file manipulation.

Repo: `github.com/hanzoai/aci`

## When to use

- Building AI agents that need to read, edit, and create files
- Adding code linting validation to agent tool pipelines
- Indexing codebases for AI-powered code search
- Converting documents (PDF, DOCX, PPTX, XLSX) to Markdown for LLM context
- Running shell commands from agent workflows
- Setting up a unified MCP + ACI development server

## Hard requirements

1. **Python 3.12+**
2. **Poetry** for dependency management
3. **hanzo-mcp** package (local path dependency at `../mcp` -- must be available)
4. **tree-sitter** native libraries (installed via tree-sitter-language-pack)

## Quick reference

| Item | Value |
|------|-------|
| PyPI Package | `hanzo-aci` |
| Version | `1.0.0` |
| Repo | `github.com/hanzoai/aci` |
| Branch | `main` |
| License | MIT |
| Python | `>=3.12` |
| CLI | `hanzo-dev` (serve, edit, index) |
| Server CLI | `hanzo-dev-server` |
| Test | `poetry run pytest` |
| Lint | `make lint` |

## One-file quickstart

### Install and use

```bash
# Install from PyPI
pip install hanzo-aci

# Or from source with all extras
git clone https://github.com/hanzoai/aci.git
cd aci
poetry install --extras llama

# Start the unified MCP + ACI server
hanzo-dev serve

# Edit a file via CLI
hanzo-dev edit myfile.py --line 42

# Index a codebase
hanzo-dev index /path/to/project --output index.json
```

### Use in Python code

```python
from hanzo_aci import file_editor, FileCache

# Initialize the file cache
cache = FileCache()

# Open and view a file with line numbers
result = file_editor.open_file("src/main.py", line_number=1)
print(result)

# The editor supports:
# - open_file(path, line_number=None)
# - create_file(path, content)
# - edit_file(path, old_str, new_str)
# - view_file(path, start_line, end_line)
# - undo_edit(path)

# Lint code before applying changes
from hanzo_aci.linter import DefaultLinter
linter = DefaultLinter()
errors = linter.lint("src/main.py")
```

## Core Concepts

### Architecture

```
┌─────────────────────────────────────────────┐
│              hanzo-dev CLI                    │
│  (unified entry point)                       │
├──────────┬──────────────┬───────────────────┤
│  serve   │    edit      │     index         │
│  (MCP)   │  (ACI)      │   (LocAgent)      │
└────┬─────┴──────┬───────┴────────┬──────────┘
     │            │                │
┌────▼────┐ ┌────▼──────────┐ ┌───▼──────────┐
│hanzo-mcp│ │ hanzo_aci/    │ │ indexing/     │
│ server  │ │  editor/      │ │  locagent/   │
│         │ │  linter/      │ │   tools.py   │
│         │ │  utils/       │ │   repo/      │
└─────────┘ └───────────────┘ └──────────────┘
```

### Editor Module

The core module -- provides file manipulation for agents.

| Component | File | Purpose |
|-----------|------|---------|
| `editor.py` | 26KB | Main `FileEditor` class: open, create, edit, view, undo |
| `file_cache.py` | 5KB | In-memory cache for file contents and metadata |
| `history.py` | 4KB | Undo/redo history tracking per file |
| `encoding.py` | 4KB | File encoding detection and handling |
| `exceptions.py` | 1KB | Custom exceptions (`EditorError`, `FileNotFoundError`, etc.) |
| `prompts.py` | 1KB | Prompt templates for agent interactions |
| `results.py` | 1KB | Structured result types for editor operations |
| `md_converter.py` | 41KB | Convert PDF/DOCX/PPTX/XLSX/HTML/YouTube to Markdown |

### Linter Module

Tree-sitter based code validation.

| Component | File | Purpose |
|-----------|------|---------|
| `linter.py` | 5KB | Main linter orchestrator -- dispatches to language-specific impl |
| `base.py` | 2KB | Base linter interface |
| `impl/` | -- | Language-specific linter implementations |

### Indexing Module (LocAgent)

Code intelligence and search.

| Component | File | Purpose |
|-----------|------|---------|
| `tools.py` | 46KB | Core indexing tools: `index_codebase`, search, file analysis |
| `results.py` | 7KB | Search result types and ranking |
| `compress.py` | 2KB | Code compression for context windows |
| `utils.py` | 1KB | Path and file utilities |
| `repo/` | -- | Repository structure analysis |

### Utilities

| Component | File | Purpose |
|-----------|------|---------|
| `shell.py` | 2KB | Safe subprocess execution with timeout |
| `diff.py` | 1KB | Unified diff generation for file changes |
| `logger.py` | 800B | Structured logging setup |

### CLI Commands

```
hanzo-dev serve [--transport stdio|sse]    # Start MCP server (default)
hanzo-dev edit <file> [--line N]           # Open file in ACI editor
hanzo-dev index <path> [--output file]     # Index a codebase

Common flags:
  --allow-path <path>       Allow access to specific paths
  --enable-all-tools        Enable all available tools
  --enable-agent-tool       Enable agent delegation tool
  --log-level DEBUG|INFO|WARNING|ERROR
```

## Directory structure

```
aci/
├── hanzo_aci/
│   ├── __init__.py            # Exports file_editor, FileCache
│   ├── cli.py                 # hanzo-dev CLI (v0.3.1)
│   ├── editor/
│   │   ├── __init__.py        # Editor module exports
│   │   ├── editor.py          # FileEditor class (26KB)
│   │   ├── file_cache.py      # File content caching
│   │   ├── history.py         # Undo/redo history
│   │   ├── encoding.py        # Charset detection
│   │   ├── exceptions.py      # EditorError types
│   │   ├── prompts.py         # Agent prompt templates
│   │   ├── results.py         # Structured results
│   │   ├── config.py          # Editor configuration
│   │   └── md_converter.py    # Document-to-Markdown (41KB)
│   ├── linter/
│   │   ├── __init__.py        # Linter exports
│   │   ├── linter.py          # Main linter
│   │   ├── base.py            # Base interface
│   │   └── impl/              # Language-specific linters
│   ├── indexing/
│   │   └── locagent/
│   │       ├── tools.py       # Code indexing (46KB)
│   │       ├── results.py     # Search results
│   │       ├── compress.py    # Code compression
│   │       ├── utils.py       # Utilities
│   │       └── repo/          # Repo analysis
│   └── utils/
│       ├── diff.py            # Diff generation
│       ├── shell.py           # Shell execution
│       └── logger.py          # Logging setup
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── data/                  # Test fixtures
├── dev_config/
│   └── python/
│       └── .pre-commit-config.yaml
├── .github/workflows/
│   ├── py-unit-tests.yml      # Unit test CI
│   ├── py-intg-tests.yml      # Integration test CI
│   ├── lint.yml               # Ruff lint CI
│   ├── pypi-release.yml       # PyPI publish on release
│   └── hanzo-resolver.yml     # Issue resolver
├── pyproject.toml             # Poetry config, v1.0.0
├── poetry.lock
├── Makefile                   # lint, install-pre-commit-hooks
└── pytest.ini
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: hanzo_mcp` | Missing local dependency | Ensure `../mcp` exists or install `hanzo-mcp` from PyPI |
| tree-sitter parse errors | Wrong language pack version | Pin `tree-sitter-language-pack==0.7.3` |
| `hanzo-dev serve` hangs | No MCP client connected | Use stdio transport with a compatible MCP client |
| llama-index import error | Optional dep not installed | Install with `poetry install --extras llama` |
| Encoding errors on file read | Binary file detected | ACI uses `binaryornot` to skip binary files |
| Lint CI fails | Pre-commit hooks not installed | Run `make install-pre-commit-hooks` |

## Related Skills

- `hanzo/hanzo-mcp.md` - MCP server that ACI integrates with
- `hanzo/hanzo-code.md` - Hanzo Code editor (uses ACI concepts for its agent tools)
- `hanzo/hanzo-agent.md` - Multi-agent SDK that can use ACI tools
- `hanzo/hanzo-operative.md` - Computer use agent (complementary to ACI)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: agent, aci, editor, linter, mcp, code-intelligence
**Prerequisites**: Python 3.12+, Poetry, tree-sitter
