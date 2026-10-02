---
name: hanzo-mcp
description: "Unified MCP server orchestrating 260+ AI development tools. Filesystem, shell, browser, code analysis, memory, agents, LLM, computer control, and more."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-mcp",
              "label": "Install Hanzo MCP (pip)",
            },
          ],
      },
  }
---

# Hanzo MCP — Unified AI Tool Server

`pip install hanzo-mcp`

Single MCP server that orchestrates 260+ tools across filesystem, shell, browser, code, memory, agents, LLM, and computer control.

## Quick Start

```bash
# Run MCP server (stdio transport)
hanzo-mcp

# Development mode with extra logging
hanzo-mcp-dev

# Interactive REPL
hanzo-plugin
```

## Tool Categories

### Filesystem (`hanzo-tools-fs`)

- `read_file`, `write_file`, `edit_file` — File operations
- `search`, `find`, `list_files` — File discovery
- `watch` — File change monitoring

### Shell (`hanzo-tools-shell`)

- `execute_command` — Run shell commands with auto-detected shell
- `run_dag` — Execute command DAGs (directed acyclic graphs)

### Browser (`hanzo-tools-browser`)

- Browser automation with Hanzo extension or Playwright fallback
- Page navigation, clicking, form filling, screenshots

### Code Analysis (`hanzo-tools-code`, `hanzo-tools-lsp`)

- AST analysis via tree-sitter (Python, JS, TS, Ruby, Go, Rust)
- LSP integration for code intelligence
- Semantic code search via grep-ast

### Refactoring (`hanzo-tools-refactor`)

- Rename symbols, extract functions, inline variables
- Move code between files, safe cross-file refactoring

### Memory (`hanzo-tools-memory`)

- Store/recall memories with semantic search
- Knowledge base management
- Fact storage and retrieval

### Agents (`hanzo-tools-agent`)

- Multi-agent orchestration
- Spawn and coordinate agent swarms
- Consensus-based decision making

### LLM (`hanzo-tools-llm`)

- Model management and selection
- Provider routing (OpenAI, Anthropic, Hanzo, local)

### Computer Control (`hanzo-tools-computer`)

- Screen capture and OCR
- Mouse/keyboard automation
- Activity-based session compression

### API (`hanzo-tools-api`)

- Call any REST API via OpenAPI spec discovery
- Dynamic API exploration and execution

### Auth & Security (`hanzo-tools-auth`, `hanzo-tools-kms`)

- IAM token management and session handling
- KMS secret CRUD operations

### PaaS (`hanzo-tools-paas`)

- Deploy apps, view logs, manage environments
- IAM user/org/role management

### IDE (`hanzo-tools-ide`, `hanzo-tools-editor`)

- VS Code, Neovim integration
- WebSocket-based editor communication

### Planning (`hanzo-tools-plan`, `hanzo-tools-todo`)

- DAG-based plan execution
- Task management and tracking

### Testing (`hanzo-tools-test`)

- Lint, typecheck, test execution

### VCS (`hanzo-tools-vcs`)

- Git operations, branch management, commit handling

## Configuration

```bash
# MCP settings (checked in order)
~/.config/hanzo/mcp-settings.json
.hanzo-mcp.json
.hanzo/mcp-settings.json
```

```json
{
  "tools": {
    "enabled": ["fs", "shell", "browser", "memory", "agent"],
    "disabled": []
  },
  "transport": "stdio"
}
```

## Environment Variables

```bash
HANZO_MCP_TOKEN=...              # Auth token (auto-generated if missing)
HANZO_MCP_TRANSPORT=stdio        # Transport: stdio or sse
HANZO_QUIET=1                    # Suppress startup output
```

## Claude Desktop Integration

```json
{
  "mcpServers": {
    "hanzo": {
      "command": "hanzo-mcp",
      "args": [],
      "env": {
        "HANZO_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```
