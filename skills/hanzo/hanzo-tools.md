# Hanzo Tools - Unified AI Agent Tool Registry

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-node.md`

## Overview

Hanzo Tools is a **unified registry of 88 tools across 20 categories** for all AI agent operations across the Hanzo and Zoo ecosystems. Core implementation in Rust with Python (PyO3) and Node.js (NAPI-RS) bindings. Also contains pre-built agent definitions and supporting scripts.

### Why Hanzo Tools?

- **One registry**: Single source of truth for all agent tools across Hanzo Node, Zoo Node, and Hanzo Agent
- **Type-safe**: Rust core with JSON Schema validation for all tool parameters
- **Cross-language**: Python and Node.js bindings from the same Rust codebase
- **Async execution**: All tool execution is asynchronous
- **Categorized**: 20 logical categories for discoverability

### Tech Stack

- **Core**: Rust
- **Python bindings**: PyO3 / maturin
- **Node.js bindings**: NAPI-RS
- **Parameter validation**: JSON Schema
- **Scripts**: TypeScript (Deno-compatible), Shell, Python
- **License**: Apache-2.0

### Repo

`github.com/hanzoai/tools`

## When to use

- Providing tool capabilities to AI agents (Hanzo Agent, Zoo Node, custom agents)
- Looking up available tools by category or name
- Adding new tools to the unified registry
- Building Rust/Python/Node.js applications that need tool execution
- Packaging and deploying agent definitions

## Hard requirements

1. **Rust** for building the core registry
2. **maturin** for Python bindings
3. **Node.js** for NAPI-RS bindings

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/tools` |
| Language | Rust (core), TypeScript (scripts), Python (tests) |
| Total tools | 88 |
| Total categories | 20 |
| Rust crate | `hanzo-tools` |
| Python module | `hanzo_tools` |
| npm package | `@hanzo/tools` |
| Catalog | `CATALOG.json` (43KB, machine-readable) |
| Manifest | `tools-manifest.json` |

## Tool categories

| Category | Count | Tools |
|----------|-------|-------|
| **filesystem** | 10 | read, write, directory_tree, tree, watch, diff, rules, content_replace, find, grep |
| **shell** | 15 | run_command, run, run_background, python_execute, processes, ps, pkill, kill, logs, open, npx, uvx, pkg, zsh, streaming_command |
| **agent** | 7 | agent, swarm, claude, critic, review, clarification, network |
| **database** | 7 | db, sql, graph, sql_query, sql_search, graph_add, graph_query |
| **system** | 6 | tool_enable, tool_disable, tool_list, stats, mode, config |
| **search** | 5 | unified_search, search, web_search, batch_search, symbols |
| **project** | 5 | project_analyze, dependency_tree, build_project, test_run, refactor_code |
| **memory** | 4 | memory, recall_memories, store_facts, summarize_to_memory |
| **todo** | 3 | todo, todo_read, todo_write |
| **mcp** | 3 | mcp, mcp_add, mcp_stats |
| **editor** | 3 | neovim_edit, neovim_command, neovim_session |
| **llm** | 3 | llm, consensus, llm_manage |
| **jupyter** | 3 | jupyter, notebook_read, notebook_edit |
| **git** | 3 | git, git_status, git_search |
| **edit** | 3 | edit, multi_edit, apply_patch |
| **vector** | 2 | vector_index, vector_search |
| **ast** | 2 | ast, ast_multi_edit |
| **browser** | 2 | screenshot, navigate |
| **thinking** | 1 | think |
| **lsp** | 1 | lsp |

## Build commands

```bash
# Build the Rust core
cd rust/
cargo build --release

# Run tests
cargo test

# Generate catalog
cargo run --example generate_catalog > ../CATALOG.json

# Python bindings
pip install maturin
maturin develop

# Node.js bindings
npm install
npm run build
```

## Usage

### Rust

```rust
use hanzo_tools::Registry;

let registry = Registry::new();
let tools = registry.list_tools();

// Execute a tool
let result = registry.execute("read", json!({
    "path": "/path/to/file.txt"
})).await?;
```

### Python

```python
from hanzo_tools import ToolRegistry

registry = ToolRegistry()
tools = registry.list_tools()

result = await registry.execute("read", {
    "path": "/path/to/file.txt"
})
```

### Node.js

```javascript
const { Registry } = require('@hanzo/tools');

const registry = new Registry();
const tools = registry.listTools();
```

## Architecture

```
hanzo-tools (Rust)
  ToolRegistry (88 tools, 20 categories)
  JSON Schema validation
  Async execution
       |
       +-- Python bindings (PyO3) --> Hanzo Agent (Python)
       +-- Node.js bindings (NAPI-RS)
       +-- Direct Rust dependency --> Hanzo Node, Zoo Node
```

## Key directories

```
rust/                       # Rust core implementation
  Makefile                  # Build automation
  build.py                  # Build script
  test_integration.py       # Integration tests
  README.md                 # Rust-specific docs
tools/                      # Tool definitions (large, 100+ entries)
agents/                     # Pre-built agent definitions
  audio-insight/
  bitcoin-chart-analyst/
  bitcoin-market-analyst/
  bitcoiner/
  code-directory-and-git-repository-analyzer/
  corporate-treasury-analyzer/
  e-librarian/
  enhanced-image-generator/
  finance_bro/
  image-analyzer/
  learning_tutor/
  linear_agent/
  markdown-content-writer/
  math_problem_solver/
  mind-map-creator/
  poster-creator/
  prompt_writer/
  youtube_expert/
  ... (35 total)
scripts/                    # TypeScript utility scripts
  download_agents.ts        # Agent download/sync
  download_tools.ts         # Tool download/sync
  build_tools.ts            # Tool build script
  launch.ts                 # Launch script
  generate_images.ts        # Image generation
  run_node.sh               # Node runner
  tests/                    # Script tests
crons/                      # Scheduled tasks
  fetch_usa_news.json       # News fetching cron config
CATALOG.json                # Machine-readable tool catalog (43KB)
AGENTS_CATALOG.json         # Machine-readable agent catalog (10KB)
tools-manifest.json         # Registry manifest with stats and integration info
package_agents.sh           # Agent packaging script
run_zoo_with_agents.sh      # Zoo integration runner
verify_agents.py            # Agent verification script
```

## Adding a new tool

1. Add tool definition to `src/registry.rs` in the appropriate category
2. Implement execution logic in the category-specific execution method
3. Run tests: `cargo test`
4. Regenerate catalog: `cargo run --example generate_catalog > ../CATALOG.json`
5. Update README.md

## Integration points

| Consumer | Method |
|----------|--------|
| Hanzo Node (Rust) | Direct Rust crate dependency |
| Zoo Node (Rust) | Direct Rust crate dependency |
| Hanzo Agent (Python) | Python bindings via PyO3 |
| MCP Server | Tool definitions exported as MCP-compatible schemas |

## Related Skills

- `hanzo/hanzo-agent.md` - Multi-agent SDK (Python, primary consumer)
- `hanzo/hanzo-mcp.md` - Model Context Protocol tools
- `hanzo/hanzo-node.md` - Hanzo Node (Rust, direct dependency)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: tools, agents, registry, rust, mcp, ai-agents
**Prerequisites**: Rust, Python (for bindings), Node.js (for bindings)
