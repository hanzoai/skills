# Hanzo Dev - AI-Powered Development Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Dev is an **AI-powered development platform** -- a monorepo containing a Rust-native CLI (`dev`, `dev-tui`, `dev-exec`), a Node.js wrapper CLI (`@hanzo/dev`), a legacy TypeScript CLI (`codex-cli`), and supporting infrastructure including an app server, MCP client/server, and cloud task system. Built with Bazel + Cargo.

**Upstream fork**: OpenAI Codex CLI. The `codex-cli/` and `codex-rs/` directories preserve the upstream lineage. The Rust rewrite lives in `hanzo-dev/` as a Cargo workspace with 40+ crates.

### What it actually is

- **`hanzo-dev/`** -- Rust workspace (v0.6.74), the primary CLI. Builds three binaries: `dev`, `dev-tui`, `dev-exec`. Contains 40+ crates: core, cli, tui, exec, execpolicy, mcp-client, mcp-server, protocol, git-tooling, file-search, linux-sandbox, ollama, login, browser, chatgpt, cloud-tasks, otel, and more.
- **`dev-cli/`** -- Node.js wrapper package `@hanzo/dev` (v0.6.61) published to npm. Provides `dev` and `hanzo` binaries that delegate to platform-specific native binaries (`@hanzo/dev-darwin-arm64`, etc.).
- **`codex-cli/`** -- Legacy TypeScript CLI from the upstream Codex fork.
- **`codex-rs/`** -- Upstream Rust codebase (preserved for reference).
- **`hanzo-cli/`** -- Additional CLI entry point.
- **`hanzo-node/`** -- Node.js integration layer.
- **`code-rs/`** -- Code editor integration.
- **`sdk/`** -- SDKs (includes `sdk/python/` with codex-app-server-sdk).
- **`console/`** -- Console UI.
- **`contracts/`** -- Smart contracts.
- **Build system**: Bazel (`MODULE.bazel`, `BUILD.bazel`, `defs.bzl`, `.bazelrc`) + Cargo + pnpm.

### What it is NOT

- Not installable via `cargo install hanzo-dev` or `pip install hanzo-dev`
- No `.hanzorc` config file format
- Not a local-only inference tool -- it connects to LLM providers (OpenAI, Anthropic, Ollama, etc.)

## When to use

- AI-assisted coding from the terminal
- Multi-file code generation and editing
- Git-aware code review and commit workflows
- MCP tool integration for development tasks

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/dev` |
| Rust workspace | `hanzo-dev/` (v0.6.74, 40+ crates) |
| npm package | `@hanzo/dev` (v0.6.61) |
| Binaries (Rust) | `dev`, `dev-tui`, `dev-exec` |
| Binaries (npm) | `dev`, `hanzo` |
| Upstream | OpenAI Codex CLI |
| Build (Rust) | `cd hanzo-dev && cargo build --release --bin dev --bin dev-tui --bin dev-exec` |
| Build (Bazel) | `bazel build //...` |
| Build (quick) | `./build-fast.sh` |
| Format | `pnpm format` (prettier), `cargo fmt` (Rust) |
| Test | `cargo test` (Rust), `pnpm test:local` |
| Package manager | pnpm >= 10.29.3 |
| Node | >= 22 |
| Rust edition | 2024 |
| License | Apache-2.0 |

## Repository structure

```
hanzoai/dev/
  MODULE.bazel          # Bazel module definition
  BUILD.bazel           # Root build file
  defs.bzl              # Bazel macros
  .bazelrc              # Bazel config
  package.json          # pnpm monorepo root
  pnpm-workspace.yaml   # Workspace members
  hanzo.sh              # Install script
  build-fast.sh         # Fast build script
  justfile              # Just task runner
  hanzo-dev/            # PRIMARY: Rust workspace (40+ crates)
    Cargo.toml          # Workspace manifest (v0.6.74)
    cli/                # CLI binary crate
    tui/                # TUI binary crate
    exec/               # Exec binary crate
    core/               # Core agent logic
    mcp-client/         # MCP client
    mcp-server/         # MCP server
    protocol/           # Wire protocol
    browser/            # Browser integration
    ollama/             # Ollama provider
    login/              # Auth/login
    linux-sandbox/      # Sandboxing
    file-search/        # File search
    git-tooling/        # Git operations
    ...                 # 25+ more crates
  dev-cli/              # npm wrapper (@hanzo/dev v0.6.61)
    bin/dev.js          # Entry point
    package.json        # Publishes `dev` and `hanzo` binaries
  codex-cli/            # Legacy TS CLI (upstream fork)
  codex-rs/             # Upstream Rust (preserved)
  code-rs/              # Code editor integration
  hanzo-cli/            # Additional CLI
  hanzo-node/           # Node.js integration
  sdk/                  # SDKs (Python, etc.)
  console/              # Console UI
  contracts/            # Smart contracts
  platform/             # Platform integration
  shell-tool-mcp/       # Shell MCP server
  npm-binaries/         # Platform-specific npm binary packages
  standalone-hanzo-dev/ # Standalone build
  published-hanzo-dev/  # Published crate
```

## Rust workspace crates (hanzo-dev/)

| Crate | Purpose |
|-------|---------|
| `cli` | Main CLI binary (`dev`) |
| `tui` | Terminal UI binary (`dev-tui`) |
| `exec` | Execution binary (`dev-exec`) |
| `core` | Core agent logic |
| `common` | Shared types |
| `protocol` | Wire protocol definitions |
| `mcp-client` | MCP client implementation |
| `mcp-server` | MCP server implementation |
| `mcp-types` | MCP type definitions |
| `browser` | Browser automation |
| `chatgpt` | ChatGPT integration |
| `ollama` | Ollama provider |
| `login` | Authentication |
| `exec` | Command execution |
| `execpolicy` | Execution policy/sandboxing |
| `file-search` | File search |
| `git-tooling` | Git operations |
| `git-apply` | Git patch application |
| `apply-patch` | Patch application |
| `linux-sandbox` | Linux sandboxing |
| `app-server` | App server |
| `app-server-protocol` | Server protocol |
| `backend-client` | Backend API client |
| `cloud-tasks` | Cloud task execution |
| `otel` | OpenTelemetry |
| `tunnel-bridge` | Tunnel bridge |
| `code-auto-drive-core` | Auto-drive coding |
| `responses-api-proxy` | Responses API proxy |

## Installation

```bash
# Via npm (recommended -- auto-selects platform binary)
npm install -g @hanzo/dev

# Via install script
curl -fsSL https://raw.githubusercontent.com/hanzoai/dev/main/hanzo.sh | sh

# From source (Rust)
git clone https://github.com/hanzoai/dev.git
cd dev/hanzo-dev
cargo build --release --bin dev --bin dev-tui --bin dev-exec

# From source (quick)
git clone https://github.com/hanzoai/dev.git
cd dev
./build-fast.sh
```

## Usage

```bash
# Interactive session
dev

# TUI mode
dev-tui

# Inline prompt
dev "Explain this code"

# With specific model
dev --model gpt-4o "Fix the build errors"
```

## Configuration

Configuration is via `config.toml` (see `config.toml.example` in repo root). The Rust CLI reads config from standard XDG paths, not a `.hanzorc` file.

## Related Skills

- `hanzo/hanzo-mcp.md` -- MCP tools used by dev
- `hanzo/hanzo-extension.md` -- VS Code/JetBrains/browser extensions

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: dev, coding, cli, terminal, rust, bazel, codex
**Prerequisites**: Node.js >= 22 (for npm wrapper) or Rust toolchain (for source build)
