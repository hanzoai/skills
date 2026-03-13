# Hanzo SDK - Unified Multi-Language CLI and Client Library

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/python-sdk.md`, `hanzo/js-sdk.md`, `hanzo/go-sdk.md`, `hanzo/rust-sdk.md`, `hanzo/hanzo-openapi.md`

## Overview

Hanzo SDK is a **unified multi-language SDK and CLI** (`@hanzo/cli`) that wraps Python, TypeScript, Rust, and Go implementations into a single `hanzo` command. Published as an npm package with optional Rust native bindings via napi-rs. Version 0.1.0.

### How This Differs from Individual SDKs

The individual SDK repos (`python-sdk`, `js-sdk`, `go-sdk`, `rust-sdk`) are **Stainless-generated API clients** focused on typed HTTP wrappers for `api.hanzo.ai`. This unified SDK is different:

- **CLI-first**: Provides the `hanzo` CLI binary with subcommands (`node`, `agent`, `mcp`, `net`, `dev`, `auth`, `config`)
- **Multi-language monorepo**: All four language implementations in one repo (`src/py/`, `src/js/`, `src/rs/`, `src/go/`)
- **Operational commands**: `hanzo node start`, `hanzo agent run`, `hanzo mcp serve`, `hanzo net status` -- not just API calls
- **Native bindings**: Rust compiled to `.node` via napi-rs for performance-critical paths, with JS fallback
- **Cross-language testing**: Integration tests verify feature parity across all implementations

The individual SDKs are what you `import` in application code. This SDK is what you `install` as a developer tool.

### Tech Stack

- **Primary**: TypeScript (CLI and JS client, compiled to `dist/`)
- **Rust**: Native bindings via napi-rs (`src/rs/`, optional)
- **Python**: `hanzo` PyPI package (`src/py/`, pyproject.toml)
- **Go**: `github.com/hanzoai/sdk` module (`src/go/`)
- **CLI**: Commander.js with chalk, inquirer, ora
- **Build**: npm + tsc + cargo (multi-stage)
- **Tests**: Jest (JS), pytest (Python), cargo test (Rust), go test (Go)

### OSS Base

Repo: `hanzoai/sdk` (298MB -- large due to committed `node_modules` and `dist`).

## When to use

- Installing the `hanzo` CLI for local development
- Running AI nodes, agents, or MCP servers from the command line
- Developing against all Hanzo services from a single entry point
- Cross-language SDK feature parity testing

## When NOT to use

- **Application code imports**: Use the individual SDKs instead:
  - Python: `pip install hanzo` (from `hanzoai/python-sdk`)
  - TypeScript: `npm install @hanzo/sdk` (from `hanzoai/js-sdk`)
  - Go: `go get github.com/hanzoai/go-sdk` (from `hanzoai/go-sdk`)
  - Rust: `cargo add hanzo` (from `hanzoai/rust-sdk`)

## Hard requirements

1. **Node.js >= 16** for the CLI
2. **npm** for installation (published as `@hanzo/cli`)
3. **Rust toolchain** optional (for native bindings, falls back to JS)

## Quick reference

| Item | Value |
|------|-------|
| npm Package | `@hanzo/cli` |
| Version | `0.1.0` |
| Binary | `hanzo` |
| Repo | `github.com/hanzoai/sdk` |
| Branch | `main` |
| License | MIT |

## Repository Structure

```
sdk/
  package.json            # @hanzo/cli, commander, axios, chalk
  Makefile                # Build/test all languages
  tsconfig.json
  jest.config.js
  bin/
    hanzo.js              # CLI entry point
  dist/                   # Compiled JS output
  src/
    js/                   # TypeScript implementation
      index.ts            # Exports: HanzoClient, HanzoAgent, HanzoMCP, HanzoNode
      client.ts           # Base HTTP client (axios)
      cli.ts              # CLI setup (commander)
      agent.ts            # Agent operations
      mcp.ts              # MCP server operations
      node.ts             # Node operations
      commands/
        agent.ts          # hanzo agent {run,list,stop,logs}
        auth.ts           # hanzo auth {login,logout,status,token}
        config.ts         # hanzo config {get,set,list,reset}
        dev.ts            # hanzo dev {start,stop,status}
        mcp.ts            # hanzo mcp {serve,list,install}
        net.ts            # hanzo net {status,peers,connect}
        node.ts           # hanzo node {start,stop,status,logs}
    py/
      pyproject.toml      # hanzo PyPI package
      hanzo/              # Python package
      src/                # Additional source
    rs/
      Cargo.toml          # hanzo-sdk crate
      src/                # Rust implementation
      build.rs            # napi-rs build script
    go/
      (scaffolded, minimal)
  tests/                  # Cross-language integration tests
  scripts/
    postinstall.js        # Optional native binding setup
```

## CLI Commands

All commands work identically across all language implementations:

```bash
# Node management
hanzo node start          # Start local AI node
hanzo node stop           # Stop node
hanzo node status         # Node status
hanzo node logs           # View node logs

# Agent operations
hanzo agent run           # Run AI agents
hanzo agent list          # List running agents
hanzo agent stop          # Stop an agent
hanzo agent logs          # Agent logs

# MCP server
hanzo mcp serve           # Start MCP server
hanzo mcp list            # List available tools
hanzo mcp install         # Install MCP tools

# Network
hanzo net status          # Network status
hanzo net peers           # List peers
hanzo net connect         # Connect to peer

# Development
hanzo dev start           # Start dev environment
hanzo dev stop            # Stop dev environment

# Auth
hanzo auth login          # Authenticate with Hanzo
hanzo auth logout         # Log out
hanzo auth status         # Check auth status
hanzo auth token          # Show current token

# Config
hanzo config get <key>    # Get config value
hanzo config set <k> <v>  # Set config value
hanzo config list         # List all config
hanzo config reset        # Reset to defaults
```

## Development

### Build All Languages

```bash
make build                # Build py + js + rs + go
make build-js             # TypeScript only
make build-py             # Python only
make build-rs             # Rust only (native bindings)
make build-go             # Go only
```

### Test All Languages

```bash
make test                 # All tests + integration
make test-js              # Jest tests
make test-py              # pytest tests
make test-rs              # cargo test
make test-go              # go test
make test-matrix          # Full version matrix
```

### Other Targets

```bash
make lint                 # Lint all languages
make format               # Format all languages
make check                # lint + format + test
make clean                # Remove all build artifacts
make install              # Install all packages locally
make dev                  # Watch mode (tsc + cargo watch)
```

### Publishing

```bash
make cd-publish           # Publish all packages
make cd-publish-js        # npm publish
make cd-publish-py        # twine upload to PyPI
make cd-publish-rs        # cargo publish to crates.io
make cd-publish-go        # go mod tidy + git tag
```

## Maturity Note

This is an early-stage unification effort (v0.1.0). The individual language SDKs (`python-sdk`, `js-sdk`, `go-sdk`, `rust-sdk`) are more mature and production-ready. Use this SDK primarily for:

- The `hanzo` CLI
- Development tooling
- Cross-language testing

For production application code, prefer the individual SDKs.

## Related Skills

- `hanzo/python-sdk.md` - Production Python SDK (`pip install hanzo`)
- `hanzo/js-sdk.md` - Production TypeScript SDK (`@hanzo/sdk`)
- `hanzo/go-sdk.md` - Production Go SDK (`github.com/hanzoai/go-sdk`)
- `hanzo/rust-sdk.md` - Production Rust SDK
- `hanzo/hanzo-openapi.md` - API specifications these SDKs implement
- `hanzo/hanzo-cli.md` - CLI tooling docs
- `hanzo/hanzo-mcp.md` - MCP tools and server

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: sdk, cli, multi-language, developer-tools
**Prerequisites**: Node.js 16+, npm
