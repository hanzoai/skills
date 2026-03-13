# Hanzo CLI - Polyglot Command-Line Interface

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-chat.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo CLI is a **hybrid Rust+Node+Python command-line tool** — a single CLI that provides unified access to all Hanzo services. Handles project scaffolding, local development, deployment, authentication, and proxies to TypeScript tooling (docs, MDX, UI, MCP) when needed.

### Why Hanzo CLI?

- **Polyglot**: Rust binary core with Node.js and Python SDK proxying
- **Unified interface**: All Hanzo services from one tool
- **TOML config**: Human-readable configuration at `~/.config/hanzo/config.toml`
- **Project scaffolding**: Init new projects with templates
- **Dev server**: Local development with hot reload
- **Deploy**: Push to Hanzo Platform from terminal
- **Scriptable**: JSON output with `--json`

## When to use

- Scaffolding new Hanzo projects
- Running local development servers
- Building and deploying applications
- Authentication and API key management
- Running agents and commerce backends
- Proxying to TypeScript SDK tooling (docs, MCP, etc.)

## Quick reference

| Item | Value |
|------|-------|
| Binary | `hanzo` (Rust + Node + Python) |
| Repo | `github.com/hanzoai/cli` |
| Config | `~/.config/hanzo/config.toml` |
| Auth | OAuth2 via browser |
| Install | `cargo install hanzo-cli` or download binary |

## Installation

```bash
# From cargo (Rust)
cargo install hanzo-cli

# From release binary (macOS)
curl -L https://github.com/hanzoai/cli/releases/latest/download/hanzo-darwin-arm64 -o /usr/local/bin/hanzo
chmod +x /usr/local/bin/hanzo

# From release binary (Linux)
curl -L https://github.com/hanzoai/cli/releases/latest/download/hanzo-linux-amd64 -o /usr/local/bin/hanzo
chmod +x /usr/local/bin/hanzo

# Verify
hanzo --version
hanzo --help
```

## Configuration

```toml
# ~/.config/hanzo/config.toml

[auth]
token = "..."
refresh_token = "..."
expires_at = "2026-04-01T00:00:00Z"

[defaults]
org = "hanzo"
project = "my-app"
environment = "production"

[api]
base_url = "https://api.hanzo.ai"
iam_url = "https://hanzo.id"

[sdk]
# Polyglot SDK runtime paths (auto-detected)
python = "~/.local/bin/python3"
go = "~/go/bin/go"
node = "~/.local/bin/node"
```

## Commands

### Project Scaffolding

```bash
hanzo init                        # Interactive project setup
hanzo init --template next        # From template (next, fastapi, go, rust)
hanzo init --template commerce    # E-commerce project scaffold
```

### Local Development

```bash
hanzo dev                         # Start dev server (auto-detect framework)
hanzo dev --port 3000             # Custom port
hanzo run                         # Run project (production mode)
```

### Build & Deploy

```bash
hanzo build                       # Build project
hanzo deploy                      # Deploy to Hanzo Platform
hanzo deploy --env production     # Deploy to specific environment
hanzo deploy --branch main        # Deploy specific branch
```

### Authentication

```bash
hanzo auth login          # Open browser for OAuth2 login
hanzo auth logout         # Clear local credentials
hanzo auth status         # Show current auth state
hanzo auth token          # Print current access token
hanzo auth whoami         # Show current user info
```

### Commerce

```bash
hanzo commerce init               # Initialize commerce backend
hanzo commerce serve              # Start commerce API server
```

### Agent

```bash
hanzo agent run my-agent.py       # Run agent via Python SDK
hanzo agent list                  # List registered agents
```

### TypeScript Proxy Commands

The CLI transparently proxies to TypeScript SDK tooling:

```bash
# Docs tooling (proxies to @hanzo/docs-cli)
hanzo docs init                   # Initialize docs project
hanzo docs dev                    # Docs dev server

# MDX processing (proxies to @hanzo/docs-mdx)
hanzo mdx build                   # Build MDX content

# UI components (proxies to @hanzo/docs-ui)
hanzo ui init                     # Initialize UI project

# MCP server (proxies to @hanzo/mcp)
hanzo mcp serve                   # Start MCP server
hanzo mcp list                    # List available MCP tools
```

## CI/CD Integration

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    steps:
      - uses: actions/checkout@v4

      - name: Install Hanzo CLI
        run: |
          curl -L https://github.com/hanzoai/cli/releases/latest/download/hanzo-linux-amd64 -o /usr/local/bin/hanzo
          chmod +x /usr/local/bin/hanzo

      - name: Deploy
        env:
          HANZO_API_KEY: ${{ secrets.HANZO_API_KEY }}
        run: |
          hanzo deploy --env production --yes
```

## Environment Variables

```bash
HANZO_API_KEY=your-api-key          # API authentication
HANZO_BASE_URL=https://api.hanzo.ai # Override API base URL
HANZO_CONFIG=~/.config/hanzo/config.toml  # Custom config path
HANZO_LOG=debug                     # Enable debug logging
HANZO_JSON=1                        # Always output JSON
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `hanzo: command not found` | Not in PATH | Add binary to PATH or install via cargo |
| Auth fails | Token expired | `hanzo auth login` |
| Deploy fails | No project linked | Run `hanzo init` first |
| TS proxy fails | Missing Node.js | Install Node.js 18+ |
| Python proxy fails | Missing Python | Install Python 3.12+ |
| Config not found | Wrong path | Check `~/.config/hanzo/config.toml` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS (CLI deploys here)
- `hanzo/hanzo-kms.md` - Secret management backend
- `hanzo/hanzo-chat.md` - Chat API
- `hanzo/hanzo-dev.md` - Terminal AI coding agent (different tool — dev is AI agent, CLI is service management)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: cli, command-line, deploy, management, rust, polyglot
**Prerequisites**: None (single binary, auto-detects runtimes)
