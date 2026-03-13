# Hanzo CLI - Polyglot Command-Line Interface

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-chat.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo CLI is a **Rust binary with polyglot SDK proxying** — a single CLI that provides unified access to all Hanzo services while automatically proxying commands through Python, Go, and TypeScript SDKs as needed. Deploy apps, manage environments, chat with models, and manage API keys.

**NOTE**: The CLI is a **Rust binary** (not an npm package). Config is **TOML** at `~/.config/hanzo/config.toml` (not JSON). It embeds SDK runtimes for polyglot command proxying.

### Why Hanzo CLI?

- **Rust binary**: Fast startup, single static binary
- **Polyglot proxying**: Transparently calls Python/Go/TypeScript SDKs
- **Unified interface**: All Hanzo services from one tool
- **TOML config**: Human-readable configuration
- **Deploy from terminal**: Git-push and manual deploy workflows
- **Chat inline**: Talk to AI models without leaving terminal
- **Scriptable**: All commands return parseable output (JSON with `--json`)

## When to use

- Deploying applications to Hanzo Platform
- Managing environment variables and secrets
- Quick AI model interactions from terminal
- API key lifecycle management
- CI/CD pipeline integration

## Quick reference

| Item | Value |
|------|-------|
| Binary | `hanzo` (Rust) |
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
base_url = "https://api.hanzo.ai/v1"
iam_url = "https://hanzo.id"

[sdk]
# Polyglot SDK runtime paths (auto-detected)
python = "~/.local/bin/python3"
go = "~/go/bin/go"
node = "~/.local/bin/node"
```

## Commands

### Authentication

```bash
hanzo auth login          # Open browser for OAuth2 login
hanzo auth logout         # Clear local credentials
hanzo auth status         # Show current auth state
hanzo auth token          # Print current access token
hanzo auth whoami         # Show current user info
```

### Deployment

```bash
hanzo deploy                      # Deploy current directory
hanzo deploy --env production     # Deploy to specific environment
hanzo deploy --branch main        # Deploy specific branch
hanzo deploy --image ghcr.io/org/app:latest  # Deploy container image
hanzo deploy status               # Show deployment status
hanzo deploy logs                 # Stream deployment logs
hanzo deploy rollback             # Rollback to previous version
hanzo deploy list                 # List recent deployments
```

### Environment Variables

```bash
hanzo env list                    # List all env vars
hanzo env list --env staging      # List for specific environment
hanzo env set KEY=VALUE           # Set environment variable
hanzo env set KEY=VALUE --env prod  # Set for specific environment
hanzo env get KEY                 # Get specific variable
hanzo env rm KEY                  # Remove variable
hanzo env pull .env               # Pull remote env to local file
hanzo env push .env               # Push local .env to remote
```

### AI Chat

```bash
hanzo chat "Hello"                           # Quick chat (default model)
hanzo chat --model zen-70b "Explain consensus"  # Specific model
hanzo chat --stream "Write a poem"           # Streaming output
hanzo chat --system "You are a pirate" "Hello"  # Custom system prompt
hanzo chat --json "List 3 colors"            # JSON output mode
echo "fix this bug" | hanzo chat --stdin     # Pipe input
```

### Models

```bash
hanzo models list                 # List all available models
hanzo models list --provider zen  # Filter by provider
hanzo models info zen-70b         # Model details
hanzo models pricing              # Show pricing table
```

### API Keys

```bash
hanzo keys create                 # Create new API key
hanzo keys create --name "CI/CD"  # Named key
hanzo keys list                   # List all keys
hanzo keys revoke KEY_ID          # Revoke specific key
```

### Projects

```bash
hanzo projects list               # List all projects
hanzo projects create my-app      # Create new project
hanzo projects delete my-app      # Delete project
hanzo projects link               # Link current directory to project
hanzo projects info               # Show linked project details
```

### Secrets (KMS)

```bash
hanzo secrets list                # List secrets for current project
hanzo secrets set DB_URL=postgres://...  # Set a secret
hanzo secrets get DB_URL          # Get secret value
hanzo secrets rm DB_URL           # Remove secret
hanzo secrets sync                # Sync secrets to K8s
```

### SDK Proxying

The CLI transparently proxies to language-specific SDKs when needed:

```bash
# These commands use the Python SDK under the hood
hanzo agent run my-agent.py       # Run agent via Python SDK
hanzo agent list                  # List registered agents

# These use the Go SDK
hanzo orm migrate                 # Run ORM migrations via Go SDK

# These use the TypeScript SDK
hanzo mcp serve                   # Start MCP server via TS SDK
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
HANZO_BASE_URL=https://api.hanzo.ai/v1  # Override API base URL
HANZO_CONFIG=~/.config/hanzo/config.toml  # Custom config path
HANZO_LOG=debug                     # Enable debug logging
HANZO_JSON=1                        # Always output JSON
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `hanzo: command not found` | Not in PATH | Add binary to PATH or install via cargo |
| Auth fails | Token expired | `hanzo auth login` |
| Deploy fails | No project linked | `hanzo projects link` |
| Permission denied | Wrong org/role | Check with `hanzo auth whoami` |
| SDK proxy fails | Missing runtime | Install Python/Go/Node.js as needed |
| Config not found | Wrong path | Check `~/.config/hanzo/config.toml` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS (CLI deploys here)
- `hanzo/hanzo-kms.md` - Secret management backend
- `hanzo/hanzo-chat.md` - Chat API (CLI calls this)
- `hanzo/hanzo-dev.md` - Terminal AI coding agent (different tool — dev is AI agent, CLI is service management)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: cli, command-line, deploy, management, rust
**Prerequisites**: None (single binary)
