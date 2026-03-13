# Hanzo CLI - Command-Line Interface

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-chat.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo CLI provides **command-line access to all Hanzo services** — deploy apps, manage environments, chat with models, manage API keys, and interact with the full Hanzo ecosystem from your terminal.

### Why Hanzo CLI?

- **Unified interface**: All Hanzo services from one tool
- **Deploy from terminal**: Git-push and manual deploy workflows
- **Chat inline**: Talk to AI models without leaving terminal
- **Key management**: Create, list, revoke API keys
- **Environment management**: Set/get env vars per project
- **Scriptable**: All commands return parseable output

## When to use

- Deploying applications to Hanzo Platform
- Managing environment variables and secrets
- Quick AI model interactions from terminal
- API key lifecycle management
- CI/CD pipeline integration

## Quick reference

| Item | Value |
|------|-------|
| Install | `npm install -g @hanzo/cli` |
| Package | `@hanzo/cli` (npm) |
| Repo | `github.com/hanzoai/cli` |
| Config | `~/.hanzo/config.json` |
| Auth | OAuth2 via browser |

## Installation

```bash
# npm (global)
npm install -g @hanzo/cli

# pnpm
pnpm add -g @hanzo/cli

# Verify
hanzo --version
hanzo --help
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

## Configuration

```json
// ~/.hanzo/config.json
{
  "auth": {
    "token": "...",
    "refreshToken": "...",
    "expiresAt": "2026-04-01T00:00:00Z"
  },
  "defaults": {
    "org": "hanzo",
    "project": "my-app",
    "environment": "production"
  },
  "apiUrl": "https://api.hanzo.ai/v1",
  "iamUrl": "https://hanzo.id"
}
```

## CI/CD Integration

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    steps:
      - uses: actions/checkout@v4

      - name: Install Hanzo CLI
        run: npm install -g @hanzo/cli

      - name: Deploy
        env:
          HANZO_API_KEY: ${{ secrets.HANZO_API_KEY }}
        run: |
          hanzo deploy --env production --yes
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `hanzo: command not found` | Not in PATH | `npm install -g @hanzo/cli` |
| Auth fails | Token expired | `hanzo auth login` |
| Deploy fails | No project linked | `hanzo projects link` |
| Permission denied | Wrong org/role | Check with `hanzo auth whoami` |
| Slow CLI | Network issues | Check `hanzo --ping` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS (CLI deploys here)
- `hanzo/hanzo-kms.md` - Secret management backend
- `hanzo/hanzo-chat.md` - Chat API (CLI calls this)
- `hanzo/hanzo-dev.md` - Terminal AI coding agent (different tool — dev is AI agent, CLI is service management)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: cli, command-line, deploy, management
**Prerequisites**: Node.js 18+, npm
