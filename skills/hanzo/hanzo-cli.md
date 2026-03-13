# Hanzo CLI - Command-Line Interface

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-chat.md`

## Overview

Hanzo CLI provides command-line access to all Hanzo services — deploy apps, manage environments, chat with models, and manage API keys from your terminal.

## Quick reference

| Item | Value |
|------|-------|
| Install | `npm install -g @hanzo/cli` |
| Repo | `github.com/hanzoai/cli` |
| Local | ``github.com/hanzoai/cli`` |

## Commands

```bash
hanzo auth login          # Authenticate via browser
hanzo auth status         # Check auth status

hanzo deploy              # Deploy current directory
hanzo deploy --env prod   # Deploy to production
hanzo env list            # List environments
hanzo env set KEY=VALUE   # Set environment variable

hanzo chat "Hello"        # Quick chat with default model
hanzo chat --model zen-70b "Explain consensus"
hanzo models list         # List available models

hanzo keys create         # Create API key
hanzo keys list           # List API keys
hanzo keys revoke KEY_ID  # Revoke API key
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
