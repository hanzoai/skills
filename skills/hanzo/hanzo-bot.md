# Hanzo Bot - Multi-Channel AI Messaging Gateway

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-playground.md`

## Overview

Hanzo Bot (`@hanzo/bot`) is a **TypeScript ESM multi-channel AI messaging gateway** connecting AI agents to 50+ messaging platforms through a plugin/extension architecture. First-class integrations for Discord, Telegram, Slack, WhatsApp, Signal, iMessage, Line, MS Teams, IRC, Matrix, and more. Includes TUI, web UI, native apps (iOS/Android/macOS), and an Agent Client Protocol (ACP) gateway.

Fork of OpenClaw/ClawdBot, fully rebranded. Package: `@hanzo/bot`, binary: `hanzo-bot`.

## When to use

- Deploying AI agents across multiple messaging platforms from a single codebase
- Running an AI assistant on Discord, Telegram, Slack, or WhatsApp
- Building a multi-channel chatbot with conversation memory
- Need an ACP-compatible agent gateway
- Connecting team channels (hanzo.team) to LLM proxy

## Hard requirements

1. **Node.js >= 22** with ESM support
2. **pnpm** for package management
3. **Platform tokens** for each channel (DISCORD_TOKEN, TELEGRAM_TOKEN, etc.)
4. **LLM backend**: Connects to Hanzo LLM Gateway (`api.hanzo.ai/v1`) or any OpenAI-compatible endpoint

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/bot` |
| Package | `@hanzo/bot` (npm) |
| Version | 2026.3.x (calver) |
| Stack | TypeScript ESM, pnpm, Node.js >= 22 |
| Binary | `hanzo-bot` |
| Image | `ghcr.io/hanzoai/bot:latest` |
| Dev | `pnpm dev` |
| Build | `pnpm build` |
| Test | `pnpm test` |
| TUI | `pnpm tui` |
| Web UI | `pnpm ui:dev` |
| Formatter | oxfmt |
| Linter | oxlint (type-aware) |
| K8s manifests | `universe/infra/k8s/bot/` |

## Architecture

```
                 Messaging Platforms
          Discord | Telegram | Slack | ...
                    |
              Extension Layer (50+)
                    |
              +-----+-----+
              |           |
         Channel      ACP Gateway
         Framework    (cross-agent)
              |           |
         Agent Runtime    |
              |           |
         +----+----+     |
         |    |    |     |
       Memory Context LLM
       (LanceDB) Engine Proxy
                         |
                  api.hanzo.ai/v1
```

## Channel extensions (extensions/)

50+ extensions in the `extensions/` directory:

| Category | Extensions |
|----------|-----------|
| **Chat** | discord, telegram, slack, whatsapp, signal, irc, matrix, mattermost |
| **Asian** | line, feishu, zalo, zalouser |
| **Enterprise** | msteams, googlechat, synology-chat, nextcloud-talk |
| **Social** | twitch, nostr, tlon, lobster |
| **Apple** | imessage, bluebubbles |
| **Voice** | voice-call, talk-voice |
| **Dev** | copilot-proxy, diffs, ci-fix-loop, flow |
| **AI/Memory** | memory-core, memory-lancedb, llm-task, continuous-learning, self-improvement |
| **Infra** | diagnostics-otel, device-pair, acpx, thread-ownership, phone-control |

## Source structure

```
hanzoai/bot/
  src/
    agents/           # AI agent runtime
    channels/         # Channel framework (registry, config, session)
    discord/          # Discord-specific logic
    telegram/         # Telegram-specific logic
    slack/            # Slack-specific logic
    whatsapp/         # WhatsApp (Baileys) integration
    signal/           # Signal integration
    imessage/         # iMessage integration
    line/             # LINE integration
    web/              # Web channel
    gateway/          # ACP gateway
    tui/              # Terminal UI
    cli/              # CLI commands
    memory/           # Conversation memory
    context-engine/   # Context management
    providers/        # LLM providers
    commerce/         # Commerce module
    browser/          # Browser automation (Playwright)
    media-understanding/  # Media processing
    plugin-sdk/       # Plugin SDK exports
  extensions/         # 50+ extension plugins
  apps/
    ios/              # Swift iOS app
    android/          # Kotlin Android app
    macos/            # macOS app
```

## Key dependencies

| Package | Purpose |
|---------|---------|
| `grammy` | Telegram Bot API |
| `@discordjs/voice` + `discord-api-types` | Discord |
| `@slack/bolt` + `@slack/web-api` | Slack |
| `@whiskeysockets/baileys` | WhatsApp |
| `@line/bot-sdk` | LINE |
| `playwright-core` | Browser automation |
| `sqlite-vec` | Vector search |
| `@agentclientprotocol/sdk` | ACP |

## Quickstart

```bash
git clone https://github.com/hanzoai/bot.git
cd bot
pnpm install

# Configure channel tokens
cp .env.example .env
# Edit .env: DISCORD_TOKEN, TELEGRAM_TOKEN, SLACK_BOT_TOKEN, etc.

# Development
pnpm dev

# Production
pnpm build && pnpm start

# TUI mode (interactive terminal)
pnpm tui

# Web UI
pnpm ui:dev
```

## Playground registration

Bot instances register with the Hanzo Playground control plane (`app.hanzo.bot`) for:
- Node registry and health monitoring
- Execution tracking and audit
- Workflow orchestration across multiple bots
- Memory scope management (global, bot, session, run)

## K8s deployment

```yaml
# universe/infra/k8s/bot/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-bot
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: bot
          image: ghcr.io/hanzoai/bot:latest
          envFrom:
            - secretRef:
                name: bot-secrets  # KMS-synced
          env:
            - name: LLM_BASE_URL
              value: http://llm.hanzo.svc:4000/v1
```

## Skills (separate repos)

Skills live outside the bot repo:
- **Monorepo**: `github.com/hanzobot/skills` -- full collection with history
- **Per-skill repos**: `github.com/hanzoskill` org (739 public repos, one per skill)
- Each skill has: `SKILL.md`, `_meta.json`, `scripts/`, `.author`

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Bot not responding | Missing channel token | Check `.env` for platform-specific token |
| LLM timeout | Gateway unreachable | Verify `LLM_BASE_URL` points to working gateway |
| Memory not persisting | LanceDB not initialized | Check `memory-lancedb` extension is enabled |
| Discord rate limited | Too many messages | Bot auto-handles rate limits via discord.js |

## Related Skills

- `hanzo/hanzo-playground.md` -- Bot control plane
- `hanzo/hanzo-agent.md` -- AI agent framework
- `hanzo/hanzo-mcp.md` -- MCP tools (extensions can expose MCP tools)
- `hanzo/hanzo-team.md` -- Team platform integration

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: bot, messaging, gateway, channels, discord, telegram, slack, whatsapp, acp
**Prerequisites**: Node.js >= 22, pnpm, messaging platform tokens
