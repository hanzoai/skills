# Hanzo Bot - Multi-Channel AI Messaging Gateway

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Bot (`@hanzo/bot`) is a **TypeScript ESM multi-channel AI messaging gateway** that connects AI agents to messaging platforms through a plugin/extension architecture. It provides first-class integrations for Discord, Telegram, Slack, WhatsApp, Signal, iMessage, Line, MS Teams, IRC, Matrix, and many more via 40+ bundled extensions. It also includes a TUI, web UI, iOS/Android/macOS native apps, and an Agent Client Protocol (ACP) gateway.

Fork of OpenClaw/ClawdBot, fully rebranded. Package: `@hanzo/bot`, binary: `hanzo-bot`.

### What it actually is

- A multi-channel AI agent runtime -- receives messages from platforms, routes them through an AI agent, sends responses back
- Plugin SDK (`@hanzo/bot/plugin-sdk/*`) with per-channel exports for Discord, Telegram, Slack, WhatsApp, Signal, iMessage, Line, MS Teams, IRC, Matrix, Feishu, Google Chat, Mattermost, Twitch, Nostr, Zalo, Nextcloud Talk, Synology Chat, Tlon, Lobster, and more
- ACP (Agent Client Protocol) gateway for cross-platform agent coordination
- TUI mode (`pnpm tui`) and web UI (`pnpm ui:dev`)
- iOS, Android, and macOS native host apps (Swift/Kotlin)
- Memory system with LanceDB vector store
- Context engine, media understanding, link understanding, browser automation
- Commerce module in `src/commerce/`

### What it is NOT

- Not a billing platform -- no subscription management or metering system
- Not a skill marketplace -- skills are in separate repos (`github.com/hanzobot/skills`, `github.com/hanzoskill`)
- Skills are not bundled in the bot runtime

## When to use

- Deploying AI agents across multiple messaging platforms
- Building multi-channel chatbots with a single codebase
- Running an AI assistant on Discord, Telegram, Slack, or WhatsApp
- Need an ACP-compatible agent gateway

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/bot` |
| Package | `@hanzo/bot` (npm) |
| Version | 2026.3.x (calver) |
| Stack | TypeScript ESM, pnpm, Node.js >= 22 |
| Binary | `hanzo-bot` |
| Dev | `pnpm dev` |
| Build | `pnpm build` |
| Test | `pnpm test` |
| TUI | `pnpm tui` |
| Web UI | `pnpm ui:dev` |
| Formatter | oxfmt |
| Linter | oxlint (type-aware) |
| License | MIT |

## Channel Extensions (extensions/)

40+ extensions in the `extensions/` directory, each a self-contained plugin:

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
| **Auth** | google-gemini-cli-auth, minimax-portal-auth, qwen-portal-auth, google-antigravity-auth |
| **Infra** | diagnostics-otel, device-pair, acpx, thread-ownership, phone-control, open-prose, test-utils, shared |

## Source structure (src/)

| Directory | Purpose |
|-----------|---------|
| `src/agents/` | AI agent runtime |
| `src/channels/` | Channel framework (registry, config, session, typing, reactions) |
| `src/discord/` | Discord-specific channel logic |
| `src/telegram/` | Telegram-specific channel logic |
| `src/slack/` | Slack-specific channel logic |
| `src/whatsapp/` | WhatsApp (Baileys) integration |
| `src/signal/` | Signal integration |
| `src/imessage/` | iMessage integration |
| `src/line/` | LINE integration |
| `src/web/` | Web channel |
| `src/gateway/` | ACP gateway |
| `src/tui/` | Terminal UI |
| `src/cli/` | CLI commands |
| `src/memory/` | Conversation memory |
| `src/context-engine/` | Context management |
| `src/providers/` | LLM providers |
| `src/commerce/` | Commerce module |
| `src/browser/` | Browser automation |
| `src/media-understanding/` | Media processing |
| `src/plugin-sdk/` | Plugin SDK exports |

## Key dependencies

- `grammy` -- Telegram Bot API
- `@discordjs/voice` + `discord-api-types` -- Discord
- `@slack/bolt` + `@slack/web-api` -- Slack
- `@whiskeysockets/baileys` -- WhatsApp
- `@line/bot-sdk` -- LINE
- `playwright-core` -- Browser automation
- `sqlite-vec` -- Vector search
- `@agentclientprotocol/sdk` -- ACP
- `@mariozechner/pi-agent-core` / `pi-coding-agent` / `pi-tui` -- Upstream agent runtime

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

# TUI mode
pnpm tui
```

## Skills (separate repos)

Skills live outside this repo:
- **Monorepo**: `github.com/hanzobot/skills` -- full collection with history
- **Per-skill repos**: `github.com/hanzoskill` org (739 public repos, one per skill)
- Each skill has: `SKILL.md`, `_meta.json`, `scripts/`, `.author`

## History

Rebranded from ClawdBot/OpenClaw. All internal references updated:
- `clawdbot`/`ClawdBot` -> `hanzo-bot`/`Hanzo Bot`
- Config: `~/.bot/`, `bot.json`
- Env vars: `BOT_*` (not `CLAWDBOT_*`)
- Some internal OpenClaw references remain in scripts and Android package names

## Related Skills

- `hanzo/hanzo-agent.md` -- AI agent framework
- `hanzo/hanzo-mcp.md` -- MCP tools (extensions can expose MCP tools)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: bot, messaging, gateway, channels, discord, telegram, slack, whatsapp
**Prerequisites**: Node.js >= 22, pnpm, messaging platform tokens
