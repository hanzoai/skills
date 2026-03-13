# Hanzo Bot - Multi-Channel Messaging Gateway

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Bot is a **TypeScript ESM multi-channel messaging gateway** supporting 50+ platforms — Discord, Slack, Telegram, WhatsApp, Teams, and more. Features billing, marketplace for skills, and 45 channel extensions. Also hosts the skills framework with 739 public skills on `github.com/hanzoskill`.

**NOTE**: This is NOT just a skills collection — it's a full messaging gateway application with channel adapters, billing, and a skill marketplace.

### Why Hanzo Bot?

- **50+ platforms**: Discord, Slack, Telegram, WhatsApp, Teams, IRC, Matrix, and more
- **TypeScript ESM**: Modern ES module architecture
- **45 channel extensions**: Pre-built adapters for messaging platforms
- **Billing system**: Usage tracking, subscription management
- **Skill marketplace**: Discover, install, and publish reusable skills
- **739 public skills**: Pre-built automations across DevOps, AI/ML, data, security
- **Extensible**: Create custom channels and skills

### History

Rebranded from ClawdBot. All references updated:
- `clawdbot`/`ClawdBot` → `hanzo-bot`/`Hanzo Bot`
- Config: `~/.bot/`, `bot.json`
- Env vars: `BOT_*` (not `CLAWDBOT_*`)
- Dir renames: 25+ directories renamed

## When to use

- Building multi-platform chatbots
- Deploying AI assistants across messaging channels
- Running pre-built automation skills
- Creating reusable automation recipes
- Integrating AI capabilities into messaging platforms

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/bot` |
| Stack | TypeScript ESM |
| Channels | 50+ platforms (45 channel extensions) |
| Skills monorepo | `github.com/hanzobot/skills` |
| Public skill repos | `github.com/hanzoskill` (739 repos) |
| Config | `~/.bot/config.json` |
| Env prefix | `BOT_*` |

## Architecture

```
┌─────────────────────────────────────────────┐
│              Hanzo Bot Gateway               │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │        Channel Adapters (45+)       │    │
│  │  Discord │ Slack │ Telegram │ ...   │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────┴─────────────────────┐    │
│  │         Message Router              │    │
│  │  Intent → Skill matching            │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────┴─────────────────────┐    │
│  │         Skill Engine                │    │
│  │  739 skills │ Marketplace │ Custom  │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────┴─────────────────────┐    │
│  │         Billing & Usage             │    │
│  │  Metering │ Subscriptions │ Limits  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Channel Extensions

| Category | Channels |
|----------|----------|
| **Chat** | Discord, Slack, Telegram, WhatsApp, Signal, Matrix, IRC |
| **Business** | Microsoft Teams, Google Chat, Webex |
| **Social** | Twitter/X, Facebook Messenger, Instagram DM |
| **Support** | Intercom, Zendesk, Freshdesk, LiveChat |
| **Dev** | GitHub (issues/PRs), GitLab, Jira |
| **Voice** | Twilio, Vonage |
| **Web** | WebSocket, HTTP webhook, SSE |
| **Custom** | Plugin API for any platform |

## Quickstart

```bash
git clone https://github.com/hanzoai/bot.git
cd bot
pnpm install

# Configure
cp .env.example .env
# Edit .env with channel tokens (DISCORD_TOKEN, SLACK_TOKEN, etc.)

# Start gateway
pnpm dev

# Or production
pnpm build && pnpm start
```

### Minimal Channel Setup (Discord)

```typescript
import { Bot, DiscordChannel } from "@hanzo/bot"

const bot = new Bot({
  channels: [
    new DiscordChannel({
      token: process.env.DISCORD_TOKEN,
      intents: ["Guilds", "GuildMessages", "MessageContent"],
    }),
  ],
  skills: ["weather", "search", "code-review"],
})

await bot.start()
```

### Multi-Channel Setup

```typescript
import { Bot, DiscordChannel, SlackChannel, TelegramChannel } from "@hanzo/bot"

const bot = new Bot({
  channels: [
    new DiscordChannel({ token: process.env.DISCORD_TOKEN }),
    new SlackChannel({ token: process.env.SLACK_BOT_TOKEN }),
    new TelegramChannel({ token: process.env.TELEGRAM_TOKEN }),
  ],
  billing: {
    enabled: true,
    provider: "hanzo-commerce",  // Hanzo Commerce API
  },
  marketplace: {
    enabled: true,
    registry: "https://api.hanzo.ai/v1/bot/skills",
  },
})
```

## Skills System

### Skill Structure

```
skills/<author>/<skill>/
├── SKILL.md          # Documentation and instructions
├── _meta.json        # Metadata (name, description, triggers, version)
├── scripts/          # Executable scripts
│   ├── run.sh        # Main entry point
│   ├── setup.sh      # One-time setup
│   └── *.py/js/go    # Language-specific scripts
├── .author           # Author info
└── tests/            # Optional tests
```

### Skill Categories

| Category | Count | Examples |
|----------|-------|---------|
| DevOps | ~200 | Docker, K8s, CI/CD, monitoring |
| Development | ~150 | Code generation, linting, testing |
| AI/ML | ~100 | Model serving, training, inference |
| Data | ~80 | ETL, databases, migrations |
| Security | ~60 | Scanning, auditing, secrets |
| Cloud | ~50 | AWS, GCP, DO provisioning |
| Misc | ~100 | Formatting, documentation, bots |

### Marketplace

```bash
# Search skills in marketplace
hanzo-bot marketplace search "kubernetes deploy"

# Install skill from marketplace
hanzo-bot marketplace install k8s-deployer

# Publish your skill
hanzo-bot marketplace publish ./my-skill/

# Rate a skill
hanzo-bot marketplace rate k8s-deployer 5
```

## Billing

```typescript
// Billing configuration
const billing = {
  plans: [
    { name: "free", messages: 100, skills: 5 },
    { name: "pro", messages: 10000, skills: "unlimited" },
    { name: "enterprise", messages: "unlimited", skills: "unlimited" },
  ],
  metering: {
    track: ["messages", "skill_runs", "api_calls"],
    provider: "hanzo-commerce",
  },
}
```

## GitHub Organization (hanzoskill)

The `hanzoskill` org on GitHub has 739 public repos — one per skill. Published from the monorepo via CI.

**Rate limits**: GitHub secondary rate limit allows ~150 repo creations per burst, ~30-40min cooldown between bursts.

**Duplicate handling**: If a skill name exists in multiple authors, the second is suffixed (e.g., `discord-doctor-alt`).

## Related Skills

- `hanzo/hanzo-agent.md` - AI agent framework (bots can use agents)
- `hanzo/hanzo-mcp.md` - MCP tools (skills can expose MCP tools)
- `hanzo/hanzo-dev.md` - AI coding agent (uses skill discovery)
- `hanzo/hanzo-commerce-api.md` - Billing backend

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: bot, messaging, gateway, channels, skills, marketplace
**Prerequisites**: TypeScript, messaging platform tokens
