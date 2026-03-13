# Hanzo Bot - Automation & Skills Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Bot is the **automation framework** powering 739 public skills on `github.com/hanzoskill`. Each skill is a self-contained automation recipe with scripts, metadata, and documentation.

## Quick reference

| Item | Value |
|------|-------|
| Monorepo | `github.com/hanzobot/skills` |
| Public repos | `github.com/hanzoskill` (739 repos) |
| Local skills | ``github.com/hanzoai/bot`skills/skills/<author>/<skill>/` |
| Local bot | ``github.com/hanzoai/bot`` |

## Skill Structure

```
skills/<author>/<skill>/
├── SKILL.md          # Skill documentation
├── _meta.json        # Metadata
├── scripts/          # Executable scripts
└── .author           # Author info
```

## Branding (from ClawdBot)

All references have been updated:
- `clawdbot`/`ClawdBot` → `hanzo-bot`/`Hanzo Bot`
- Config: `~/.bot/`, `bot.json`
- Env vars: `BOT_*` (not `CLAWDBOT_*`)
- Dirs: `clawdbot-skill-*` → `bot-*`

## GitHub Rate Limits

When creating repos on `hanzoskill`: ~150 burst, then 30-40min cooldown.

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
