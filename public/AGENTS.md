# Hanzo — Progressive Agent Instructions (`CLAUDE.md`)

> Canonical drop-in `CLAUDE.md` / `AGENTS.md` for Claude Code, Codex, Cursor, Windsurf, Gemini CLI, Antigravity, or any AI agent.
> Live at: https://hanzoskills.com/CLAUDE.md · https://hanzo.ai/CLAUDE.md · Docs: https://docs.hanzo.ai/docs/skills

## Progressive Skill Loading (3 Tiers — Never Load Everything Up Front)

Hanzo publishes **1477 plain-markdown skills** across **14 categories** over HTTPS with zero auth.
Keep context small by loading skills **only when a task requires them**:

1. **Tier 0 (Category Tree, ~4 KB)**: `curl -fsSL https://hanzoskills.com/llms.txt` (or `https://hanzoskills.com/catalog.json`).
2. **Tier 1 (Single Category)**: Fetch ONLY the matching category `https://hanzoskills.com/categories/<slug>.md` (or rank via `POST https://api.hanzo.ai/v1/tool/detect`).
3. **Tier 2 (Single Skill)**: Fetch ONLY the skill you need via `https://hanzoskills.com/skills/<id>/SKILL.md` (or `/<id>.md`).
   - Claude Code local install on demand: save to `~/.claude/skills/<id>/SKILL.md`.
   - Never fetch `/all.md` (16 MB) into an active agent context.

## Forge Open Source Repositories (Progressive Discovery)

Hanzo hosts thousands of open-source repositories across Hanzo, Lux, Zoo, Zen, and ZAP on **https://forge.hanzo.ai**.
Discover and clone repos progressively instead of indexing everything up front:

1. **Tier 0 (Org Directory)**: `curl -fsSL https://hanzoskills.com/oss/index.md` (or `/oss/index.json`).
2. **Tier 1 (Single Org)**: `curl -fsSL https://hanzoskills.com/oss/<org>.md` (e.g. `hanzoai`, `hanzoskill`, `luxfi`, `zooai`, `zenlm`, `zap-proto`).
3. **Tier 2 (Single Repo)**: `git clone https://forge.hanzo.ai/<org>/<repo>.git`

## Quickstart: CLI, MCP & Free API Onboarding

- **Install native Hanzo CLI + MCP + Dev agent**: `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo` / `npx -y hanzo mcp`).
- **Headless agent signup & free API key (Enso, Zen, Kai, no human required)**: read https://hanzoskills.com/hanzo.md.
- **Local-first runtime manifest**: read https://hanzoskills.com/skill.md.
- **Architecture & HIP specs**: https://hips.hanzo.ai/llms.txt.

## Canonical Links

- Skills Registry: https://hanzoskills.com
- Progressive `CLAUDE.md`: https://hanzoskills.com/CLAUDE.md
- Documentation: https://docs.hanzo.ai (Skills guide: https://docs.hanzo.ai/docs/skills)
- Forge OSS: https://forge.hanzo.ai (Index: https://hanzoskills.com/oss/index.md)
- Platform & Cloud: https://hanzo.ai · https://api.hanzo.ai/v1
