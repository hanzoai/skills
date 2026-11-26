# Hanzo — Complete Progressive Agent Instructions (`CLAUDE.md`)

> Canonical drop-in `CLAUDE.md` / `AGENTS.md` for Claude Code, Codex, Cursor, Windsurf, Gemini CLI, Antigravity, Copilot, Hanzo Dev, Hanzo Bot, or any AI agent.
> Live at: https://hanzoskills.com/CLAUDE.md · https://hanzo.ai/CLAUDE.md · Docs: https://docs.hanzo.ai/docs/skills

## 1. Progressive Skill Loading (3 Tiers — Never Load Everything Up Front)

Hanzo publishes **1477 plain-markdown skills** across **14 categories** over HTTPS with zero auth.
Keep context small by loading skills **only when a task requires them**:

1. **Tier 0 (Category Tree, ~8 KB)**: `curl -fsSL https://hanzoskills.com/llms.txt` (or `/catalog.json`).
2. **Tier 1 (Single Category)**: Fetch ONLY the matching category `https://hanzoskills.com/categories/<slug>.md` (or rank via `POST https://api.hanzo.ai/v1/tool/detect`).
3. **Tier 2 (Single Skill)**: Fetch ONLY the skill you need via `https://hanzoskills.com/skills/<id>/SKILL.md` (or `/<id>.md`).
   - Never fetch `/all.md` (16 MB) into an active agent context.

## 2. Native Agent Setup (Use Each Agent's Native Instructions, Skills & MCP)

| Agent | Native Instructions File | Native Skills Path | Native MCP Setup |
| --- | --- | --- | --- |
| **Claude Code** | `./CLAUDE.md` & `~/.claude/CLAUDE.md` | `~/.claude/skills/<id>/SKILL.md` | `claude mcp add --scope user hanzo -- npx -y hanzo mcp` |
| **Codex** | `./AGENTS.md` & `~/.codex/AGENTS.md` | `~/.codex/skills/<id>/SKILL.md` | `codex mcp add hanzo -- npx -y hanzo mcp` |
| **Cursor** | `.cursor/rules/hanzo.mdc` (`alwaysApply: true`) | `.cursor/rules/<id>.mdc` (`alwaysApply: false`) | `.cursor/mcp.json` (`hanzo` -> `npx -y hanzo mcp`) |
| **Windsurf** | `.windsurfrules` / `.windsurf/rules/hanzo.md` | `.windsurf/rules/<id>.md` | `~/.codeium/windsurf/mcp_config.json` |
| **Gemini CLI / Antigravity** | `./GEMINI.md` & `~/.gemini/GEMINI.md` | `~/.gemini/antigravity/skills/<id>/SKILL.md` | `~/.gemini/settings.json` (`mcpServers.hanzo`) |
| **GitHub Copilot** | `.github/copilot-instructions.md` | `.github/instructions/<id>.instructions.md` | `.vscode/mcp.json` |
| **Hanzo Dev** | Native (`AGENTS.md` / `CLAUDE.md`) | `<repo>/.agents/skills/<id>/SKILL.md` or `~/.agents/skills/` | Built-in native MCP + ZAP |
| **Hanzo Bot** | Native (`hanzo.bot`) | `<workspace>/skills/` or `~/.hanzo/bot/skills/<id>/SKILL.md` | Built-in native MCP + browser sandbox |

## 3. Hanzo MCP (Unified Tool Surface — Local Stdio & Hosted HTTP)

- **Local stdio (recommended, zero pre-install)**: `npx -y hanzo mcp` (or `hanzo mcp` after installing the CLI).
  Exposes the **13 HIP-0300 unified tools** (`fs`, `exec`, `code`, `git`, `fetch`, `workspace`, `computer`, `think`, `memory`, `hanzo`, `plan`, `tasks`, `ui`) plus your linked cloud connectors (GitHub, Slack, Google, Discord, X).
- **JSON MCP config** (Cursor / Claude Desktop / Windsurf / Gemini CLI / Antigravity):
  `{"mcpServers":{"hanzo":{"command":"npx","args":["-y","hanzo","mcp"]}}}`
- **Hosted Cloud MCP (Streamable HTTP, zero local setup)**: `https://api.hanzo.ai/v1/mcp` (`Authorization: Bearer $HANZO_API_KEY`).
- **MCP schemas (progressive)**: `https://hanzoskills.com/mcp/index.json` -> `/mcp/<tool>.json` · Docs: https://docs.hanzo.ai/docs/mcp

## 4. Hanzo Browser Extension (Chrome, Firefox, Safari, Edge)

Install the Hanzo Browser Extension so your agent gets live browser control (tabs, CDP, DOM snapshot, click, type, screenshot, social sessions) bridged over ZAP/MCP:

- **Install & Downloads**: https://hanzo.ai/extension · https://hanzo.ai/chrome · Latest zips: https://github.com/hanzoai/extension/releases/latest
- **Chrome / Chromium unpacked load**: download `hanzo-ai-chrome-*.zip` -> unzip -> open `chrome://extensions` -> enable **Developer mode** -> **Load unpacked** -> sign in via Hanzo ID (`https://hanzo.id`).
- **Skill**: `https://hanzoskills.com/skills/hanzo-extension/SKILL.md`

## 5. Hanzo CLI, Hanzo Dev & Hanzo Bot Agents

- **Install native stack** (CLI, Dev agent, MCP server, local inference engine): `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo`), then `hanzo auth login`.
- **Hanzo Dev (coding agents)**: `hanzo dev` (native Rust coding agent), `hanzo code claude`, `hanzo code codex`, `hanzo engine serve <model>` (local inference), `hanzo up` (run Hanzo Cloud locally). Skill: `/skills/hanzo-dev/SKILL.md`.
- **Hanzo Bot (resident autonomous agents — Slack, Discord, Telegram, WhatsApp, iMessage, browser + terminal + cron)**:
  - Cloud (unlimited bots on paid plans): https://hanzo.bot · CLI: `hanzo bot runs list`
  - Local open-source daemon: `npm i -g @hanzo/bot && hanzo-bot onboard --install-daemon`
  - Skill: `https://hanzoskills.com/skills/hanzo-bot/SKILL.md` · Docs: https://docs.hanzo.bot

## 6. Full Hanzo Platform Fluency (`https://api.hanzo.ai/v1`)

One OpenAI-compatible API, one key, same `/v1/<capability>` contract across cloud and local self-host:

- **Discover everything (zero auth)**: `GET https://api.hanzo.ai/v1`, `GET /v1/<capability>`, `GET /v1/openapi.json`.
- **AI & Models**: `/v1/chat/completions` (`enso-auto`, `zen-free`, `zen-*`, 500+ models), **Kai** typed calibrated decisions (`POST /v1/decisions`), embeddings, audio/video/image.
- **Cloud & Data**: IAM (`https://hanzo.id`), KMS (`/v1/kms`), Base realtime DB (`/v1/base`), SQL (`/v1/sql`), KV (`/v1/kv`), Vector (`/v1/vector`), S3/Storage (`/v1/storage`), PubSub, Tasks, Pipelines, Sandboxes, Deploy, Billing/USDC top-up, CRM, CMS, Post/Social.
- **Headless agent signup & free API key (no human required)**: read https://hanzoskills.com/hanzo.md (or https://docs.hanzo.ai/docs/agents).
- **Local-first runtime manifest**: https://hanzoskills.com/skill.md · **HIPs**: https://hips.hanzo.ai/llms.txt

## 7. Forge Open Source Repositories (2,800+ Public Repos — Progressive Discovery)

1. **Tier 0 (Org Directory)**: `curl -fsSL https://hanzoskills.com/oss/index.md` (or `/oss/index.json`).
2. **Tier 1 (Single Org)**: `curl -fsSL https://hanzoskills.com/oss/<org>.md` (e.g. `hanzoai`, `hanzoskill`, `luxfi`, `zooai`, `zenlm`, `zap-proto`).
3. **Tier 2 (Single Repo)**: `git clone https://forge.hanzo.ai/<org>/<repo>.git` (every public repo also serves `/<org>/<repo>/CLAUDE.md`).
