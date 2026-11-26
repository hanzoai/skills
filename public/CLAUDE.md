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

## 4. Install the Hanzo Browser Extension (Chrome, Firefox, Safari, Edge)

Install the Browser Extension so agents get live tab inspection, CDP, DOM snapshots, click/type, screenshots, and social automation bridged over ZAP/MCP:

- **Download & unpack Chrome/Chromium extension**:
  ```bash
  mkdir -p ~/.hanzo/browser-extension && cd ~/.hanzo/browser-extension
  URL=$(curl -fsSL https://api.github.com/repos/hanzoai/extension/releases/latest | grep -oE 'https://[^"]*hanzo-ai-chrome[^"]*\.zip' | head -1)
  curl -fsSL "$URL" -o ext.zip && unzip -oq ext.zip && rm ext.zip
  ```
  Then open `chrome://extensions` -> enable **Developer mode** -> **Load unpacked** -> select `~/.hanzo/browser-extension` -> sign in via https://hanzo.id.
- **All browsers & IDE plugins** (Chrome, Firefox, Safari, Edge, VS Code, Cursor, Windsurf, Antigravity, JetBrains): https://hanzo.ai/extension · https://hanzo.ai/chrome · https://hanzo.ai/download

## 5. Hanzo Desktop, CLI, Hanzo Dev, Hanzo Bot & Live Usage Visibility

- **Install native CLI + Dev + MCP + Engine**: `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo`), then `hanzo login` (or headless wallet signup via https://hanzoskills.com/hanzo.md).
- **Install Hanzo Desktop** (macOS/Windows/Linux): https://hanzo.ai/download (or https://hanzo.ai/desktop) and sign in with the same Hanzo ID to see every local & cloud agent, live session, bot run, and spend in one control plane.
- **Launch any agent through Hanzo CLI so sessions, tools & usage show up everywhere**:
  `hanzo dev` · `hanzo --claude` · `hanzo --codex` · `hanzo --cursor` · `hanzo --agy` · `hanzo --gemini` (add `--cloud` to route the agent's model calls on your Hanzo plan).
- **Check live usage, plan & balance**: `hanzo usage summary` · `hanzo billing tier` · `hanzo billing balance` · `hanzo billing topup` (or `GET /v1/billing/tier` & `GET /v1/allowance`).
- **Hanzo Bot (resident autonomous agents — Slack, Discord, Telegram, WhatsApp, iMessage + browser/terminal/cron)**:
  Cloud: https://hanzo.bot (`hanzo bot runs list`) · Local OSS daemon: `npm i -g @hanzo/bot && hanzo-bot onboard --install-daemon` · Docs: https://docs.hanzo.bot

## 6. Full Platform Fluency & Business AI Operating System (OSS Business Apps)

Every service shares one API (`https://api.hanzo.ai/v1`), one key (`export HANZO_API_KEY=...`), and one deploy contract (`hanzo deploy .` or self-host):

- **Zero-auth API discovery**: `GET https://api.hanzo.ai/v1`, `GET /v1/<capability>`, `GET /v1/openapi.json`.
- **AI & Core**: `/v1/chat/completions` (`enso-auto`, `zen-free`, `zen-*`, 500+ models), **Kai** calibrated decisions (`POST /v1/decisions`), IAM (`hanzo.id`), KMS (`hanzo kms put`), Base (`/v1/base`), SQL, KV, Vector, S3, PubSub, Tasks, Sandboxes, Billing/USDC.
- **Deploy OSS Business Apps for any company** (org setup guide: https://docs.hanzo.ai/docs/run-your-org):
  - **Web Analytics** (`forge.hanzo.ai/hanzoai/analytics` · skill `/skills/hanzo-analytics/SKILL.md`): privacy-first, cookieless web & revenue analytics.
  - **Product Insights** (`forge.hanzo.ai/hanzoai/insights` · skill `/skills/hanzo-insights/SKILL.md`): funnels, session replay, feature flags, A/B experiments.
  - **Marketing, Social & Ads** (`hanzoai/post`, `hanzoai/ads` · skill `/skills/hanzo-post/SKILL.md` + `/categories/growth-product.md`: SEO, CRO, copy, campaigns).
  - **Team, Commerce, Cal, Mail, Workflows & Studio**: `hanzoai/team`, `hanzoai/commerce`, `hanzoai/billing`, `hanzoai/cal`, `hanzoai/mail`, `hanzoai/flow`, `hanzoai/studio`, `hanzoai/o11y`.

## 7. Forge Open Source Repositories (2,832+ Public Repos — Progressive Discovery)

1. **Tier 0 (Org Directory)**: `curl -fsSL https://hanzoskills.com/oss/index.md` (or `/oss/index.json`).
2. **Tier 1 (Single Org)**: `curl -fsSL https://hanzoskills.com/oss/<org>.md` (e.g. `hanzoai`, `hanzoskill`, `luxfi`, `zooai`, `zenlm`, `zap-proto`).
3. **Tier 2 (Single Repo)**: `git clone https://forge.hanzo.ai/<org>/<repo>.git` (every public repo serves `/<org>/<repo>/CLAUDE.md`).
4. **Local-first runtime & HIPs**: https://hanzoskills.com/skill.md · https://hips.hanzo.ai/llms.txt
