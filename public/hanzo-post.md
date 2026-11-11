---
name: hanzo-post
description: Run, extend, and automate Hanzo Post (post.hanzo.ai, hanzoai/post) via MCP (post.hanzo.ai/mcp), Hanzo AI drafting, Hanzo CMS publishing, and persistent sandboxed browser social automation (hanzo-browser).
---

# Hanzo Post (`post.hanzo.ai`) & Social Automation

> **Hanzo Post** ([`hanzoai/post`](https://github.com/hanzoai/post) / [`hanzoai/postiz`](https://github.com/hanzoai/postiz), live at `https://post.hanzo.ai` and `https://postiz.hanzo.ai`) is our private, Hanzo-IAM-gated fork of Postiz with native **MCP management** (`/mcp` & `/mcp-oauth-dynamic`), **Hanzo AI** drafting & Autopost (`api.hanzo.ai/v1`), **Hanzo CMS** (`cms.hanzo.ai`) publishing, and **Temporal-backed social scheduling**.

## 1. Architecture & Repository

- **Repositories**: `github.com/hanzoai/post` & `github.com/hanzoai/postiz` (`~/work/hanzo/postiz`)
- **Kubernetes Deployment**: `deployment/postiz` + `deployment/postiz-temporal` in namespace `hanzo` (`~/work/hanzo/universe/charts/app/values/hanzo/postiz.yaml`)
- **Secrets**: Sealed in Hanzo KMS (`hanzo` org `/postiz`, `env: prod` -> `Secret/postiz-secrets`)
- **Authentication**: Hanzo IAM OIDC (`https://hanzo.id`, client `hanzo-postiz`, `DISABLE_REGISTRATION=true`, policy `admin/hanzo-postiz`)
- **AI Copilot & Autopost**: Wired to `https://api.hanzo.ai/v1` via `HANZO_API_KEY` (`CopilotController`, `LoadToolsService`, `OpenaiService`, `AgentGraphService`, `AutopostService`)
- **Media Storage**: Hanzo S3 (`https://s3.hanzo.ai`, bucket `postiz`, `CLOUDFLARE_KEY_PREFIX=media`, served at `https://post.hanzo.ai/uploads/<file>`)

## 2. MCP Management (`https://post.hanzo.ai/mcp`)

Hanzo Post exposes a native Streamable HTTP MCP server on `/mcp` (API key / Bearer auth) and `/mcp-oauth-dynamic` (RFC 9728 / RFC 8414 OAuth 2.1 Dynamic Client Registration).

### Interactive / IDE Agents (Zero Secrets on Disk)

Register in `~/.claude.json` or MCP client config using OAuth 2.1 DCR:

```json
{
  "mcpServers": {
    "hanzo-post": {
      "type": "http",
      "url": "https://post.hanzo.ai/mcp-oauth-dynamic"
    }
  }
}
```

### Headless / CI Agents (KMS-Backed Bearer Auth)

Fetch `POSTIZ_API_KEY` in-memory from Hanzo KMS (never write plaintext keys to disk):

```bash
POSTIZ_KEY="$(hanzo kms secrets get channels/postiz/POSTIZ_API_KEY --env prod | jq -r '.value // .secret.secretValue')"
curl -sS https://post.hanzo.ai/mcp \
  -H "Authorization: Bearer ${POSTIZ_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"integrationList","arguments":{}}}'
```

### Core MCP Tools

- `integrationList` — List channels with live health flags (`connected`, `refreshNeeded`, `disabled`, `inBetweenSteps`, `platform`, `display`). **Only schedule to channels with `connected: true`.**
- `integrationSchema` — Inspect platform rules, character limits, and required settings DTO before scheduling.
- `integrationTriggerTool` — Invoke provider helper methods (e.g., list Discord guild channels, Slack channels, Facebook pages, Subreddits).
- `integrationSchedulePostTool` — Create, draft, schedule, or immediately publish posts/threads across channels.
- `postsListTool` — Query posts in a UTC window (`state`: `QUEUE`, `PUBLISHED`, `ERROR`, `DRAFT`, plus `releaseURL` and `error`).
- `postSettingsTool` / `deletePostTool` — Update provider settings on unpublished posts or delete posts.
- `autopostListTool` / `autopostManageTool` — List, preview RSS/Atom feeds, create, update, toggle (`active`), or delete Autopost workflows.
- `uploadFromUrlTool` / `mediaListTool` / `generateImageTool` / `generateVideoTool` — Upload or generate media attachments.

## 3. Channel Credentials & Reconnection Rules

- **Never seed fake `hanzo-session-*` or `hanzo-browser-session-*` tokens**: Temporal `PostActivity` rejects stub tokens with non-retryable `BadBody` rather than fabricating profile URLs.
- **Direct Bot / App-Password Channels** (headless-ready via KMS):
  - **Bluesky** (`@hanzoai.bsky.social`): ATProto app password encrypted in `customInstanceDetails`.
  - **Discord** (`Hanzo` Guild `445309799979024395`): Bot token `DISCORD_BOT_TOKEN_ID` (`#announcements` `519324670814846976`).
  - **Telegram** (`@HanzoHelpBot`): `TELEGRAM_TOKEN` wired; add `@HanzoHelpBot` as channel admin on `@hanzoai` and run `/connect <code>` in UI.
- **OAuth User-Consent Channels**:
  - **X (`@hanzoai`, `@zeekay`)** and **Slack**: `X_API_KEY`/`X_API_SECRET` and `SLACK_ID`/`SLACK_SECRET` are wired in `postiz-secrets`; click **Reconnect** in `https://post.hanzo.ai/launches` while signed in as the target account.
  - **LinkedIn, Meta (Facebook/Instagram/Threads), YouTube, TikTok, Reddit**: Seal platform OAuth client IDs/secrets into KMS `/orgs/hanzo/postiz` and add env refs in `universe/charts/app/values/hanzo/postiz.yaml` before connecting in `/launches`.

## 4. Persistent Sandboxed Browser (`hanzo-browser`)

When automating social platforms directly through a logged-in browser session (with **1Password**, **LastPass**, and **Hanzo Browser Extension** pre-installed):

```bash
hanzo-browser start              # Launch or resume persistent Chromium on CDP 127.0.0.1:9222
hanzo-browser suspend            # Freeze Chromium in-place (SIGSTOP, 0% CPU)
hanzo-browser resume             # Resume suspended Chromium (SIGCONT)
hanzo-browser stop               # Stop Chromium cleanly (preserves ~/.hanzo/browser-sandbox/profile)
hanzo-browser status             # Verify state, CDP port, and extensions
hanzo-browser sessions           # Inspect logged-in social accounts across 14 networks
hanzo-browser post x "Shipping Hanzo Post at post.hanzo.ai!"
```
