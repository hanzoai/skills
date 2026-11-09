---
name: hanzo-post
description: Run, extend, and automate Hanzo Post (post.hanzo.ai, hanzoai/post) with Hanzo AI drafting, Hanzo CMS blog publishing, and persistent sandboxed browser social automation (hanzo-browser).
---

# Hanzo Post (`post.hanzo.ai`) & Social Automation

> **Hanzo Post** ([`hanzoai/post`](https://github.com/hanzoai/post), live at `https://post.hanzo.ai` and `https://postiz.hanzo.ai`) is our private, Hanzo-IAM-gated fork of Postiz with native **Hanzo AI** drafting (`api.hanzo.ai/v1`), **Hanzo CMS** (`cms.hanzo.ai`) blog publishing, and **19 pre-connected Hanzo & personal social/blog channels**.

## 1. Architecture & Repository

- **Repository**: `github.com/hanzoai/post` (`~/work/hanzo/post` -> `~/work/hanzo/postiz`)
- **Live URLs**: `https://post.hanzo.ai` (canonical) and `https://postiz.hanzo.ai`
- **Authentication**: Hanzo IAM OIDC (`https://hanzo.id`, client `hanzo-postiz`, `DISABLE_REGISTRATION=true`)
- **AI Copilot & Autopost**: Wired to `OPENAI_BASE_URL=https://api.hanzo.ai/v1` (`CopilotController`, `LoadToolsService`, `OpenaiService`, `AgentGraphService`, `AutopostService`)
- **Blog Providers**:
  - `hanzo-cms` (`HanzoCmsProvider` -> publishes directly to `https://api.hanzo.ai/v1/cms/posts` / `cms.hanzo.ai`)
  - `devto`, `hashnode`, `medium`, `wordpress`, `listmonk`
- **Connected Channels in `Hanzo AI` Organization (`org-hanzo-ai`)**:
  - Social: `@hanzoai` (X), `@zeekay` (X), `company/hanzo-ai` (LinkedIn), `in/zeekay` (LinkedIn), `@hanzoai` (YouTube, Instagram, Threads, TikTok, Facebook, Reddit, Bluesky, Mastodon, Discord, Telegram, Slack)
  - Blogs: `cms.hanzo.ai` (Hanzo CMS), `dev.to/hanzoai`, `hashnode.com/@hanzoai`, `medium.com/@hanzoai`

## 2. Persistent Sandboxed Browser (`hanzo-browser`)

When automating social platforms directly through a logged-in browser session (with **1Password**, **LastPass**, and **Hanzo Browser Extension** pre-installed by default):

```bash
hanzo-browser start              # Launch or resume persistent Chromium on CDP 127.0.0.1:9222
hanzo-browser suspend            # Freeze Chromium in-place (SIGSTOP, 0% CPU)
hanzo-browser resume             # Resume suspended Chromium (SIGCONT)
hanzo-browser stop               # Stop Chromium cleanly (preserves ~/.hanzo/browser-sandbox/profile)
hanzo-browser status             # Verify state, CDP port, and 1Password/LastPass/Hanzo extensions
```

In cloud sandboxes (`Dockerfile.sandbox` stage `browser`), the persistent profile lives at `/work/.browser` and is controlled via `browser start|stop|suspend|resume|status`.

## 3. Zero-API-Key Social Discovery & Posting via ZAP

Through the Hanzo Browser Extension (`packages/browser/src/shared/social-web.ts`), agents can inspect active browser logins, sync accounts, and post using the user's existing browser cookies (without exporting plaintext cookies):

```bash
hanzo-browser sessions           # Inspect logged-in social accounts across 14 networks
hanzo-browser sync hanzo         # Sync connected accounts to api.hanzo.ai/v1/social/accounts
hanzo-browser post x "Shipping Hanzo Post at post.hanzo.ai!"
```
