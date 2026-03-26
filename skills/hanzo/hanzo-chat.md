# Hanzo Chat

AI chat interface with multi-model support, MCP integration, agents, and RAG.

## URLs
- **Production**: hanzo.chat, chat.hanzo.ai
- **IAM Client ID**: hanzo-chat
- **Docker Image**: ghcr.io/hanzoai/chat:main

## Tech Stack
- **Runtime**: Node.js 20 (Alpine)
- **Backend**: Express.js (api/server/)
- **Frontend**: React + Vite (client/)
- **Database**: MongoDB (DocumentDB)
- **Search**: Meilisearch
- **Package Manager**: pnpm 10.x (Dockerfile uses pnpm install --frozen-lockfile)

## Key Files
- `api/server/` — Express backend (port 3080)
- `api/strategies/openidStrategy.js` — OIDC SSO via hanzo.id (Host header injection for internal IAM rewrite)
- `client/` — React frontend (Vite)
- `packages/api/` — API client (@librechat/api)
- `packages/mcp/` — MCP server integration
- `librechat.yaml` / ConfigMap `chat-config` — runtime config

## Auth
- SSO via hanzo.id OIDC (`OPENID_ISSUER=https://hanzo.id`, `OPENID_CLIENT_ID=hanzo-chat`)
- `/oauth/openid` redirects to hanzo.id (302)
- `ALLOW_EMAIL_LOGIN=false`, `ALLOW_REGISTRATION=false`
- Social login: Google, GitHub, Wallet

## Build & Deploy
```bash
pnpm install --frozen-lockfile
pnpm run build:packages
pnpm run frontend
```
- K8s: deployment/chat, port 3080
- CI: `docker-publish.yml` → shared docker-build.yml workflow
- Image: ghcr.io/hanzoai/chat:main (multi-arch amd64+arm64)

## Known Issues (2026-03-26)
- `@modelcontextprotocol/sdk` pinned to 1.25.3 via pnpm.overrides (1.26+ requires zod/v4 which breaks)
- `zod-to-json-schema` pinned to 3.24.6 via pnpm.overrides (3.25+ requires zod/v4)
- Internal package names kept from upstream: @librechat/api, librechat-data-provider

## Upstream
- Fork of: danny-avila/LibreChat
- Behind upstream: ~3,200 commits
- Branding: Hanzo red #fd4444, "Hanzo Chat" title, H logo
