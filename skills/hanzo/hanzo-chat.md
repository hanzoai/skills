# Hanzo Chat - AI Chat Application

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Chat (`@hanzochat/chat`) is a **full-featured AI chat application** -- a LibreChat v0.8.3-rc1 fork providing a web UI for conversing with AI models, managing conversations, using agents, and integrating with RAG pipelines and MCP tools. It connects to the Hanzo LLM Gateway (`api.hanzo.ai/v1`) as its backend for model access.

**This is the chat UI application, NOT the LLM API.** The LLM gateway that provides model access is a separate service (`hanzoai/llm`). Hanzo Chat is the frontend that users interact with at `chat.hanzo.ai`.

### What it actually is

- A LibreChat fork (v0.8.3-rc1) rebranded as Hanzo Chat
- Web UI (React/Next.js client) + Node.js API server
- pnpm workspace monorepo: `api/`, `client/`, `packages/*`
- Connects to Hanzo LLM Gateway for model inference
- Supports agents, RAG, file uploads, code execution, conversation management
- Full user management: create, invite, ban, delete users; balance management
- MongoDB for conversation storage, MeiliSearch for search
- Playwright E2E tests, Jest unit tests
- Deployed at `chat.hanzo.ai`

### What it is NOT

- Not the LLM Gateway API (that is `hanzoai/llm`, the proxy for 435+ models)
- Not providing models directly -- it consumes them via the gateway
- Does not serve `api.hanzo.ai/v1` endpoints

## When to use

- Running a self-hosted AI chat interface
- Building a multi-model chat application with conversation history
- Need a UI for agents, RAG, and MCP tool integration
- Deploying a team chat with user management and usage tracking

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/chat` |
| Package | `@hanzochat/chat` |
| Version | v0.8.3-rc1 |
| Upstream | LibreChat v0.8.x |
| Stack | Node.js, React, pnpm |
| Live | `chat.hanzo.ai` |
| API server | `api/server/index.js` (port 3080) |
| Client | `client/` (React) |
| Packages | `packages/data-provider`, `packages/data-schemas`, `packages/api`, `packages/client` |
| Config | `librechat.yaml` (ConfigMap in K8s) |
| Database | MongoDB |
| Dev (backend) | `pnpm backend:dev` |
| Dev (frontend) | `pnpm frontend:dev` |
| Build | `pnpm build` (turbo) |
| Test | `pnpm test:all` |
| E2E | `pnpm e2e` (Playwright) |
| License | ISC |

## Architecture

```
                    chat.hanzo.ai
                         |
              +----------+----------+
              |                     |
         React Client          Node.js API
         (port 3080)          (api/server/)
              |                     |
              +----------+----------+
                         |
              +----------+----------+
              |          |          |
           MongoDB   MeiliSearch   LLM Gateway
           (convos)  (search)    (api.hanzo.ai/v1)
                                      |
                                 435+ models
                                 28 Zen models
```

## Workspace structure

```
hanzoai/chat/
  package.json        # Root workspace (@hanzochat/chat v0.8.3-rc1)
  api/                # Express API server
    server/           # Server entry point
  client/             # React frontend
  packages/
    data-provider/    # Data access layer
    data-schemas/     # Schema definitions
    api/              # API client package
    client/           # Client component library
  config/             # CLI admin tools
    create-user.js
    add-balance.js
    list-balances.js
    ban-user.js
    delete-user.js
    reset-password.js
    ...
  e2e/                # Playwright E2E tests
```

## Quickstart

```bash
git clone https://github.com/hanzoai/chat.git
cd chat
pnpm install

# Configure
cp .env.example .env
# Required: MONGO_URI, JWT_SECRET, CREDS_KEY, CREDS_IV
# LLM backend: OPENAI_BASE_URL=http://llm.hanzo.svc.cluster.local:4000/v1

# Build packages first
pnpm build:packages

# Development
pnpm backend:dev    # API server with nodemon
pnpm frontend:dev   # React client with hot-reload

# Production
pnpm build
pnpm backend
```

## Admin CLI tools

```bash
# User management
pnpm create-user
pnpm invite-user
pnpm list-users
pnpm ban-user
pnpm delete-user
pnpm reset-password

# Balance management
pnpm add-balance
pnpm set-balance
pnpm list-balances

# Maintenance
pnpm flush-cache
pnpm reset-meili-sync
pnpm update-banner
```

## K8s deployment (production)

- Image: `hanzoai/chat:latest` on Docker Hub
- Config: ConfigMap `chat-config` mounted at `/app/librechat.yaml`
- Secret: `chat-secrets` (MONGO_URI, JWT_SECRET, CREDS_KEY/IV)
- Env: `OPENAI_BASE_URL=http://llm.hanzo.svc.cluster.local:4000/v1`
- Replicas: 2, port 3080
- Ingress: `chat.hanzo.ai`
- CI: `docker-publish.yml` builds and pushes to Docker Hub

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MONGO_URI` | MongoDB connection string |
| `JWT_SECRET` | Session JWT secret |
| `CREDS_KEY` | Credential encryption key |
| `CREDS_IV` | Credential encryption IV |
| `OPENAI_BASE_URL` | LLM Gateway URL (e.g., `http://llm.hanzo.svc.cluster.local:4000/v1`) |
| `APP_TITLE` | UI title (default: `Hanzo Chat`) |
| `DOMAIN_CLIENT` | Client domain |
| `DOMAIN_SERVER` | Server domain |
| `CUSTOM_FOOTER` | Footer text |

## Upstream (LibreChat)

Internal package names from LibreChat are preserved as-is:
- `@librechat/agents` -- Agent framework
- `librechat-data-provider` -- Data access
- Function names like `extractLibreChatParams`, `importLibreChatConvo` -- kept for compatibility

## Related Skills

- `hanzo/hanzo-llm-gateway.md` -- The LLM proxy that Chat connects to (435+ models, 28 Zen)
- `hanzo/hanzo-mcp.md` -- MCP tools accessible through Chat
- `hanzo/hanzo-console.md` -- Console for API key management

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: chat, ui, librechat, conversations, agents, rag
**Prerequisites**: Node.js, pnpm, MongoDB, Hanzo LLM Gateway (or OpenAI-compatible endpoint)
