# Hanzo Chat - AI Chat Application

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Chat is a **full-featured AI chat application** -- a LibreChat v0.8.3-rc1 fork providing a web UI for multi-model conversations, agents, RAG pipelines, MCP tools, and file uploads. Connects to the Hanzo LLM Gateway for model access. Live at `chat.hanzo.ai`.

**This is the chat UI, NOT the LLM API.** The LLM gateway (`hanzoai/llm`) provides model access. Chat is the frontend users interact with.

## When to use

- Running a self-hosted AI chat interface
- Multi-model conversations with 14 Zen models + 100+ third-party
- Agents with RAG and MCP tool integration
- Team chat with user management and usage tracking

## Hard requirements

1. **MongoDB** for conversation storage
2. **LLM Gateway** (`hanzoai/llm`) or any OpenAI-compatible endpoint
3. **Secrets encrypted**: `JWT_SECRET`, `CREDS_KEY`, `CREDS_IV` must be set
4. **pnpm workspace** for builds

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://chat.hanzo.ai` |
| Repo | `github.com/hanzoai/chat` |
| Package | `@hanzochat/chat` |
| Version | v0.8.3-rc1 |
| Upstream | LibreChat v0.8.x |
| API server | `api/server/index.js` (port 3080) |
| Client | `client/` (React) |
| Config | `librechat.yaml` (ConfigMap in K8s) |
| Database | MongoDB |
| Search | MeiliSearch |
| Image | `ghcr.io/hanzoai/chat:latest` |
| K8s manifests | `universe/infra/k8s/chat/` |
| Dev (backend) | `pnpm backend:dev` |
| Dev (frontend) | `pnpm frontend:dev` |
| Build | `pnpm build` (turbo) |
| Test | `pnpm test:all` |
| E2E | `pnpm e2e` (Playwright) |

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
     (convos)  (search)    (llm.hanzo.svc:4000/v1)
                                  |
                             100+ models
                             14 Zen models
```

## Workspace structure

```
hanzoai/chat/
  package.json        # Root (@hanzochat/chat v0.8.3-rc1)
  api/                # Express API server
    server/           # Entry point
  client/             # React frontend
  packages/
    data-provider/    # Data access layer
    data-schemas/     # Schema definitions
    api/              # API client package
    client/           # Component library
  config/             # CLI admin tools
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
# LLM: OPENAI_BASE_URL=http://llm.hanzo.svc.cluster.local:4000/v1

# Build packages
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
pnpm create-user        # Create new user
pnpm invite-user        # Send invitation
pnpm list-users         # List all users
pnpm ban-user           # Ban user
pnpm delete-user        # Delete user
pnpm reset-password     # Reset password

# Balance management
pnpm add-balance        # Add credits
pnpm set-balance        # Set balance
pnpm list-balances      # View balances

# Maintenance
pnpm flush-cache        # Clear cache
pnpm reset-meili-sync   # Reset search index
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MONGO_URI` | MongoDB connection string |
| `JWT_SECRET` | Session JWT secret |
| `CREDS_KEY` | Credential encryption key |
| `CREDS_IV` | Credential encryption IV |
| `OPENAI_BASE_URL` | LLM Gateway URL (`http://llm.hanzo.svc:4000/v1`) |
| `APP_TITLE` | UI title (default: `Hanzo Chat`) |
| `DOMAIN_CLIENT` | Client domain |
| `DOMAIN_SERVER` | Server domain |

## K8s deployment

```yaml
# universe/infra/k8s/chat/
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: chat
          image: ghcr.io/hanzoai/chat:latest
          ports:
            - containerPort: 3080
          envFrom:
            - secretRef:
                name: chat-secrets  # KMS-synced
          env:
            - name: OPENAI_BASE_URL
              value: http://llm.hanzo.svc:4000/v1
```

Config via ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chat-config
data:
  librechat.yaml: |
    # Model configuration, endpoints, features
```

## Zen models available

14 Zen frontier models (Mixture of Diverse Experts architecture):
- zen-1b, zen-4b, zen-8b, zen-14b, zen-32b, zen-70b, zen-235b, zen-480b
- zen-coder-32b, zen-coder-70b
- zen-vision-32b, zen-vision-70b
- zen-math-32b
- zen-reasoning-32b

Plus 100+ third-party models via the LLM Gateway.

## Related Skills

- `hanzo/hanzo-llm-gateway.md` -- LLM proxy (100+ providers)
- `hanzo/hanzo-mcp.md` -- MCP tools accessible through Chat
- `hanzo/hanzo-console.md` -- Observability and cost tracking
- `hanzo/hanzo-cloud.md` -- API key management

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: chat, ui, librechat, conversations, agents, rag, zen-models
**Prerequisites**: Node.js, pnpm, MongoDB, LLM Gateway
