# Hanzo Platform - Cloud PaaS for AI Applications

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-id.md`, `hanzo/hanzo-kms.md`

## Overview

Hanzo Platform is a **Kubernetes-native PaaS** (Platform as a Service) for deploying AI applications, APIs, and services. Hono.js + tRPC + Drizzle ORM API backend (port 4000) with Next.js 16 frontend (port 3000). Turborepo monorepo. Live at `platform.hanzo.ai`.

### Why Hanzo Platform?

- **Git-push deploys**: Push to main, auto-deploy to K8s
- **Self-hostable**: Run on your own K8s cluster with custom domain
- **IAM integrated**: Auth via hanzo.id (Casdoor-based)
- **Secrets via KMS**: Infisical-backed secret management
- **AI-native**: GPU scheduling, model serving, inference endpoints
- **White-label**: Fork and rebrand for your own PaaS

### Tech Stack

- **API**: Hono.js + tRPC + Drizzle ORM on port 4000
- **Frontend**: Next.js 16 on port 3000
- **Build**: Turborepo + pnpm workspace
- **Images**: `ghcr.io/hanzoai/paas-{api,studio,monitor,sync,webhook}:latest`

### OSS Base

Repo: `hanzoai/paas`.

## When to use

- Deploying Docker containers or applications to K8s
- Managing environments (staging, production)
- Git-push deployment workflows
- Self-hosting a PaaS platform
- White-labeling a deployment platform

## Hard requirements

1. **Kubernetes cluster** with `hanzo-paas-sa` service account
2. **IAM** configured at hanzo.id for authentication
3. **KMS** at kms.hanzo.ai for secrets (or self-hosted Infisical)
4. **GHCR access** for container images

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://platform.hanzo.ai` |
| API | `https://platform.hanzo.ai/v1` |
| Auth | IAM via hanzo.id (OAuth2) |
| Images | `ghcr.io/hanzoai/paas-{api,studio,monitor,sync,webhook}` |
| K8s SA | `hanzo-paas-sa` |
| KMS Project | `hanzo-paas` (slug min 5 chars) |
| Repo | `github.com/hanzoai/paas` |

## One-file quickstart

### Deploy an app via API

```bash
# Create a project
curl -X POST https://platform.hanzo.ai/v1/projects \
  -H "Authorization: Bearer ${HANZO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-ai-app",
    "environment": "production",
    "image": "ghcr.io/myorg/my-app:latest",
    "port": 3000,
    "replicas": 2
  }'
```

### Docker Compose (self-hosted)

```yaml
# compose.yml
services:
  paas-api:
    image: ghcr.io/hanzoai/paas-api:latest
    ports:
      - "8000:8000"
    environment:
      IAM_ENDPOINT: https://hanzo.id
      KMS_ENDPOINT: https://kms.hanzo.ai
      DATABASE_URL: postgresql://user:pass@postgres:5432/paas
      REDIS_URL: redis://redis:6379

  paas-studio:
    image: ghcr.io/hanzoai/paas-studio:latest
    ports:
      - "5173:5173"
    environment:
      API_URL: http://paas-api:8000

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: paas
      POSTGRES_USER: user
      POSTGRES_PASSWORD: "${DB_PASSWORD}"

  redis:
    image: redis:7-alpine
```

## Core Concepts

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Studio UI  │────▶│   PaaS API   │────▶│  K8s Cluster │
│  (Next.js)  │     │  (Hono.js)   │     │  (workloads) │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │   Services   │
                    ├──────────────┤
                    │ Monitor      │  ← Health checks
                    │ Sync         │  ← Git sync
                    │ Webhook      │  ← Deploy triggers
                    └──────────────┘
```

### Auth Flow

Platform uses `provider=hanzo` via Better Auth generic OAuth:
1. User clicks "Sign in with Hanzo" on platform.hanzo.ai
2. Redirected to hanzo.id for authentication
3. Token returned to platform via callback
4. Token validated against IAM userinfo endpoint

**IMPORTANT**: `platform/app/platform/pages/index.tsx` must target provider `"hanzo"` (NOT `"github"`).

### Environment Variables

```bash
# IAM Configuration (multiple fallback aliases)
HANZO_IAM_URL=https://hanzo.id          # Primary
HANZO_IAM_ENDPOINT=https://hanzo.id     # Alias
HANZO_IAM_SERVER_URL=https://hanzo.id   # Alias
IAM_ENDPOINT=https://hanzo.id           # Legacy

# IAM Client (multiple fallback aliases)
HANZO_IAM_CLIENT_ID=app-hanzo
HANZO_IAM_CLIENT_SECRET=secret
HANZO_CLIENT_ID=app-hanzo               # Alias
HANZO_CLIENT_SECRET=secret              # Alias
```

### KMS Integration

Secrets are KMS-first — no plaintext in manifests:

```yaml
# KMSSecret CRD for auto-sync
apiVersion: kms.hanzo.ai/v1
kind: KMSSecret
metadata:
  name: hanzo-paas
spec:
  project: hanzo-paas
  environment: production
  secrets:
    - DOCKERHUB_USERNAME
    - DOCKERHUB_TOKEN
    - DIGITALOCEAN_ACCESS_TOKEN
```

## White-Label / Self-Host

1. Fork `hanzoai/paas`
2. Update branding in `paas-studio/`
3. Configure IAM provider (any Casdoor/OIDC)
4. Deploy to your K8s cluster
5. Set custom domain via ingress

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Sign in with Hanzo" fails | Provider set to "github" | Change to "hanzo" in index.tsx |
| KMS slug error | Slug < 5 chars | Use `hanzo-paas` not `paas` |
| Token expired | `expireInHours=0` on Casdoor app | Set `expireInHours=168` |
| Duplicate users on IAM login | Email match not attempted | Auth falls back to email match |

## Related Skills

- `hanzo/hanzo-id.md` - IAM and authentication
- `hanzo/hanzo-kms.md` - Secret management
- `hanzo/hanzo-cloud.md` - Cloud dashboard
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: paas, deployment, kubernetes, platform
**Prerequisites**: Kubernetes, Docker, OAuth2 basics
