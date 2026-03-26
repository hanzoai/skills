<!-- Updated: 2026-03-26T15:03:36Z -->
# Hanzo Platform - Cloud PaaS for AI Applications

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-id.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-k8s.md`

## Overview

Hanzo Platform is a **Kubernetes-native PaaS** (Platform as a Service) for deploying AI applications, APIs, and services. Hono.js + tRPC + Drizzle ORM API backend with Next.js 16 frontend. Turborepo monorepo. Fork of Dokploy. Live at `platform.hanzo.ai`.

## When to use

- Deploying Docker containers or applications to K8s
- Managing environments (staging, production) with git-push deploys
- Self-hosting a PaaS platform for your team
- White-labeling a deployment platform

## Hard requirements

1. **Kubernetes cluster** with `hanzo-paas-sa` service account
2. **IAM** configured at hanzo.id for authentication (provider `hanzo`, NOT `github`)
3. **KMS** at kms.hanzo.ai for secrets (KMSSecret CRDs, never plaintext)
4. **GHCR access** for container images
5. **No nginx, no caddy** -- Hanzo Ingress (Traefik) handles routing
6. **Self-hosted runners only** for CI/CD -- never GitHub Actions on hanzoai org

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://platform.hanzo.ai` |
| API | `https://platform.hanzo.ai/v1` |
| Auth | IAM via hanzo.id (OAuth2, provider=`hanzo`) |
| Images | `ghcr.io/hanzoai/paas-{api,studio,monitor,sync,webhook}:latest` |
| K8s SA | `hanzo-paas-sa` |
| KMS Project | `hanzo-paas` (slug must be >= 5 chars) |
| Repo | `github.com/hanzoai/paas` |
| API port | 4000 (Hono.js + tRPC) |
| Frontend port | 3000 (Next.js 16) |
| Build | Turborepo + pnpm workspace |
| K8s namespace | `hanzo` |
| K8s manifests | `universe/infra/k8s/paas/` |

## Architecture

```
                  platform.hanzo.ai
                        |
             +----------+----------+
             |                     |
        Studio UI            PaaS API
        (Next.js 16)       (Hono.js + tRPC)
        port 3000            port 4000
             |                     |
             +----------+----------+
                        |
                 +------+------+
                 |      |      |
              Monitor  Sync  Webhook
              (health) (git) (deploy)
                        |
                 +------+------+
                 |             |
              PostgreSQL    K8s Cluster
              (Drizzle)    (workloads)
```

## Auth flow

Platform uses `provider=hanzo` via Better Auth generic OAuth.

1. User clicks "Sign in with Hanzo" on platform.hanzo.ai
2. Redirected to hanzo.id for authentication
3. Token returned to platform via callback
4. Token validated against IAM userinfo endpoint

**CRITICAL**: `platform/app/platform/pages/index.tsx` MUST target provider `"hanzo"` (NOT `"github"`).

### IAM environment variables (multiple fallback aliases supported)

```bash
# IAM URL (checked in order)
HANZO_IAM_URL=https://hanzo.id
HANZO_IAM_ENDPOINT=https://hanzo.id
HANZO_IAM_SERVER_URL=https://hanzo.id
IAM_ENDPOINT=https://hanzo.id

# IAM Client (checked in order)
HANZO_IAM_CLIENT_ID=app-hanzo
HANZO_IAM_CLIENT_SECRET=secret
HANZO_CLIENT_ID=app-hanzo
HANZO_CLIENT_SECRET=secret
```

Auth validator supports `provider=hanzo` by validating bearer tokens against IAM userinfo (`IAM_USERINFO_URL` / `IAM_ENDPOINT`).

## KMS integration

All secrets are KMS-first. No plaintext in manifests.

```yaml
# universe/infra/k8s/paas/secrets.yaml
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

CI/CD also pulls from KMS at runtime:

```yaml
# paas/.github/workflows/deploy.yml
jobs:
  deploy:
    steps:
      - name: Login to KMS
        run: |
          export KMS_TOKEN=$(curl -s -X POST $KMS_URL/api/v1/auth/universal-auth/login \
            -d '{"clientId":"...","clientSecret":"..."}' | jq -r '.accessToken')
      - name: Fetch deploy secrets
        run: |
          DOCKERHUB_TOKEN=$(curl -s "$KMS_URL/api/v1/secrets/raw/DOCKERHUB_TOKEN?..." \
            -H "Authorization: Bearer $KMS_TOKEN" | jq -r '.secret.secretValue')
```

## Deploy an app via API

```bash
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

## Self-host via Docker Compose

```yaml
# compose.yml
services:
  paas-api:
    image: ghcr.io/hanzoai/paas-api:latest
    ports:
      - "4000:4000"
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
      API_URL: http://paas-api:4000

  postgres:
    image: ghcr.io/hanzoai/sql:latest
    environment:
      POSTGRES_DB: paas
      POSTGRES_USER: paas

  redis:
    image: ghcr.io/hanzoai/kv:latest
```

## White-label / self-host

1. Fork `hanzoai/paas`
2. Update branding in `paas-studio/`
3. Configure IAM provider (any Casdoor/OIDC)
4. Deploy to your K8s cluster
5. Set custom domain via Ingress

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Sign in with Hanzo" fails | Provider set to `"github"` | Change to `"hanzo"` in index.tsx |
| KMS slug error | Slug < 5 chars | Use `hanzo-paas` not `paas` |
| Token expired | `expireInHours=0` on Casdoor app | Set `expireInHours=168` |
| Duplicate users on IAM login | Email match not attempted | Auth falls back to email match for existing users |
| Studio login fails for non-git | `addGitProvider` called for IAM | Studio loader treats `hanzo` as first-class, skips `addGitProvider` |

## Related Skills

- `hanzo/hanzo-id.md` -- IAM and authentication
- `hanzo/hanzo-kms.md` -- Secret management
- `hanzo/hanzo-k8s.md` -- K8s infrastructure
- `hanzo/hanzo-deploy.md` -- Deployment workflow
- `hanzo/hanzo-ingress.md` -- Ingress routing

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: paas, deployment, kubernetes, platform, dokploy
**Prerequisites**: Kubernetes, Docker, OAuth2 basics
