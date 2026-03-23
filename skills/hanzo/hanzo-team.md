# Hanzo Team - Collaboration Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-bot.md`, `hanzo/hanzo-k8s.md`

## Overview

Hanzo Team is the **collaboration platform** at `hanzo.team`. Upstream fork of hcengineering/platform (Huly), merged to v0.7.395 (fa03d2b). Fully rebranded: all `@hcengineering` packages renamed to `@hanzo`, `hardcoreeng` to `hanzoai`, `Huly` to `Hanzo Team`. IAM-only login (no local auth, no signup). Svelte frontend with a TypeScript backend.

## When to use

- Team project management, task tracking, and collaboration
- Issue tracking with boards, lists, and Kanban views
- Document editing and knowledge base
- Team chat and messaging (integrated channels)
- HR/recruiting workflows
- Deploying or maintaining the hanzo.team instance

## Hard requirements

1. **IAM-only login**: `HIDE_LOCAL_LOGIN=true`, `DISABLE_SIGNUP=true` -- all auth through hanzo.id
2. **Rebrand is complete**: All `@hcengineering` references are `@hanzo`, all `hardcoreeng` are `hanzoai`
3. **Front serves gzipped HTML**: When replacing `index.html`, MUST also replace `index.html.gz`
4. **Network policy**: Must allow ingress to `front`, `playground`, `account` pods
5. **Images**: `ghcr.io/hanzoai/team-*:latest` (built via CI/CD, never locally)

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://hanzo.team` |
| Upstream | hcengineering/platform v0.7.395 |
| Repo | `github.com/hanzoai/team` |
| Branch (prod) | `main` |
| Branch (dev) | `develop` |
| Frontend | Svelte |
| Backend | TypeScript |
| K8s manifests | `universe/infra/k8s/team/` |
| K8s namespace | `hanzo` |
| IngressClass | `hanzo` |
| Cloudflare DNS | `*.hanzo.team` -> `165.232.146.176` (worker-pool-h7y6j) |
| Images | `ghcr.io/hanzoai/team-{front,account,collaborator,love}:latest` |

## Architecture

```
                  hanzo.team
                      |
               Hanzo Ingress
                      |
          +-----------+-----------+
          |           |           |
       Front       Account    Collaborator
       (Svelte)    (auth)     (realtime)
          |           |           |
          +-----+-----+-----+----+
                |           |
            MongoDB      MinIO
            (state)    (files/blobs)
```

### Key services

| Service | Image | Purpose |
|---------|-------|---------|
| `front` | `ghcr.io/hanzoai/team-front` | Svelte SPA frontend |
| `account` | `ghcr.io/hanzoai/team-account` | Account/auth service |
| `collaborator` | `ghcr.io/hanzoai/team-collaborator` | Real-time collaboration (CRDT) |
| `love` | `ghcr.io/hanzoai/team-love` | Video/audio calling |

### Environment variables (front)

```bash
ACCOUNTS_URL=https://account.hanzo.team
UPLOAD_URL=https://hanzo.team/files
COLLABORATOR_URL=wss://collaborator.hanzo.team
LOVE_ENDPOINT=https://love.hanzo.team
BRANDING_PATH=/branding.json
HIDE_LOCAL_LOGIN=true
DISABLE_SIGNUP=true
```

## IAM integration

All authentication goes through hanzo.id. No local login forms, no signup flows.

The account service validates OIDC tokens from hanzo.id and maps them to team workspace memberships. Users must exist in IAM before they can access the team platform.

## Deployment

### Production (K8s)

```bash
# Apply manifests from universe
cd ~/work/hanzo/universe/infra/k8s/team
kubectl kustomize . | kubectl apply -f -
```

### CI/CD

Images are built via GitHub Actions CI/CD pipeline on the `hanzoai/team` repo:

```yaml
# .github/workflows/build.yml builds all team service images to GHCR
# Triggered on push to main/develop
# Uses self-hosted runners (never GHA billing)
```

### Branding

The front pod serves a `/branding.json` that controls logos, colors, and app name. This file is mounted via ConfigMap and read at `BRANDING_PATH`.

## Gotchas

1. **Gzip replacement**: The front serves `index.html.gz` directly. If you update `index.html`, you MUST also update `index.html.gz` or the old version will be served to clients.

2. **Network policies**: The hanzo-k8s cluster has network policies. Ensure ingress is allowed to front, playground, and account pods.

3. **DNS**: All `*.hanzo.team` subdomains resolve via Cloudflare to the worker pool node. The IngressRoute CRDs handle routing to the correct service.

4. **Tasks proxy**: The front pod proxies `/api/tasks/*` to `tasks.hanzo.ai` for external task integration.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login page shows local auth | `HIDE_LOCAL_LOGIN` not set | Set `HIDE_LOCAL_LOGIN=true` in front env |
| Old UI after deploy | Stale `index.html.gz` | Replace BOTH `.html` and `.html.gz` |
| 403 on front pod | Network policy blocking | Add ingress allow rule for front |
| Account service unreachable | DNS not resolving | Check `*.hanzo.team` Cloudflare record |
| Branding not loading | Wrong `BRANDING_PATH` | Ensure ConfigMap mounted at `/branding.json` |

## Related Skills

- `hanzo/hanzo-id.md` -- IAM for authentication
- `hanzo/hanzo-k8s.md` -- K8s cluster details
- `hanzo/hanzo-deploy.md` -- Deployment workflow
- `hanzo/hanzo-bot.md` -- Bot integration for team channels
- `hanzo/hanzo-ingress.md` -- Ingress routing

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: team, collaboration, huly, svelte, project-management
**Prerequisites**: K8s, IAM setup, Cloudflare DNS
