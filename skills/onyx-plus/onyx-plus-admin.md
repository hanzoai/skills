---
name: onyx-plus-admin
description: OnyxPlus operator dashboard -- Vite SPA on hanzoai/spa, OAuth via Hanzo IAM
---

# OnyxPlus Admin Dashboard

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-iam.md`

## Overview

`admin/` is the OnyxPlus operator-facing dashboard. Vite 8 SPA running on `hanzoai/spa` base image (the upstream Hanzo SPA runtime that writes runtime env into `/public/config.json` on boot).

- Repo: `github.com/onyx-plus/admin`
- Host: `onyxplus.{env}.satschel.com`
- Image: `us-docker.pkg.dev/onyxplus-registry/onyx-plus/admin:{semver}`
- Live tag: `1.0.0`

The `universe/k8s/{local,dev,test,main}/` directory in this repo holds the K8s manifests for the **entire** OnyxPlus stack -- not just admin. So a bump to onyxd, internal, or any other component goes through this repo's manifest tree.

## Purpose

Operator surface for:
- Browsing enrollments + attestations per org
- Auditing the on-chain claim trail per user
- Reject / override-liveness / re-enroll / revoke admin actions
- SEC/FINRA/IRS subpoena response export

## Authentication

OAuth 2.1 + PKCE against the shared Hanzo IAM at `id.satschel.com` (same IAM the rest of Liquidity/OnyxPlus uses):

- IAM client ID: `onyxplus-platform` (legacy alias) -- pending migration to `onyxplus-admin`
- Redirect URI: `https://onyxplus.{env}.satschel.com/auth/callback`
- Scope: `openid profile email`

After login the SPA holds a JWT and calls onyxd at `https://onyxplus-api.{env}.satschel.com/v1/onyxplus/*` with `Authorization: Bearer <JWT>`. onyxd validates against IAM's JWKS (cached).

## Stack

| Layer | Tech |
|---|---|
| Build | Vite 8 (SPA) |
| Runtime | `hanzoai/spa` Alpine base image (writes runtime env to `/public/config.json` on boot) |
| Lang | TypeScript |
| UI | React 19 |
| State | TBD (per repo convention) |
| Auth | OAuth 2.1 + PKCE against Hanzo IAM |
| Container | Listens on `:3000`; `runAsNonRoot=false` because `hanzoai/spa` writes the runtime config from boot scripts |

## Required env (manifest, runtime)

| Var | Value (dev) |
|---|---|
| `SPA_ENV` | `dev` |
| `SPA_API_HOST` | `https://onyxplus-api.dev.satschel.com` |
| `SPA_IAM_HOST` | `https://iam.dev.satschel.com` |
| `SPA_ID_HOST` | `https://id.dev.satschel.com` |
| `SPA_OAUTH_CLIENT_ID` | `onyxplus-platform` (→ `onyxplus-admin`) |
| `SPA_OAUTH_REDIRECT_URI` | `https://onyxplus.dev.satschel.com/auth/callback` |
| `SPA_OAUTH_SCOPE` | `"openid profile email"` |

`hanzoai/spa` reads these `SPA_*` env vars on container boot and writes them to `/public/config.json`, which the client-side bundle fetches before instantiating the OAuth client. Means the SAME image ships to dev/test/main with different env -- one immutable tag, three runtime configurations.

## K8s

```yaml
spec:
 containers:
 - name: admin
 image: us-docker.pkg.dev/onyxplus-registry/onyx-plus/admin:1.0.0
 ports:
 - containerPort: 3000
 securityContext:
 allowPrivilegeEscalation: false
 readOnlyRootFilesystem: false # SPA_* env → /public/config.json on boot
 capabilities: { drop: ["ALL"] }
 env: [ ... SPA_* ... ]
 readinessProbe: { httpGet: { path: /healthz, port: 3000 } }
 livenessProbe: { httpGet: { path: /healthz, port: 3000 }, initialDelaySeconds: 10, periodSeconds: 30 }
```

`readOnlyRootFilesystem` is intentionally `false` here -- the upstream `hanzoai/spa` runs as root and writes the runtime config file at boot. PR #6 in the admin repo addressed this.

## Deploy

```bash
# 1. Bump VERSION (e.g. 1.0.0 → 1.0.1) in ~/work/onyxplus/admin/VERSION
# 2. Submit Cloud Build
gcloud builds submit --config=cloudbuild.yaml \
 --project=onyxplus-registry \
 --gcs-source-staging-dir=gs://onyxplus-registry_cloudbuild/source

# 3. Bump image tag in admin/universe/k8s/{dev,test,main}/services.yaml
# 4. Commit + push

# 5. Roll out dev → test → main
kubectl --context=onyxplus-dev -n onyxplus set image deployment/admin \
 admin=us-docker.pkg.dev/onyxplus-registry/onyx-plus/admin:1.0.1
kubectl -n onyxplus rollout status deployment/admin
```

## Source pointers

| Subject | Source |
|---|---|
| Repo root | `~/work/onyxplus/admin/` |
| Vite config | `~/work/onyxplus/admin/vite.config.ts` |
| Cloud Build | `~/work/onyxplus/admin/cloudbuild.yaml` |
| Dockerfile (hanzoai/spa base) | `~/work/onyxplus/admin/Dockerfile` |
| K8s manifests (whole stack) | `~/work/onyxplus/admin/universe/k8s/{dev,test,main}/` |

## Related skills

- `onyx-plus/onyx-plus.md` - umbrella
- `onyx-plus/onyx-plus-onyxd.md` - backend the admin SPA calls
- `onyx-plus/onyx-plus-iam.md` - OAuth issuer
- `onyx-plus/onyx-plus-deploy.md` - deploy pipeline
- `hanzo/hanzo-spa.md` - `hanzoai/spa` runtime base image

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: admin, dashboard, vite, spa, hanzoai-spa, oauth
