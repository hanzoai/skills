---
name: onyx-plus-admin
description:  operator dashboard -- Vite SPA on hanzoai/spa, OAuth via Hanzo IAM
---

#  Admin Dashboard

**Category**: 
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-iam.md`

## Overview

`admin/` is the  operator-facing dashboard. Vite 8 SPA running on `hanzoai/spa` base image (the upstream Hanzo SPA runtime that writes runtime env into `/public/config.json` on boot).

- Repo: `github.com/onyx-plus/admin`
- Host: `.{env}.`
- Image: `us-docker.pkg.dev/-registry/onyx-plus/admin:{semver}`
- Live tag: `1.0.0`

The `universe/k8s/{local,dev,test,main}/` directory in this repo holds the K8s manifests for the **entire**  stack -- not just admin. So a bump to onyxd, internal, or any other component goes through this repo's manifest tree.

## Purpose

Operator surface for:
- Browsing enrollments + attestations per org
- Auditing the on-chain claim trail per user
- Reject / override-liveness / re-enroll / revoke admin actions
- SEC/FINRA/IRS subpoena response export

## Authentication

OAuth 2.1 + PKCE against the shared Hanzo IAM at `id.` (same IAM the rest of Liquidity/ uses):

- IAM client ID: `-platform` (legacy alias) -- pending migration to `-admin`
- Redirect URI: `https://.{env}./auth/callback`
- Scope: `openid profile email`

After login the SPA holds a JWT and calls onyxd at `https://-api.{env}./v1//*` with `Authorization: Bearer <JWT>`. onyxd validates against IAM's JWKS (cached).

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
| `SPA_API_HOST` | `https://-api.dev.` |
| `SPA_IAM_HOST` | `https://iam.dev.` |
| `SPA_ID_HOST` | `https://id.dev.` |
| `SPA_OAUTH_CLIENT_ID` | `-platform` (→ `-admin`) |
| `SPA_OAUTH_REDIRECT_URI` | `https://.dev./auth/callback` |
| `SPA_OAUTH_SCOPE` | `"openid profile email"` |

`hanzoai/spa` reads these `SPA_*` env vars on container boot and writes them to `/public/config.json`, which the client-side bundle fetches before instantiating the OAuth client. Means the SAME image ships to dev/test/main with different env -- one immutable tag, three runtime configurations.

## K8s

```yaml
spec:
 containers:
 - name: admin
 image: us-docker.pkg.dev/-registry/onyx-plus/admin:1.0.0
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
# 1. Bump VERSION (e.g. 1.0.0 → 1.0.1) in ~/work//admin/VERSION
# 2. Submit Cloud Build
gcloud builds submit --config=cloudbuild.yaml \
 --project=-registry \
 --gcs-source-staging-dir=gs://-registry_cloudbuild/source

# 3. Bump image tag in admin/universe/k8s/{dev,test,main}/services.yaml
# 4. Commit + push

# 5. Roll out dev → test → main
kubectl --context=-dev -n  set image deployment/admin \
 admin=us-docker.pkg.dev/-registry/onyx-plus/admin:1.0.1
kubectl -n  rollout status deployment/admin
```

## Source pointers

| Subject | Source |
|---|---|
| Repo root | `~/work//admin/` |
| Vite config | `~/work//admin/vite.config.ts` |
| Cloud Build | `~/work//admin/cloudbuild.yaml` |
| Dockerfile (hanzoai/spa base) | `~/work//admin/Dockerfile` |
| K8s manifests (whole stack) | `~/work//admin/universe/k8s/{dev,test,main}/` |

## Related skills

- `onyx-plus/onyx-plus.md` - umbrella
- `onyx-plus/onyx-plus-onyxd.md` - backend the admin SPA calls
- `onyx-plus/onyx-plus-iam.md` - OAuth issuer
- `onyx-plus/onyx-plus-deploy.md` - deploy pipeline
- `hanzo/hanzo-spa.md` - `hanzoai/spa` runtime base image

---

**Last Updated**: 2026-05-12
**Category**: 
**Related**: admin, dashboard, vite, spa, hanzoai-spa, oauth
