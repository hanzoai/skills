# Hanzo ID - Identity and Authentication Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo ID is the **unified identity platform** for the entire Hanzo/Lux/Zoo/Pars ecosystem. Casdoor-based IAM backend with a custom Next.js login UI and Cloudflare Worker for edge OAuth flows. Serves OAuth2/OIDC for all Hanzo services. White-label-ready with per-org customization. Every authenticated service in the stack uses Hanzo ID.

## When to use

- Implementing authentication for any Hanzo service
- Setting up OAuth2/OIDC flows
- Configuring multi-org SSO
- White-labeling login for client deployments
- Debugging auth token issues
- Extracting org from JWT `owner` claim for data scoping

## Hard requirements

1. **Casdoor** IAM backend running (Go/Beego)
2. **App registered** in Casdoor admin (`owner=admin`, NOT org name)
3. **Implicit flow preferred** -- code flow has Casdoor bug with empty grant_type
4. **All data queries scoped to org** -- extract org from JWT `owner` claim
5. **RFC 6749/OIDC endpoints only** for new integrations (never legacy Casdoor paths)

## Quick reference

| Item | Value |
|------|-------|
| Login UI | `https://hanzo.id` |
| IAM API | `https://iam.hanzo.ai` (backend, alias) |
| Authorize | `/oauth/authorize` |
| Token | `/oauth/token` |
| Userinfo | `/oauth/userinfo` (limited: sub/iss/aud) |
| Full profile | `/api/get-account` (needs auth) |
| App lookup | `/api/get-app-login` (source of truth) |
| Discovery | `/.well-known/openid-configuration` |
| Worker | `hanzo.id-worker` (Cloudflare) |
| Repo (UI) | `github.com/hanzoai/id` |
| Repo (IAM) | `github.com/hanzoai/iam` |
| K8s manifests | `universe/infra/k8s/iam/` |
| Image | `ghcr.io/hanzoai/iam:latest` |

## CRITICAL: RFC-Compliant endpoints

**Standardized 2026-03-05.** Always use these:

| Endpoint | Path | Legacy (do NOT use) |
|----------|------|---------------------|
| Authorize | `/oauth/authorize` | ~~/login/oauth/authorize~~ |
| Token | `/oauth/token` | ~~/api/login/oauth/access_token~~ |
| Introspect | `/oauth/introspect` | ~~/api/login/oauth/introspect~~ |
| Revoke | `/oauth/revoke` | -- |
| UserInfo | `/oauth/userinfo` | Also `/api/userinfo` |
| Device | `/oauth/device` | -- |
| Logout | `/oauth/logout` | -- |

Legacy Casdoor paths still work for backward compat but NEVER use them for new integrations.

## Multi-org architecture

```
hanzo.id (login UI)
  +-- org: hanzo  --> app-hanzo, app-console, app-platform, app-kms
  +-- org: lux    --> app-lux-cloud, app-lux-explorer
  +-- org: zoo    --> app-zoo-gym
  +-- org: pars   --> app-pars
```

Each org can have its own domain, branding, and app set. Domains: hanzo.id, lux.id, zoo.id, pars.id.

## Auth flow: Implicit (recommended for browser)

```typescript
// Start auth flow
const authUrl = new URL("https://hanzo.id/oauth/authorize")
authUrl.searchParams.set("response_type", "token")  // implicit flow
authUrl.searchParams.set("client_id", "app-hanzo")
authUrl.searchParams.set("redirect_uri", "https://myapp.com/callback")
authUrl.searchParams.set("scope", "openid profile email")
authUrl.searchParams.set("state", crypto.randomUUID())

window.location.href = authUrl.toString()
```

### Extract token from callback

```typescript
const hash = new URLSearchParams(window.location.hash.substring(1))
const accessToken = hash.get("access_token")

// userinfo only returns sub -- use get-account for full profile
const profile = await fetch("https://iam.hanzo.ai/api/get-account", {
  headers: { "Authorization": `Bearer ${accessToken}` }
}).then(r => r.json())
```

## App lookup (dynamic org/app resolution)

```typescript
// ALWAYS use this before login -- source of truth for app/org
const appLogin = await fetch("https://hanzo.id/api/get-app-login", {
  method: "POST",
  body: JSON.stringify({ clientId: "app-hanzo" })
}).then(r => r.json())

// Use appLogin.application and appLogin.organization
// Do NOT hardcode organization: "hanzo" -- breaks scoped SSO
```

## Worker architecture (Cloudflare)

`hanzo.id-worker` handles edge OAuth:

- `/oauth/hanzo/platform` -- starts IAM auth, pins callback to `/callback/platform/hanzo`
- `/callback/platform/hanzo` -- exchanges authorization code, redirects back with `access_token`, `refresh_token`, `expires_at`, `provider=hanzo`, `status=200`
- Uses `/api/get-app-login` as source of truth (NOT hardcoded)
- Includes `signinMethod`/`language` in password login payload
- Fallback client-id map only for degraded operation

## JWT claims for service auth

All authenticated services extract org from JWT:

```go
// Go service pattern
claims := extractClaims(token)
orgID := claims["owner"].(string)  // "hanzo", "lux", "zoo", "pars"

// Scope ALL data queries to org
db.Where("org_id = ?", orgID).Find(&results)
```

Gateway (`api.hanzo.ai`) injects identity headers from JWT for downstream services.

## Token flow gotchas

| Issue | Detail |
|-------|--------|
| **Code flow bug** | `response_type=code` maps to empty grant_type `""` in Casdoor. Even with `""` in DB `grant_types` array, Go xorm drops empty strings. |
| **Solution** | Use `response_type=token` (implicit flow) for all browser-based auth. |
| **Token expiry** | Casdoor apps default `expireInHours=0` (instant expiry). Always set `expireInHours=168` (7 days). |
| **API masking** | Casdoor's `/api/get-application` masks fields for unauthed requests. Use `/api/get-app-login` instead. |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login returns empty token | `expireInHours=0` | Set to 168 in Casdoor app config |
| Code flow rejected | Empty grant_type bug | Use implicit flow (`response_type=token`) |
| Wrong org after login | Hardcoded `organization: "hanzo"` | Use `/api/get-app-login` lookup |
| Userinfo only returns `sub` | Normal Casdoor behavior | Use `/api/get-account` for full profile |
| App appears to have no config | API masking on unauthed requests | Use authenticated request |
| Missing cert reference | Wrong client_id for scoped SSO | Use `/api/get-app-login` to resolve |
| Platform sign-in fails | Provider set to `"github"` | Must be `"hanzo"` in platform code |

## Related Skills

- `hanzo/hanzo-platform.md` -- PaaS uses Hanzo ID for auth
- `hanzo/hanzo-kms.md` -- Secrets management (also uses IAM)
- `hanzo/hanzo-cloud.md` -- Cloud dashboard auth
- `hanzo/hanzo-billing.md` -- Billing portal auth

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: auth, oauth, oidc, iam, casdoor, identity, sso, multi-org
**Prerequisites**: OAuth2/OIDC concepts, JWT
