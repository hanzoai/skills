# Hanzo ID - Identity & Authentication

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo ID is the **unified identity platform** for the entire Hanzo/Lux/Zoo ecosystem. Custom login UI built on Casdoor (IAM), serving OAuth2/OIDC for all Hanzo services. White-label-ready with per-org customization.

### Why Hanzo ID?

- **Single sign-on**: One identity across all Hanzo services
- **RFC 6749/OIDC compliant**: Standard OAuth2 endpoints
- **Multi-org**: hanzo, lux, zoo, pars (extensible)
- **White-label**: Custom login UI per organization
- **Worker-based**: Cloudflare Worker for edge OAuth flows

### OSS Base

**Casdoor** (IAM backend) + custom **Next.js** login UI (`hanzo/id`).

## When to use

- Implementing authentication for any Hanzo service
- Setting up OAuth2/OIDC flows
- Configuring multi-org SSO
- White-labeling login for client deployments
- Debugging auth token issues

## Hard requirements

1. **Casdoor** IAM backend running
2. **App registered** in Casdoor admin (owner=`admin`, NOT org name)
3. **Implicit flow preferred** — code flow has Casdoor bug with empty grant_type

## Quick reference

| Item | Value |
|------|-------|
| Login UI | `https://hanzo.id` |
| IAM API | `https://iam.hanzo.ai` (backend) |
| Authorize | `/oauth/authorize` |
| Token | `/oauth/token` |
| Userinfo | `/oauth/userinfo` (limited: sub/iss/aud only) |
| Full profile | `/api/get-account` (needs auth) |
| App lookup | `/api/get-app-login` (source of truth) |
| Discovery | `/.well-known/openid-configuration` |
| Worker | `hanzo.id-worker` (Cloudflare) |
| Repo (UI) | `github.com/hanzoai/id` |

## CRITICAL: RFC-Compliant Endpoints

**Always use these** (standardized 2026-03-05):

| Endpoint | Path | Notes |
|----------|------|-------|
| Authorize | `/oauth/authorize` | NOT `/login/oauth/authorize` |
| Token | `/oauth/token` | NOT `/api/login/oauth/access_token` |
| Introspect | `/oauth/introspect` | NOT `/api/login/oauth/introspect` |
| Revoke | `/oauth/revoke` | |
| UserInfo | `/oauth/userinfo` | Also `/api/userinfo` |
| Device | `/oauth/device` | |
| Logout | `/oauth/logout` | |

Legacy Casdoor paths still work but NEVER use them for new integrations.

## One-file quickstart

### Implicit OAuth2 Flow (recommended)

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
// On callback page
const hash = new URLSearchParams(window.location.hash.substring(1))
const accessToken = hash.get("access_token")
const state = hash.get("state")

// Get full profile (userinfo only returns sub)
const profile = await fetch("https://iam.hanzo.ai/api/get-account", {
  headers: { "Authorization": `Bearer ${accessToken}` }
}).then(r => r.json())
// profile.name, profile.email, profile.avatar, etc.
```

### App Lookup (dynamic org/app resolution)

```typescript
// ALWAYS use this before login — source of truth for app/org
const appLogin = await fetch("https://hanzo.id/api/get-app-login", {
  method: "POST",
  body: JSON.stringify({ clientId: "app-hanzo" })
}).then(r => r.json())

// Use appLogin.application and appLogin.organization
// Do NOT hardcode organization: "hanzo" — breaks scoped SSO
```

## Core Concepts

### Multi-Org Architecture

```
hanzo.id (login UI)
├── org: hanzo    → app-hanzo, app-console, app-platform
├── org: lux      → app-lux-cloud, app-lux-explorer
├── org: zoo      → app-zoo-gym
└── org: pars     → app-pars
```

Each org can have its own domain, branding, and app set.

### Token Flow Gotchas

**Code flow bug**: `response_type=code` maps to empty grant_type `""` in Casdoor. Even with `""` in DB `grant_types` array, Go xorm drops empty strings.

**Solution**: Use `response_type=token` (implicit flow) for all browser-based auth.

**Token expiry**: Casdoor apps default `expireInHours=0` which means instant expiry. Always set `expireInHours=168` (7 days).

### Worker Architecture

`hanzo.id-worker` (Cloudflare Worker) handles:
- `/oauth/hanzo/platform` → starts IAM auth, pins callback
- `/callback/platform/hanzo` → exchanges code, redirects with tokens
- Uses `/api/get-app-login` as source of truth (NOT hardcoded)

### API Masking Warning

Casdoor's `/api/get-application` masks fields for unauthenticated requests — appears empty but DB has real values. Always use authenticated requests or `/api/get-app-login`.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login returns empty token | `expireInHours=0` | Set to 168 in Casdoor app config |
| Code flow rejected | Empty grant_type bug | Use implicit flow (response_type=token) |
| Wrong org after login | Hardcoded `organization: "hanzo"` | Use `/api/get-app-login` lookup |
| Userinfo only returns `sub` | Normal Casdoor behavior | Use `/api/get-account` for full profile |
| App appears to have no config | API masking | Use authenticated request |
| Missing cert reference | Wrong client_id for scoped SSO | Use `/api/get-app-login` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS uses Hanzo ID for auth
- `hanzo/hanzo-kms.md` - Secrets management (also uses IAM)
- `hanzo/hanzo-extension.md` - Browser extension auth flow
- `hanzo/hanzo-cloud.md` - Cloud dashboard auth

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: auth, oauth, oidc, iam, casdoor, identity
**Prerequisites**: OAuth2/OIDC concepts, JWT
