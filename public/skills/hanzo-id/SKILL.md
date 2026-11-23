<!-- Updated: 2026-07-23 -->
# Hanzo ID - Client-Side Auth (OAuth2/OIDC via @hanzo/iam)

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-iam.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-cloud.md`
**Spec**: HIP-0111 (auth) + HIP-0112 (cloud topology). Ground truth: `~/work/hanzo/CLAUDE.md`.

## Overview

Hanzo ID is the **client-side identity story** for every Hanzo/Lux/Zoo/Pars
service: how an app authenticates a user against the shared IAM server and
scopes data to the user's org. There is **one and only one way** to do it —
the **`@hanzo/iam`** SDK against the canonical OIDC endpoints. This skill is
the client contract. For the IAM **server** (Go backend, orgs, apps, admin
API) see `hanzo/hanzo-iam.md`.

## The one way (non-negotiable)

- **One package: `@hanzo/iam`** (npm, latest **0.13.7**). Framework-specific
  logic lives in subpaths — never re-implement any of it.
- **PKCE `S256` always.** The default and only browser flow is OAuth2
  authorization-code + PKCE. No implicit flow. No ROPC (no password to the
  token endpoint).
- **`client_secret_basic`** for confidential clients; **scopes**
  `openid profile email`.
- **`client_id = <org>-<app>`** (e.g. `hanzo-cloud`, `lux-console`,
  `zoo-cloud`). `redirectUris` must be the framework's exact callback path.
- **Secrets in KMS.** Superuser is `z@<domain>` / `Ilove<App>2026!!`.

### Canonical endpoints (host-relative to the brand `serverUrl`)

The SDK owns these in `src/paths.ts` (`OIDC_PATHS`) — the single source of
truth. Every module resolves through it; no path string is written anywhere
else.

| Purpose | Path |
|---------|------|
| OIDC discovery | `/.well-known/openid-configuration` |
| Authorize | `/v1/iam/oauth/authorize` |
| Token | `/v1/iam/oauth/token` |
| UserInfo | `/v1/iam/oauth/userinfo` |
| JWKS | `/v1/iam/.well-known/jwks` |
| Logout (RP-initiated) | `/v1/iam/oauth/logout` |

### Brands (set `serverUrl` to the brand host)

White-label is **host-based**: one IAM deployment serves every brand and
selects the tenant by the origin it is reached on.

| Brand | `serverUrl` |
|-------|-------------|
| hanzo | `https://hanzo.id` |
| lux | `https://lux.id` |
| zoo | `https://zoolabs.id` (NOT `zoo.id`) |
| pars | `https://pars.id` |
| bootnode | `https://id.bootno.de` |

`iam.hanzo.ai` is the **internal backend/admin host** — it is NOT the issuer
and NOT a public `serverUrl`. The Hanzo-brand issuer is **`hanzo.id`**.
Discovery is host-relative, so keep `originFrontend` EMPTY in the server's
`app.prod.conf` (see hanzo-iam.md).

## SDK subpaths (import the one that matches your runtime)

| Subpath | Use |
|---------|-----|
| `@hanzo/iam/browser` | SPA PKCE: `configureIam`, `startLogin`, `handleCallback`, `getSession`, `getUser`, `logout`, `completePopupSignin`, `loginWithWallet` (SIWx). |
| `@hanzo/iam/react` | React hooks/provider over the browser SDK. |
| `@hanzo/iam/nextauth` | `IamProvider` for NextAuth. |
| `@hanzo/iam/betterauth` | `iamProvider` for Better Auth — **explicit endpoints**, never `genericOAuth({discoveryUrl})`. |
| `@hanzo/iam/passport` | Passport strategy (Node). |
| `@hanzo/iam/server` | `validateToken`, `getServerSession` (verify bearer/JWTs server-side). |
| `@hanzo/iam/express`, `@hanzo/iam/hono`, `@hanzo/iam/sveltekit`, `@hanzo/iam/remix` | Framework middleware adapters. |
| `@hanzo/iam/validation` | Shared request/claim validation helpers. |
| `@hanzo/iam/paths` | `OIDC_PATHS`, `serverUrlForBrand`, `iamUrl` — the endpoint SoT. |

## When to use

- Adding auth to any Hanzo service (web, SPA, API, CLI, extension)
- Configuring OAuth2/OIDC/PKCE login
- Multi-org / white-label SSO
- Extracting org from the JWT `owner` claim to scope data
- Debugging login/token/callback issues

## Quick reference

| Item | Value |
|------|-------|
| SDK package | `@hanzo/iam` (npm, 0.13.7) |
| Login/issuer (hanzo) | `https://hanzo.id` |
| Internal API/admin host | `https://iam.hanzo.ai` (not the issuer) |
| Flow | Authorization code + PKCE `S256` (only) |
| Scopes | `openid profile email` |
| `client_id` | `<org>-<app>` |
| Token auth | `client_secret_basic` |
| Login UI repo | `github.com/hanzoai/id` |
| Server repo | `github.com/hanzoai/iam` (clean rewrite) |
| K8s manifests | `universe/infra/k8s/iam/` |

## Browser SPA (PKCE) — the canonical flow

```typescript
// One-time config (e.g. app bootstrap)
import { configureIam } from "@hanzo/iam/browser"

configureIam({
  serverUrl: "https://hanzo.id",       // brand host (see table)
  clientId: "hanzo-cloud",             // <org>-<app>
  redirectUri: "https://app.hanzo.ai/auth/callback", // exact registered URI
  scope: "openid profile email",
})
```

```typescript
// Start login (button handler). PKCE S256 + state handled by the SDK.
import { startLogin } from "@hanzo/iam/browser"
await startLogin()                       // or startLogin({ provider: "github" })
```

```typescript
// Callback route (/auth/callback). Exchange the single-use code ONCE.
import { handleCallback } from "@hanzo/iam/browser"
await handleCallback()                   // validates state, PKCE token exchange
```

```typescript
// Reading identity anywhere after login
import { getUser, getSession, logout } from "@hanzo/iam/browser"
const user = await getUser()             // { sub, email, owner, ... }
```

`getUser()` returns the OIDC userinfo shaped to camelCase. `owner` is the org
slug — use it to scope data (see below). Social login (Google/GitHub) that
must stay in-place uses `signinPopup()` + `completePopupSignin()` on the
callback route; wallet login uses `loginWithWallet()` (inline SIWx / EIP-4361,
no redirect). All paths converge on the SAME PKCE `handleCallback` exchange.

## PITFALL — single-use authorization code (guard the callback once)

The authorization `code` is **single-use** and the PKCE `code_verifier` is
**removed before** the token POST. If your callback effect runs twice — React
18 StrictMode double-invoke, a Fast-Refresh remount, or the user refreshing
the callback URL — the second run has no verifier (or the server rejects the
already-spent code), surfacing as "Missing PKCE code verifier" or a token
exchange 400.

- `handleCallback` is being made **idempotent in v0.13.8+** (a second call
  returns the already-exchanged session instead of throwing).
- **Consumers must still guard the callback effect to run exactly once** — the
  code is single-use regardless of SDK version. Use a run-once ref/guard, not
  a bare `useEffect(fn, [])`:

```typescript
// ✅ Run-once callback effect
const ran = useRef(false)
useEffect(() => {
  if (ran.current) return
  ran.current = true
  handleCallback().then(() => router.replace("/")).catch(showError)
}, [])
```

Do NOT retrigger `handleCallback` on re-render, and do NOT navigate the user
back onto the `?code=...` URL.

## Framework recipes

### Next.js (NextAuth)

```typescript
import NextAuth from "next-auth"
import { IamProvider } from "@hanzo/iam/nextauth"

export const { handlers, auth } = NextAuth({
  providers: [
    IamProvider({
      serverUrl: "https://hanzo.id",
      clientId: process.env.IAM_CLIENT_ID!,      // hanzo-<app>
      clientSecret: process.env.IAM_CLIENT_SECRET!, // from KMS
    }),
  ],
})
```

### Better Auth

```typescript
import { betterAuth } from "better-auth"
import { iamProvider } from "@hanzo/iam/betterauth"

export const auth = betterAuth({
  // iamProvider wires the EXPLICIT /v1/iam/oauth/* endpoints.
  // NEVER genericOAuth({ discoveryUrl }) — a wrong path returns IAM's 200
  // HTML SPA catch-all (silent breakage, not a 404).
  plugins: [iamProvider({ serverUrl: "https://hanzo.id", clientId, clientSecret })],
})
```

### Server-side token validation

```typescript
import { validateToken, getServerSession } from "@hanzo/iam/server"

const claims = await validateToken(bearer, { serverUrl: "https://hanzo.id" })
const orgId = claims.owner   // "hanzo" | "lux" | "zoo" | "pars"
```

## Org scoping (every authenticated service)

Extract the org from the JWT `owner` claim and scope ALL queries to it.

```go
// Go service pattern (behind the gateway)
orgID := claims["owner"].(string)   // "hanzo", "lux", "zoo", "pars"
db.Where("org_id = ?", orgID).Find(&results)
```

The gateway (`api.hanzo.ai`) validates the JWT and injects identity headers
for downstream services (`X-Org-Id`, `X-User-Id`, `X-User-Email`), and
**strips any client-supplied** identity headers first. Only the JWT-validated
path sets them.

## Forbidden (one-way violations)

| Never | Use instead |
|-------|-------------|
| `@hanzo/iam-js-sdk` (retired) | `@hanzo/iam` |
| Hand-rolled OAuth (manual authorize-URL building, `response_type=token`) | `@hanzo/iam/browser` |
| `genericOAuth({ discoveryUrl })` | `@hanzo/iam/betterauth` `iamProvider` (explicit endpoints) |
| Legacy `/oauth/*`, `/login/oauth/*`, `/api/login/*`, `/api/get-app-login`, `/api/get-account` | `/v1/iam/oauth/*` + SDK `getUser()` |
| Implicit flow | PKCE `S256` auth code |
| `iam.hanzo.ai` as issuer/serverUrl | brand host (`hanzo.id`, …) |
| Hardcoding `organization` in the login payload | `client_id = <org>-<app>` (org is derived) |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Missing PKCE code verifier" | Callback effect ran twice / page refreshed on `?code=` | Guard the callback to run once (see pitfall); upgrade to `@hanzo/iam` 0.13.8+ |
| Login returns a 200 HTML page, not JSON/tokens | Wrong path hit IAM's SPA catch-all | Use canonical `/v1/iam/oauth/*`; never `genericOAuth({discoveryUrl})`; keep `originFrontend` empty |
| Token exchange 400 on second attempt | Single-use code reused | Run `handleCallback` once; don't re-navigate to the code URL |
| Wrong org after login | Hardcoded `organization` | Use `client_id=<org>-<app>`; org comes from the app |
| CORS failure on discovery | IAM omits ACAO | SDK falls back to the SAME canonical paths automatically; set `proxyBaseUrl` to keep token/userinfo same-origin via the gateway |
| `zoo.id` returns nothing | Wrong host | Zoo's IAM is `zoolabs.id` |

## Agent compatibility

This skill is tool-agnostic markdown discoverable by both Claude (via
`CLAUDE.md` → `LLM.md` and `INDEX.md`) and Codex (via `AGENTS.md` → `LLM.md`).
Instructions are imperative and reference file paths, not any one agent.

## Related Skills

- `hanzo/hanzo-iam.md` — the IAM server (orgs, apps, admin API, ops)
- `hanzo/hanzo-identity.md` — DID / on-chain identity (distinct from OIDC auth)
- `hanzo/hanzo-platform.md` — PaaS (authenticates via `@hanzo/iam`)
- `hanzo/hanzo-kms.md` — where client secrets live
- `hanzo/hanzo-cloud.md` — cloud dashboard auth + billing

---

**Last Updated**: 2026-07-23
**Category**: Hanzo Ecosystem
**Related**: auth, oauth2, oidc, pkce, iam, identity, sso, multi-org, @hanzo/iam
**Prerequisites**: OAuth2/OIDC + PKCE concepts, JWT
