---
name: onyx-plus-internal
description: OnyxPlus internal docs site -- Next.js 16 + @hanzo/docs + NextAuth v5 (Google OAuth restricted to @satschel.com)
---

# OnyxPlus Internal Docs Site

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `hanzo/hanzo-docs.md`, `onyx-plus/onyx-plus-deploy.md`

## Overview

`internal/` is the OnyxPlus engineering documentation site. Next.js 16 (Turbopack, standalone output) + `@hanzo/docs` framework, gated to `@satschel.com` Google accounts via NextAuth v5.

- Repo: `github.com/onyx-plus/internal`
- Hosts: `internal.onyxplus.{env}.satschel.com`
- Image: `us-docker.pkg.dev/onyxplus-registry/onyx-plus/internal:{semver}`
- Live tag: `0.1.1` (2026-05-12)

## Pages

| Slug | Topic |
|---|---|
| `/docs` | Index |
| `/docs/architecture` | Identity model, 1:1 invariant, IAM-KMS-MPC handshake, env vars, deploy order |
| `/docs/platform-stack` | Old (Simplici) vs new (in-house) stack comparison |
| `/docs/iam` | IAM tenant detail (full Casdoor flow) |
| `/docs/kms` | KMS consumer detail (ZAP transport, secret paths) |
| `/docs/mpc` | MPC consumer detail (BD → TA → MPC; JWT-claims invariant) |
| `/docs/product-comparison` | Refutation of misleading slide claims |
| `/signin` | NextAuth signin (Google OAuth, satschel.com-only) |

Sidebar order pinned by `content/docs/meta.json`.

## Auth model

NextAuth v5 with Google OAuth provider. Two callback guards in `auth.ts`:

```ts
async signIn({ profile }) {
  return emailDomainAllowed(profile?.email as string | undefined);
},
async session({ session }) {
  if (!emailDomainAllowed(session.user?.email)) {
    return { ...session, user: undefined } as typeof session;
  }
  return session;
},
```

`allowedEmailDomains` lives in `lib/shared.ts`:

```ts
export const allowedEmailDomains = ['satschel.com'];
```

Anonymous traffic (no session OR session whose email failed `allowedEmailDomains`) is redirected to `/signin` by `proxy.ts` (Next 16 renamed `middleware.ts` to `proxy.ts`; both jobs -- auth gate + markdown content negotiation -- are combined in one file).

## Required env (production)

| Var | Source | Notes |
|---|---|---|
| `AUTH_SECRET` | K8s Secret `internal-secrets` | `openssl rand -base64 32`; never the build-time placeholder |
| `AUTH_GOOGLE_ID` | K8s Secret `internal-secrets` | Real Google Cloud OAuth client |
| `AUTH_GOOGLE_SECRET` | K8s Secret `internal-secrets` | Real Google Cloud OAuth client |
| `AUTH_URL` | manifest env | `https://internal.onyxplus.{env}.satschel.com` |
| `AUTH_TRUST_HOST` | manifest env | `"true"` (NextAuth v5 strict-host check off for K8s) |

Google OAuth redirect URI: `https://internal.onyxplus.{env}.satschel.com/api/auth/callback/google`.

## Why `@hanzo/docs` is vendored

`@hanzo/docs@16.4.2` and sibling 16.7.5 packages were published to npm with unresolved `workspace:*` deps -- including `@hanzo/docs-tailwind` which is never published to npm at all. Neither npm nor pnpm can resolve `workspace:*` from a standalone consumer.

Fix in `onyx-plus/internal`:
1. Copy `~/work/hanzo/docs/packages/{hanzo-docs,base-ui,core,mdx,mdx-runtime,tailwind,radix-ui,content,content-collections}` to `vendor/hanzo-docs/packages/`
2. `pnpm-workspace.yaml` includes `vendor/hanzo-docs/packages/*` so `workspace:*` resolves locally
3. `.pnpmfile.cjs` strips `workspace:*` refs to packages NOT in the vendor dir (build-only tooling -- docs-cli, eslint-config-custom, tsconfig)

Long-term fix: upstream `@hanzo/docs` needs to either (a) add a `prepublishOnly` hook that rewrites `workspace:*` to concrete versions, (b) publish `@hanzo/docs-tailwind` to npm, or (c) use `pnpm publish` correctly with workspace protocol substitution. Once upstream is clean, `internal/` drops the vendor dir.

## App-code fixes for the vendor setup

- `app/layout.tsx`: wrap `<RootProvider>` (from `@hanzo/docs/ui/provider/base`) with `<NextProvider>` (from `@hanzo/docs-core/framework/next`). The published umbrella's `provider/next` re-export was missing.
- `lib/source.ts`: call `docs.toSource()` not `docs.toFumadocsSource()` -- the published `@hanzo/docs-mdx` rename hadn't propagated.
- `next.config.mjs`: `outputFileTracingRoot: import.meta.dirname` + `turbopack.root: import.meta.dirname` so standalone server.js lands at `.next/standalone/server.js` (default Turbopack behaviour put it at `.next/standalone/work/onyxplus/internal/server.js`).
- `next.config.mjs`: `typescript.ignoreBuildErrors: true` -- `@hanzo/docs` version skew injects `body`/`toc` on `page.data` at runtime that don't appear in the schema's static types.

## Dockerfile pattern

```dockerfile
FROM node:24-alpine AS deps
WORKDIR /src
RUN corepack enable && corepack prepare pnpm@10 --activate
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml .pnpmfile.cjs ./
COPY vendor ./vendor
RUN pnpm install --frozen-lockfile

FROM node:24-alpine AS build
WORKDIR /src
ENV NEXT_TELEMETRY_DISABLED=1 AUTH_SECRET=build-time-placeholder
RUN corepack enable && corepack prepare pnpm@10 --activate
COPY --from=deps /src/node_modules ./node_modules
COPY --from=deps /src/vendor ./vendor
COPY . .
RUN pnpm exec docs-mdx source.config.ts .source && pnpm build

FROM node:24-alpine AS runtime
RUN addgroup -S internal && adduser -S -G internal -h /app internal
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build --chown=internal:internal /src/.next/standalone ./
COPY --from=build --chown=internal:internal /src/.next/static ./.next/static
COPY --from=build --chown=internal:internal /src/public ./public
USER internal
EXPOSE 3000
CMD ["node", "server.js"]
```

## K8s

`Deployment internal` + `Service internal` + ingress rule for `internal.onyxplus.{env}.satschel.com` in `admin/universe/k8s/{dev,test,main}/`:

```yaml
spec:
  containers:
    - name: internal
      image: us-docker.pkg.dev/onyxplus-registry/onyx-plus/internal:0.1.1
      ports:
        - containerPort: 3000
      env:
        - name: AUTH_URL
          value: https://internal.onyxplus.{env}.satschel.com
        - name: AUTH_TRUST_HOST
          value: "true"
      envFrom:
        - secretRef:
            name: internal-secrets
      readinessProbe:
        httpGet: { path: /signin, port: 3000 }
```

Main runs 2 replicas; dev/test run 1.

## Smoke

```bash
POD=$(kubectl -n onyxplus get pod -l app=internal -o jsonpath='{.items[0].metadata.name}')
kubectl -n onyxplus exec $POD -- wget -qO- http://127.0.0.1:3000/signin | grep -o 'OnyxPlus — Internal'
# Expect: OnyxPlus — Internal

kubectl -n onyxplus exec $POD -- wget -qO- -S http://127.0.0.1:3000/docs/iam 2>&1 | head -3
# Expect: HTTP/1.1 307 Temporary Redirect
#         location: https://internal.onyxplus.{env}.satschel.com/signin?callbackUrl=%2Fdocs%2Fiam
```

The 307 confirms the auth gate is correctly intercepting unauthenticated `/docs/*` traffic.

## Source pointers

| Subject | Source |
|---|---|
| NextAuth config | `~/work/onyxplus/internal/auth.ts` |
| Allowed domains | `~/work/onyxplus/internal/lib/shared.ts::allowedEmailDomains` |
| Proxy (auth gate + markdown rewrites) | `~/work/onyxplus/internal/proxy.ts` |
| Source loader | `~/work/onyxplus/internal/lib/source.ts` |
| Next config | `~/work/onyxplus/internal/next.config.mjs` |
| Pnpm workspace | `~/work/onyxplus/internal/pnpm-workspace.yaml` |
| Workspace-strip hook | `~/work/onyxplus/internal/.pnpmfile.cjs` |
| MDX content | `~/work/onyxplus/internal/content/docs/*.mdx` |
| Sidebar order | `~/work/onyxplus/internal/content/docs/meta.json` |

## Related skills

- `hanzo/hanzo-docs.md` - upstream framework
- `onyx-plus/onyx-plus-deploy.md` - Cloud Build + K8s rollout
- `onyx-plus/onyx-plus.md` - umbrella

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: docs, nextjs, hanzo-docs, nextauth, google-oauth, satschel
