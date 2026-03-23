# Hanzo Billing - Subscription and Payment Portal

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-commerce-api.md`, `hanzo/hanzo-ui.md`

## Overview

Hanzo Billing is a **multi-org billing portal** for managing subscriptions, payments, credits, invoices, and usage across the Hanzo ecosystem. Static-exported Next.js 16 app with `@hanzo/ui` components, RainbowKit wallet integration, and IAM OIDC/PKCE auth. Live at `billing.hanzo.ai`.

## When to use

- Managing subscription plans and billing for Hanzo/Lux/Zoo/Pars orgs
- Viewing and paying invoices
- Adding/managing payment methods (card + crypto wallet)
- Topping up API credits
- Viewing transaction history and usage
- Configuring spend alerts

## Hard requirements

1. **IAM configured** at hanzo.id (OIDC/PKCE auth)
2. **Commerce API** at `api.hanzo.ai` for backend billing operations
3. **Static export** -- no Node.js runtime in production (served via Traefik static plugin)
4. **No nginx, no caddy** -- static serving handled by the container's built-in HTTP server

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://billing.hanzo.ai` (alias: `commerce.hanzo.ai`) |
| Framework | Next.js 16 (static export) |
| UI | `@hanzo/ui`, Tailwind CSS 4, Geist font |
| Web3 | RainbowKit 2, wagmi 2, viem 2 |
| Auth | Self-contained OIDC/PKCE module (`lib/iam-auth.ts`) |
| Serving | nginx:alpine (security headers, gzip, `/health`) |
| Image | `ghcr.io/hanzoai/billing:latest` |
| Dev port | 3005 |
| Repo | `github.com/hanzoai/billing` |
| K8s manifests | `universe/infra/k8s/billing/` |

## Multi-org support

Auto-detects org from hostname with per-org IAM, branding, and theme:

| Hostname | Org | IAM |
|----------|-----|-----|
| `billing.hanzo.ai` | hanzo | hanzo.id |
| `billing.lux.network` | lux | lux.id |
| `billing.zoo.ngo` | zoo | zoo.id |
| `billing.pars.id` | pars | pars.id |

## Subscription tiers

| Tier | Price | Includes |
|------|-------|----------|
| Developer | Free / $5 credit | 1M tokens/mo, basic models |
| Pro | $49/mo | 10M tokens/mo, all models, priority |
| Team | $199/mo | 100M tokens/mo, team management, SLA |
| Enterprise | Custom | Unlimited, dedicated support, SLA |

## Web3 payments

RainbowKit + wagmi + viem for wallet-based payments across 7 chains:
- Ethereum, Polygon, Optimism, Arbitrum, Base, Avalanche, BSC

## Auth flow (OIDC/PKCE)

Self-contained PKCE auth module -- no external auth package:

```typescript
// lib/iam-auth.ts
// 1. Generate code_verifier and code_challenge
// 2. Redirect to hanzo.id/oauth/authorize with PKCE params
// 3. Exchange code for token at /oauth/token
// 4. Store token in httpOnly cookie
```

## Development

```bash
cd ~/work/hanzo/billing
pnpm install

# Development
pnpm dev  # Port 3005

# Build (static export)
pnpm build  # Output: out/

# Docker build
docker build -t ghcr.io/hanzoai/billing:latest .
```

## K8s deployment

```yaml
# universe/infra/k8s/billing/
apiVersion: apps/v1
kind: Deployment
metadata:
  name: billing
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: billing
          image: ghcr.io/hanzoai/billing:latest
          ports:
            - containerPort: 80
          livenessProbe:
            httpGet:
              path: /health
              port: 80
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Auth redirect loop | Wrong IAM client_id | Check OIDC config matches IAM app |
| Wallet not connecting | Wrong chain ID | Verify RainbowKit chain config |
| Stale UI after deploy | Cached `.gz` files | Update both `.html` and `.html.gz` |
| 404 on client routes | SPA fallback not configured | Ensure static serving handles SPA routing |

## Related Skills

- `hanzo/hanzo-id.md` -- IAM for authentication
- `hanzo/hanzo-commerce-api.md` -- Backend billing API
- `hanzo/hanzo-ui.md` -- Component library
- `hanzo/hanzo-static.md` -- Static file serving

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: billing, payments, subscriptions, web3, commerce, stripe
**Prerequisites**: Next.js, Tailwind CSS, OIDC concepts
