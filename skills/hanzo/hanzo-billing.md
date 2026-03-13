# Hanzo Billing - Subscription and Payment Management Portal

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-commerce-api.md`, `hanzo/hanzo-ui.md`

## Overview

Hanzo Billing is a **multi-org billing portal** for managing subscriptions, payments, credits, invoices, and usage across the Hanzo ecosystem. Static-exported Next.js 16 app with `@hanzo/ui` components, RainbowKit wallet integration, and IAM OIDC/PKCE auth. Served via nginx on K8s. Live at `billing.hanzo.ai` (also `commerce.hanzo.ai`).

### Why Hanzo Billing?

- **Multi-org**: Auto-detects org from hostname (hanzo, lux, zoo, pars) with per-org IAM, branding, and theme
- **Subscription tiers**: Developer (free/$5 credit), Pro ($49/mo), Team ($199/mo), Enterprise, Custom
- **Web3 payments**: RainbowKit + wagmi + viem for wallet-based payments across 7 chains
- **IAM-integrated**: OIDC/PKCE login via Casdoor (hanzo.id, lux.id, zoo.id, pars.id)
- **Commerce API**: Backend via `api.hanzo.ai` for subscriptions, invoices, credits, transactions
- **Static export**: `next build` produces static HTML served by nginx (no Node.js runtime)

### Tech Stack

- **Framework**: Next.js 16 (static export via `output: 'export'`)
- **UI**: `@hanzo/ui` (billing components), Tailwind CSS 4, Geist font
- **Web3**: RainbowKit 2, wagmi 2, viem 2 (Ethereum, Polygon, Optimism, Arbitrum, Base, Avalanche, BSC)
- **Auth**: Self-contained OIDC/PKCE module (`lib/iam-auth.ts`), no external auth package
- **Serving**: nginx:alpine with security headers, gzip, health check at `/health`
- **Image**: `ghcr.io/hanzoai/billing:latest`
- **Dev port**: 3005

### OSS Base

Repo: `hanzoai/billing`.

## When to use

- Managing subscription plans and billing for Hanzo/Lux/Zoo/Pars orgs
- Viewing and paying invoices
- Adding/managing payment methods (card + crypto wallet)
- Topping up API credits
- Viewing transaction history and usage
- Configuring spend alerts
- Managing business profile and tax compliance
- Account member management with role-based access

## Hard requirements

1. **Hanzo IAM** configured at hanzo.id (or per-org IAM server) for OIDC/PKCE auth
2. **Commerce API** at `api.hanzo.ai` for backend billing operations
3. **GHCR access** for container image (`ghcr.io/hanzoai/billing`)

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://billing.hanzo.ai` |
| Alt domain | `https://commerce.hanzo.ai` |
| Dev server | `pnpm dev` (port 3005) |
| Build | `pnpm build` (static export to `out/`) |
| Image | `ghcr.io/hanzoai/billing:latest` |
| K8s namespace | `hanzo` |
| K8s port | 80 (nginx) |
| Auth | OIDC/PKCE via hanzo.id |
| Commerce API | `api.hanzo.ai` |
| Repo | `github.com/hanzoai/billing` |
| Package | `@hanzo/billing` (private, v0.1.0) |

## One-file quickstart

### Local development

```bash
git clone https://github.com/hanzoai/billing.git
cd billing
pnpm install
pnpm dev          # http://localhost:3005
```

### Production build

```bash
pnpm build        # Static export to out/
# Serve with nginx or any static file server
```

### Docker

```bash
# Build
docker build -t billing .

# Or use pre-built deploy image (requires out/ already built)
docker build -f Dockerfile.deploy -t billing .

# Run
docker run -p 80:80 billing
```

## Core Concepts

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  billing.hanzo.ai│────>│  Hanzo IAM       │     │  Commerce API   │
│  (Next.js static)│     │  (hanzo.id)      │     │  (api.hanzo.ai) │
│  served by nginx │     │  OIDC/PKCE auth  │     │  billing backend│
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
    ┌────┴──────────┐
    │  RainbowKit   │
    │  (7 EVM chains)│
    └───────────────┘
```

### Multi-Org Configuration

The app auto-detects organization from the hostname:

| Hostname contains | Org | IAM Server | Theme Color |
|-------------------|-----|------------|-------------|
| `lux` | Lux | lux.id | `#6366f1` (indigo) |
| `zoo` | Zoo | zoo.id | `#22c55e` (green) |
| `pars` | Pars | pars.id | `#f59e0b` (amber) |
| default | Hanzo | hanzo.id | `#ffffff` (white) |

All orgs share `api.hanzo.ai` as their Commerce API backend.

### Auth Flow

Self-contained OIDC/PKCE implementation in `lib/iam-auth.ts`:
1. User lands on billing portal, org detected from hostname
2. If not logged in, starts PKCE flow: generates `code_verifier`, computes `code_challenge`
3. Redirects to IAM authorize endpoint (`/oauth/authorize`) with S256 challenge
4. On callback, exchanges authorization code for tokens via `/oauth/token`
5. JWT stored in sessionStorage; user info parsed from JWT payload
6. Also supports implicit flow (access_token in query/hash)

### Subscription Plans

Five tiers defined in `lib/config.ts`, synced with Commerce API:

| Plan | Price | Requests/min | Tokens/min |
|------|-------|-------------|------------|
| Developer | Free ($5 credit) | 60 | 100K |
| Pro | $49/mo ($39 annual) | 500 | 1M |
| Team | $199/mo ($159 annual) | 2,000 | 5M |
| Enterprise | $9,999/mo | 50,000 | 100M |
| Custom | Contact sales | Custom | Custom |

### App Routes

| Route | Purpose |
|-------|---------|
| `/` | Main billing dashboard (BillingShell) |
| `/auth/callback` | OAuth callback handler |
| `/plans` | Subscription plan selection |
| `/commerce` | Commerce dashboard with admin components |
| `/topup` | Credit top-up page |
| `/transactions` | Transaction history |
| `/payment-methods` | Payment method management |

### Key Billing Components (lib/billing/)

- `overview-dashboard.tsx` - Main dashboard with balance, usage, plan summary
- `subscription-portal.tsx` - Plan selection and management
- `payment-manager.tsx` - Card and wallet payment methods
- `invoice-manager.tsx` - Invoice viewing and payment
- `cost-explorer.tsx` - Detailed cost breakdown
- `credits-panel.tsx` - Credit grants and balance
- `transactions-panel.tsx` - Transaction history
- `spend-alerts.tsx` - Budget alert configuration
- `account-members.tsx` - Team member management
- `business-profile-panel.tsx` - Business info for invoicing
- `tax-compliance-panel.tsx` - Tax ID and compliance
- `card-form.tsx` / `square-card-form.tsx` - Card input forms

### Admin Access

Admin emails (`admin@hanzo.ai`, `zach@hanzo.ai`, `ant@hanzo.ai`) have super-user billing access and can grant credits to any user. Checked via `isAdminUser()` in `lib/config.ts`.

### CI/CD

Three deployment targets from `ci.yml`:
1. **GitHub Pages**: Static export with `/billing` basePath (for docs/preview)
2. **Docker (GHCR + K8s)**: Builds `ghcr.io/hanzoai/billing:latest`, deploys to `hanzo` namespace
3. **Cloudflare Pages**: Optional (gated by `DEPLOY_CLOUDFLARE` var), credentials fetched from KMS

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Redirecting to sign in" loop | IAM callback URL mismatch | Verify redirect_uri matches `/auth/callback` path |
| Wallet connect fails | Missing WalletConnect project ID | Set `NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID` |
| Wrong org theme | Hostname detection | Check `getOrgByHost()` logic in `lib/config.ts` |
| Static export error | SSR code in client components | Use `'use client'` directive, lazy-load wallet providers |
| GitHub Pages 404 | Missing basePath | Set `GITHUB_PAGES=true` for `/billing` prefix |

## Related Skills

- `hanzo/hanzo-id.md` - IAM and authentication (OIDC/PKCE provider)
- `hanzo/hanzo-commerce-api.md` - Commerce API backend
- `hanzo/hanzo-ui.md` - Shared UI components (`@hanzo/ui/billing`)
- `hanzo/hanzo-payments.md` - Payment processing
- `hanzo/hanzo-cloud.md` - Cloud dashboard

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: billing, subscriptions, payments, commerce, wallet
**Prerequisites**: Node.js 20, pnpm, Hanzo IAM credentials
