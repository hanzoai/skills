# Hanzo Static Sites, CI/CD & Commerce Deployment Guide

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-deploy.md`, `hanzo/hanzo-website.md`, `hanzo/hanzo-static.md`, `hanzo/hanzo-commerce.md`, `hanzo/hanzo-iam.md`

## Overview

This skill provides the unified, authoritative standard for building, configuring, deploying, and managing all static sites and web applications across the Hanzo ecosystem. It covers:
1. **Deterministic Project Configuration** via `hanzo.yml` (or `hanzo.config.toml`, `~/.hanzo/settings.json`, `hanzo.config.ts`).
2. **Dual-Workflow CI/CD Synchronization** (`.github/workflows` and `.hanzo/workflows` on push to forge `git.hanzo.ai`).
3. **Sites Plane & Native Deployment** via `bin/site <slug> <dir>`, Docker container images (`ghcr.io/hanzoai/static:0.5.15`), and native `/v1/projects/sites/deploy`.
4. **Commerce & Checkout Integration**: **STRICT NO-STRIPE POLICY**. All payments must route through Hanzo Commerce (`https://api.hanzo.ai/v1/commerce` and `https://pay.hanzo.ai`) backed by Square (Sandbox & Production), Crypto, and Wire.

---

## 1. Hard Requirements & Principles

1. **NO STRIPE**: Stripe is completely prohibited and deprecated across the Hanzo estate. All billing and card processing goes exclusively through Hanzo Commerce (`api.hanzo.ai/v1/commerce` / `pay.hanzo.ai`) using Square, Crypto, or Wire.
2. **Deterministic Configuration**: Every repo defines its site metadata, framework, build target, domains, commerce, and IAM settings in `hanzo.yml` (or `hanzo.config.ts` / `hanzo.config.toml`).
3. **Dual Workflow Plane**:
   - `git.hanzo.ai` (the forge) reads `.hanzo/workflows` and `.github/workflows`.
   - Public mirrors (GitHub) read `.github/workflows`.
   - Always keep `.hanzo/workflows` synchronized with `.github/workflows` (e.g. symlinked via `ln -s ../.github/workflows .hanzo/workflows`) so pushes to both remotes trigger identical CI/CD.
4. **Single Identity System**: Authentication and account/order management is powered by Hanzo ID (`https://hanzo.id`) via OIDC PKCE, and billing is managed at `https://billing.hanzo.ai`.
5. **Standard Checkout Contract**: Pricing CTAs must link to:
   ```
   https://pay.hanzo.ai/?plan=<planId>&return=<returnUrl>
   ```

---

## 2. Deterministic Configuration Schema (`hanzo.yml`)

Place `hanzo.yml` in the root of any site or application repo:

```yaml
name: agency
slug: agency
version: 0.1.6
type: site # site | app | service

framework:
  name: vite # vite | next | astro | static | react
  preset: react
  buildCommand: pnpm build
  outputDir: dist # dist | out | build | public
  singlePageApp: true # true for BrowserRouter / SPA routing fallback

routing:
  domain: hanzo.agency
  aliases:
    - www.hanzo.agency
    - agency.hanzo.ai

deploy:
  plane: sites # sites (S3/SeaweedFS static plane) | k8s (container image)
  publishScript: https://git.hanzo.ai/hanzoai/ci/.github/actions/site@v1
  image: ghcr.io/hanzoai/agency # when built as container
  kubernetes:
    namespace: hanzo
    service: agency
    port: 3000

commerce:
  org: hanzo
  checkoutUrl: https://pay.hanzo.ai
  commerceApi: https://api.hanzo.ai/v1/commerce
  providers:
    - square # Square Sandbox/Production Web Payments SDK
    - crypto # BTC, ETH, USDC
    - wire   # Bank wire (ACH / SWIFT)
  disallowed:
    - stripe # STRICT RULE: No Stripe

iam:
  issuer: https://hanzo.id
  clientId: hanzo-agency
  billingPortal: https://billing.hanzo.ai

calendar:
  bookingUrl: https://calendar.app.google/z1YsZQrqR4s6jQqD8

analytics:
  googleAnalyticsId: G-BCXWVKFL1K
  insightsUrl: https://api.hanzo.ai/v1/event
```

### TypeScript Alternative (`hanzo.config.ts`)
For TypeScript packages using `@hanzo/ai` or `hanzo`:
```typescript
import { defineConfig } from '@hanzo/config';

export default defineConfig({
  name: 'agency',
  slug: 'agency',
  type: 'site',
  framework: 'vite',
  build: {
    command: 'pnpm build',
    output: 'dist',
    spa: true,
  },
  commerce: {
    org: 'hanzo',
    checkoutUrl: 'https://pay.hanzo.ai',
    providers: ['square', 'crypto', 'wire'],
  },
  iam: {
    issuer: 'https://hanzo.id',
    clientId: 'hanzo-agency',
  },
});
```

---

## 3. Deployment Modes

### Mode A: The Sites Plane (`bin/site` via CI Action)
The static export plane sends zero runtime compute to Kubernetes; files are served directly from S3/SeaweedFS edge cache:
1. **Self-provision**: `POST https://api.hanzo.ai/v1/projects` (creates or confirms project slug).
2. **Enqueue**: `POST https://api.hanzo.ai/v1/projects/<slug>/deployments` -> returns `{id, bucket, prefix, upload: {url, fields}}`.
3. **Upload**: Direct multipart POST to S3 presigned URL in parallel (assets first, then HTML documents as the activation switch).
4. **Complete**: `POST https://api.hanzo.ai/v1/projects/<slug>/deployments/<id>/complete` with the key manifest.

### Mode B: Containerized Static Server (`ghcr.io/hanzoai/static:0.5.15`)
For deployments requiring a Kubernetes Service and Ingress:
```dockerfile
# Dockerfile
FROM node:22 AS build
WORKDIR /src
RUN npm i -g pnpm@9
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

FROM ghcr.io/hanzoai/static:0.5.15
COPY --from=build /src/dist /public
ENTRYPOINT ["/static", "-spa"]
```

### Mode C: Native Agent Deployment (`/v1/projects/sites/deploy`)
Agents and tools can programmatically publish a static site in one API call:
```bash
curl -X POST "https://api.hanzo.ai/v1/projects/sites/deploy" \
  -H "Authorization: Bearer $HANZO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "my-site",
    "name": "My Site",
    "files": [
      {"path": "index.html", "content": "<!DOCTYPE html><html>...</html>"},
      {"path": "style.css", "content": "body { background: #000; color: #fff; }"}
    ]
  }'
```

---

## 4. Checkout & Payment Flow (No Stripe)

### Architecture
```
User clicks "Get Started" / Plan CTA
  │
  ├──> https://pay.hanzo.ai/?plan=agency&return=https://hanzo.agency/payment-success
  │       │
  │       ├── Square Web Payments SDK (Sandbox or Production)
  │       ├── Crypto Rails (BTC, ETH, USDC)
  │       └── Bank Wire Instructions
  │
  └──> POST https://api.hanzo.ai/v1/billing/topup/token
          │
          └── Hanzo Commerce Engine (authorizes & captures via Square)
```

### In Code (`src/lib/commerce.ts`)
```typescript
const PAY = 'https://pay.hanzo.ai';
const COMMERCE = 'https://api.hanzo.ai/v1/commerce';
const ORG = 'hanzo';

export function checkoutUrl(planId?: string, returnPath = '/payment-success'): string {
  if (!planId) return PAY;
  const back = new URL(returnPath, window.location.origin).toString();
  return `${PAY}/?plan=${encodeURIComponent(planId)}&return=${encodeURIComponent(back)}`;
}
```

### Account & Order Management
- Users view orders, subscriptions, and payment history at `https://billing.hanzo.ai` or `https://hanzo.id`.
- Single Sign-On and session management route through `https://hanzo.id` (`/v1/iam/oauth/authorize`).

---

## 5. Verification Checklist for Any Site

- [ ] `hanzo.yml` exists and defines `slug`, `framework`, `routing`, `commerce`, and `iam`.
- [ ] `.github/workflows` and `.hanzo/workflows` are synchronized.
- [ ] No Stripe packages (`@stripe/*`), scripts (`js.stripe.com`), or keys exist in codebase.
- [ ] All pricing buttons point to `checkoutUrl(planId)` resolving to `pay.hanzo.ai`.
- [ ] Square sandbox credentials and environment match `GET /v1/commerce/org?org=hanzo`.
- [ ] Google Analytics (`G-BCXWVKFL1K` or repo tag) tracks both initial load and SPA route changes.
- [ ] Auth / Login buttons link to `https://hanzo.id` or use Hanzo IAM OIDC.
- [ ] Calendar booking link points to official Google Calendar (`https://calendar.app.google/z1YsZQrqR4s6jQqD8`).
