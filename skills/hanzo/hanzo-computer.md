# Hanzo Computer - AI Hardware Marketplace

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-billing.md`

## Overview

Hanzo Computer (`hanzoai/computer`) is an **ecommerce web application for purchasing AI compute hardware**. It sells the DGX Spark instance ($4,000) as the only credit-card purchasable product, with GPU On-Demand and Enterprise tiers requiring sales consultation. The storefront includes a shopping cart, checkout flow (via Hanzo Billing), order management, and an account dashboard.

### What it actually is

- A React 19 + TypeScript + Vite single-page application
- Deployed to GitHub Pages (and optionally Vercel for serverless API routes)
- No backend database in production -- cart and orders stored in LocalStorage (demo mode)
- Supabase schema exists for future production use (orders, customers, products, analytics)
- Vercel serverless API functions in `api/` for email, invoices, and rate limiting
- Payments redirect to Hanzo Billing (billing.hanzo.ai) -- no PCI data handled in-app
- Auth via Hanzo IAM (hanzo.id) with client ID `app-computer`

### What it is NOT

- Not a "computer use" agent or desktop automation tool (that is `hanzoai/operative`)
- Not a general-purpose ecommerce platform
- No backend order processing in the current deployment

## When to use

- Building or modifying the hanzo.computer storefront
- Adding new hardware products to the catalog
- Integrating Hanzo Billing payment flows
- Implementing the production backend (orders, auth, emails)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/computer` |
| Package | `hanzo-computer` (private) |
| Version | 0.0.0 |
| Branch | `main` |
| Language | TypeScript |
| Framework | React 19 + Vite 6 |
| Styling | Tailwind CSS v4 |
| Routing | React Router v7 |
| Payments | Hanzo Billing (billing.hanzo.ai) |
| Auth | Hanzo IAM (hanzo.id), client `app-computer` |
| Dev | `npm run dev` (port 5173) |
| Build | `npm run build` |
| Test | `npm run test` (Playwright) |
| Deploy | GitHub Pages (auto on push to main) |
| API | Vercel serverless functions in `api/` |
| License | Proprietary (Hanzo AI Inc.) |

## Project structure

```
hanzoai/computer/
  package.json              # hanzo-computer, React 19, Vite 6
  App.tsx                   # Main app with React Router
  index.html                # SPA entry point
  index.tsx                 # React DOM render
  vite.config.ts            # Vite config
  vercel.json               # Vercel serverless function config
  tailwind.config.js        # Tailwind CSS v4
  .env.example              # IAM, Commerce API, Upstash Redis config
  api/
    _middleware.ts           # Rate limiting middleware (Upstash Redis)
    generate-invoice.ts     # PDF invoice generation
    send-email.ts           # Generic email sending
    send-order-confirmation.ts
    send-quote-email.ts
    send-rfq-confirmation.ts
    send-subscription-confirmation.ts
    send-cluster-notification.ts
    admin/                  # Admin API endpoints
  components/
    Header.tsx              # Navigation with cart icon + search
    Hero.tsx                # Landing page hero section
    Pricing.tsx             # Product cards (DGX Spark, GPU, Enterprise)
    CloudPricing.tsx        # Cloud compute pricing
    DGXSparkHighlight.tsx   # Featured product section
    Features.tsx            # Feature highlights
    HardwareSpec.tsx        # Hardware specifications table
    WhyBuyHardware.tsx      # Marketing content
    UseCases.tsx            # Use case showcases
    IndustrySolutions.tsx   # Industry vertical solutions
    FAQ.tsx                 # Frequently asked questions
    SearchModal.tsx         # Product search
    Testimonials.tsx        # Customer testimonials
    TrustSecurity.tsx       # Security badges and info
    Partners.tsx            # Partner logos
    ImageGallery.tsx        # Product images
    CallToAction.tsx        # CTA section
    Footer.tsx              # Site footer
  src/
    context/
      CartContext.tsx        # Shopping cart state (React Context)
    pages/
      Cart.tsx              # Cart page with quantity controls
      Checkout.tsx          # Checkout → Hanzo Billing redirect
      Account.tsx           # Order history dashboard
    components/             # Additional UI components
    lib/                    # Utility libraries
  supabase/
    schema.sql              # Production DB schema (orders, customers, products)
    schema_analytics.sql    # Analytics tables
    storage-buckets.sql     # File storage config
    migrations/             # DB migrations
  e2e/                      # Playwright end-to-end tests
  tests/                    # Test suite
  docs/                     # Documentation
  public/                   # Static assets
```

## Dependencies

From `package.json`:
- `react` ^19.2.0, `react-dom` ^19.2.0
- `react-router-dom` ^7.9.5
- `lucide-react` ^0.552.0, `@heroicons/react` ^2.2.0 -- icons
- `recharts` ^3.3.0 -- dashboard charts
- `@react-pdf/renderer` ^4.3.1 -- invoice PDF generation
- `@upstash/ratelimit` ^2.0.6, `@upstash/redis` ^1.35.6 -- API rate limiting
- `date-fns` ^4.1.0 -- date formatting
- `express-rate-limit` ^8.2.1 -- fallback rate limiting

Dev: Vite 6, TypeScript 5.8, Tailwind CSS v4, Playwright, Jest

## Quickstart

```bash
git clone https://github.com/hanzoai/computer.git
cd computer
cp .env.example .env
# Edit .env with IAM and billing config
npm install
npm run dev
# Open http://localhost:5173
```

## Environment variables

```bash
# Hanzo IAM
VITE_IAM_URL=https://hanzo.id
VITE_IAM_CLIENT_ID=app-computer

# Commerce API
VITE_COMMERCE_API_URL=https://billing.hanzo.ai

# Site
VITE_SITE_URL=https://hanzo.computer

# Upstash Redis (API rate limiting)
UPSTASH_REDIS_REST_URL=https://your-redis.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token

# Admin
ADMIN_API_KEY=your-admin-key
```

## Shopping flow

1. Browse products on homepage (DGX Spark, GPU On-Demand, Enterprise)
2. Add DGX Spark to cart (only credit-card purchasable item)
3. Cart page with quantity controls and total
4. Checkout collects billing details, redirects to Hanzo Billing portal
5. Order confirmation on Account page
6. Instance provisioned in 24-48 hours

## Troubleshooting

- **Cart not persisting**: Data stored in LocalStorage, cleared on browser data wipe
- **Checkout redirect fails**: Verify `VITE_COMMERCE_API_URL` points to billing.hanzo.ai
- **API rate limiting errors**: Configure Upstash Redis env vars
- **E2E tests fail**: Ensure Playwright browsers installed (`npx playwright install`)

## Related Skills

- `hanzo/hanzo-billing.md` -- Payment processing backend
- `hanzo/hanzo-commerce.md` -- Commerce API
- `hanzo/hanzo-iam.md` -- Authentication (hanzo.id)
- `hanzo/hanzo-ui.md` -- Shared UI components

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ecommerce, hardware, gpu, dgx, marketplace, billing
**Prerequisites**: Node.js 18+, npm
