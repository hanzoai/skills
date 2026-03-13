# Hanzo Website - Corporate Marketing Site (hanzo.ai)

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-platform.md`

## Overview

The Hanzo AI corporate website at **hanzo.ai**. A Next.js 16 App Router site with static export (`output: 'export'`), React 19, Tailwind CSS 4, Framer Motion animations, and shadcn/ui components. Dark theme by default with monochrome brand palette. Over 60 marketing pages covering every Hanzo product (AI, Cloud, Chat, MCP, Commerce, Platform, Zen models, etc.). Uses `@hanzo/ui` and `@hanzo/logo` packages. Deployed as a static site.

### Tech Stack

- **Framework**: Next.js 16.1.6 (App Router, static export)
- **React**: 19.0.0
- **Styling**: Tailwind CSS 4.2.1 + CSS variables + golden ratio design system
- **Animations**: Framer Motion 12.4.2
- **UI Library**: shadcn/ui (v4) + Radix UI primitives
- **Charts**: Recharts 2.12.7
- **3D**: Three.js 0.134
- **Forms**: React Hook Form + Zod validation
- **Flow diagrams**: @xyflow/react 12.4.4
- **Testing**: Playwright (e2e)
- **Font**: Geist Sans + Geist Mono (via next/font/google)
- **Analytics**: Hanzo Insights (`insights.hanzo.ai`)
- **Package manager**: npm (has both package-lock.json and pnpm-lock.yaml)

### OSS Base

Repo: `github.com/hanzoai/hanzo.ai` (branch: `main`).

## When to use

- Making changes to the hanzo.ai marketing site
- Adding new product landing pages
- Updating branding, copy, or design system
- Working with the Hanzo component library patterns

## Hard requirements

1. **Node.js 20+** (`.nvmrc`)
2. **npm** or **pnpm** for package management
3. Static export -- no server-side features (no API routes, no SSR, no ISR)

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://hanzo.ai` |
| Framework | Next.js 16 (App Router, static export) |
| React | 19.0.0 |
| Tailwind | 4.2.1 |
| Package name | `@hanzo/site` (private) |
| Node | v20+ |
| Dev server | `npm run dev` |
| Build | `npm run build` (static export to `out/`) |
| Tests | `npm test` (Playwright) |
| Default theme | Dark |
| Brand palette | Monochrome (`--brand: #e4e4e7`) |
| Repo | `github.com/hanzoai/hanzo.ai` |

## One-file quickstart

### Local development

```bash
git clone https://github.com/hanzoai/hanzo.ai.git
cd hanzo.ai
npm install
npm run dev
# Open http://localhost:3000
```

### Add a new product page

```bash
# 1. Create the route directory
mkdir -p app/\(marketing\)/my-product

# 2. Create the page
cat > app/\(marketing\)/my-product/page.tsx << 'EOF'
import { MyProductHero } from '@/components/my-product/Hero'
import { MyProductFeatures } from '@/components/my-product/Features'

export default function MyProductPage() {
  return (
    <>
      <MyProductHero />
      <MyProductFeatures />
    </>
  )
}
EOF

# 3. Create the component directory
mkdir -p components/my-product
# Add Hero.tsx, Features.tsx following existing patterns
```

## Core Concepts

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Next.js 16 App Router (Static Export)                   │
│                                                          │
│  app/                                                    │
│    layout.tsx          Root layout (Geist font, theme)   │
│    (marketing)/                                          │
│      layout.tsx        Marketing layout (Navbar+Footer)  │
│      page.tsx          Homepage                          │
│      ai/page.tsx       /ai                               │
│      cloud/page.tsx    /cloud                            │
│      chat/page.tsx     /chat                             │
│      zen/page.tsx      /zen                              │
│      pricing/page.tsx  /pricing                          │
│      ... 60+ pages                                       │
│                                                          │
│  components/                                             │
│    Hero.tsx, Navbar.tsx, Footer.tsx, Features.tsx  (shared) │
│    ui/              shadcn/ui components                  │
│    ai/              AI product components                 │
│    cloud/           Cloud product components              │
│    zen/             Zen models components                 │
│    ... 40+ component directories                         │
│                                                          │
│  data/              Static data (JSON, TS)               │
│  contexts/          React contexts                       │
│  hooks/             Custom hooks                         │
│  lib/               Utility functions                    │
└─────────────────────────────────────────────────────────┘
```

### Design System

The site uses a golden-ratio-based design system:

- **Font scale**: Golden ratio (1.618x) progression: xs (0.75rem) through 6xl (11.089rem)
- **Spacing**: `golden-1` (0.25rem) through `golden-9` (11.749rem)
- **Grid**: `golden` template: 38.2% / 61.8% split
- **Brand colors**: Monochrome -- primary white, secondary neutral-300, hover neutral-400
- **CSS variables**: `--brand: #e4e4e7`, `--brand-muted: #a3a3a3`
- **Dark mode**: Default, enforced via `darkMode: "class"` + ThemeProvider

### Typography

| Level | Classes |
|-------|---------|
| h1 | `text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight` |
| h2 | `text-3xl md:text-5xl font-bold` |
| h3 | `text-2xl font-bold` |
| Body large | `text-xl text-neutral-300` |
| Body default | `text-lg text-neutral-400` |
| Caption | `text-sm text-neutral-500` |

### Key Hanzo Packages

| Package | Purpose |
|---------|---------|
| `@hanzo/ui` ^5.3.34 | Shared UI component library |
| `@hanzo/logo` ^1.0.5 | Hanzo logo SVG components |
| `@zenlm/models` ^1.0.2 | Zen model metadata and specs |

### Route Groups

All marketing pages live under `app/(marketing)/` which applies the shared marketing layout (Navbar + Footer). Major product routes:

- `/ai` - AI platform
- `/cloud` - Cloud infrastructure
- `/chat` - Hanzo Chat
- `/zen` - Zen LLM models
- `/mcp` - Model Context Protocol
- `/platform` - PaaS platform
- `/commerce` - Commerce/payments
- `/bot` - Hanzo Bot
- `/extension` - Browser/IDE extension
- `/pricing` - Pricing page
- `/about`, `/team`, `/careers`, `/contact` - Company pages
- `/sql`, `/kv`, `/vector`, `/datastore`, `/storage` - Database products
- `/auth`, `/identity`, `/kms` - Security products
- `/flow`, `/functions`, `/edge`, `/realtime`, `/pubsub` - Infrastructure

### Environment Variables

```bash
# IAM OAuth2 (from .env.example)
VITE_HANZO_IAM_URL=https://id.hanzo.ai
VITE_HANZO_CLIENT_ID=your-client-id
VITE_HANZO_REDIRECT_URI=http://localhost:8084/auth/callback

# Commerce API
VITE_HANZO_API_URL=https://api.hanzo.ai

# Payments
VITE_SQUARE_APPLICATION_ID=sandbox-sq0idb-xxx
VITE_SQUARE_LOCATION_ID=your-location-id

# Crypto
VITE_CRYPTO_DEPOSIT_ADDRESS=0x...
VITE_WALLETCONNECT_PROJECT_ID=your-project-id
```

**Note**: The `.env.example` uses `VITE_` prefixes (from the earlier Vite-based version). The current Next.js build uses `NEXT_PUBLIC_` prefix convention, but the env file has not been updated.

### Analytics

Hanzo Insights is embedded in the root layout via a script tag:
- Endpoint: `insights.hanzo.ai`
- Project key: `hi_e16a2d5a8033442d87f090b24c606825`
- Registers: `app: 'hanzo-ai'`, `org: 'hanzo'`

## Directory structure

```
hanzo.ai/
  package.json            # @hanzo/site v0.0.1 (private)
  next.config.ts          # Static export, unoptimized images
  tailwind.config.ts      # Golden ratio design system
  tsconfig.json           # TypeScript config
  postcss.config.js       # PostCSS with Tailwind
  components.json         # shadcn/ui config
  eslint.config.js        # ESLint config
  playwright.config.ts    # E2E test config
  .nvmrc                  # Node v20
  .env.example            # Environment variables template
  LLM.md                  # Agent documentation
  CLAUDE.md               # Agent instructions
  app/
    layout.tsx            # Root layout (fonts, theme, analytics)
    globals.css           # Global styles + CSS variables
    icon.svg              # Favicon
    not-found.tsx         # 404 page
    (marketing)/
      layout.tsx          # Marketing layout (Navbar + Footer)
      page.tsx            # Homepage
      about/              # About page
      ai/                 # AI product
      cloud/              # Cloud product
      chat/               # Chat product
      zen/                # Zen models
      mcp/                # MCP tools
      pricing/            # Pricing
      ... (60+ routes)
  components/
    Hero.tsx              # Homepage hero
    Navbar.tsx            # Site navigation
    Footer.tsx            # Site footer
    Features.tsx          # Feature grid
    Products.tsx          # Product cards
    CommandPalette.tsx    # Command palette (Cmd+K)
    GlobalChatWidget.tsx  # Embedded chat widget
    ui/                   # shadcn/ui components
    shadcn-v4/            # shadcn v4 components
    shared/               # Shared components
    navigation/           # Nav components
    animations/           # Animation components
    ai/                   # AI section components
    cloud/                # Cloud section components
    zen/                  # Zen section components
    ... (40+ directories)
  data/                   # Static data files
  contexts/               # React context providers
  hooks/                  # Custom React hooks
  lib/                    # Utility functions
  public/                 # Static assets
  scripts/                # Build/utility scripts
  e2e/                    # Playwright e2e tests
  test-results/           # Test output
  .github/
    workflows/            # GitHub Actions CI
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build warnings (ESLint/TS) | `ignoreDuringBuilds: true` in next.config | Warnings suppressed intentionally for static export |
| Images not loading | `unoptimized: true` in config | Expected for static export -- use standard `<img>` or next/image with unoptimized |
| VITE_ env vars | Legacy from Vite era | Use `NEXT_PUBLIC_` prefix for Next.js client-side vars |
| Merge conflict in README | Unresolved `<<<<<<<` markers present | README has an active merge conflict on main branch |
| pnpm vs npm | Both lock files present | Use npm (package-lock.json is more recent) |
| Dark mode flash | ThemeProvider not wrapping content | Ensure ThemeProvider is in root layout with `suppressHydrationWarning` on html |

## Related Skills

- `hanzo/hanzo-cloud.md` - Cloud dashboard (cloud.hanzo.ai)
- `hanzo/hanzo-platform.md` - PaaS platform (platform.hanzo.ai)
- `hanzo/hanzo-chat.md` - Chat product (chat.hanzo.ai)
- `hanzo/hanzo-commerce.md` - Commerce frontend

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: website, marketing, next.js, tailwind, react, landing-pages
**Prerequisites**: Node.js 20+, npm
