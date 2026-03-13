# Hanzo Store - AI Agent & MCP Server Marketplace

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-desktop.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Store is a **flatfile-based marketplace** for discovering and installing AI agent tools and MCP servers. Built with Next.js 15 and static site generation, it serves as the app store for Hanzo Desktop. Each app is a JSON file in `data/apps/`, and new submissions happen via Pull Requests. Includes Web3 wallet integration via RainbowKit/wagmi for token-gated features.

### What it actually is

- Static Next.js 15 site with client-side filtering
- Flatfile database: one JSON per app in `data/agents/` and `data/tools/`
- Git-based submission workflow (fork, add JSON, PR)
- Build script generates `public/store.json` from flatfiles
- Web3 integration: RainbowKit, wagmi v2, viem v2, WalletConnect
- Playwright E2E tests
- Uses `@hanzo/ui` and `@hanzo/logo` component libraries
- Deployed at `store.hanzo.ai`

### What it is NOT

- Not a backend service or API -- it is a static site
- Not handling payments or token transfers -- Web3 is for wallet identity only
- Not a package registry -- apps link to external repos/install commands

## When to use

- Adding a new AI agent or MCP server to the Hanzo ecosystem marketplace
- Browsing available tools for Hanzo Desktop
- Building a custom store frontend that consumes `store.json`

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/store` |
| Package | `hanzo-store` (private) |
| Version | 0.1.0 |
| Branch | `main` |
| Stack | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Web3 | wagmi v2, viem v2, RainbowKit |
| Dev Port | 3200 |
| Live | `store.hanzo.ai` |
| License | MIT |
| Tests | Playwright E2E |

## Quickstart

```bash
git clone https://github.com/hanzoai/store.git
cd store
npm install

# Configure
cp .env.example .env
# Set NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID

# Generate store data from flatfiles
npm run generate-store

# Development
npm run dev    # http://localhost:3200

# Production build (static export to out/)
npm run build
```

## Architecture

```
store.hanzo.ai
      |
  Next.js SSG
      |
  store.json (generated)
      |
  data/agents/*.json + data/tools/*.json
```

Hanzo Desktop fetches `https://store.hanzo.ai/store.json` for the catalog.

## Directory structure

```
hanzoai/store/
  package.json           # hanzo-store v0.1.0
  next.config.js         # Static export, ignores TS/ESLint errors
  app/
    layout.tsx           # Root layout
    page.tsx             # Main store page
    apps/[id]/
      page.tsx           # Server component (app detail)
      page-client.tsx    # Client component (app detail)
    providers.tsx        # QueryClient + Web3 providers
  data/
    agents/              # AI agents (one JSON per agent)
    tools/               # Automation tools (one JSON per tool)
  scripts/
    generate-store.js    # Reads data/ and writes public/store.json
  lib/
    wagmi.ts             # Web3 config (WalletConnect, chain definitions)
  types/
    index.ts             # App/store TypeScript types
    hanzo-ui.d.ts        # @hanzo/ui type shim
  components/            # React UI components
  public/
    store.json           # Generated catalog (gitignored)
  e2e/
    store.spec.ts        # Playwright E2E tests
  playwright.config.ts
```

## Adding an app

Create a JSON file in `data/agents/` or `data/tools/`:

```json
{
  "id": "my-server",
  "name": "My Server",
  "description": "What it does",
  "version": "1.0.0",
  "author": "Your Name",
  "category": "Development Tools",
  "tags": ["mcp", "tools"],
  "installCommand": "npx -y @you/my-server",
  "mcpConfig": {
    "command": "npx",
    "args": ["-y", "@you/my-server"]
  },
  "createdAt": "2026-01-01T00:00:00Z",
  "updatedAt": "2026-01-01T00:00:00Z"
}
```

Categories: Development Tools, Productivity, Data & Analytics, Communication, File Management, AI & Machine Learning, Security, Utilities, Entertainment, Other.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID` | WalletConnect project ID (required for Web3) |

## Known issues (from LLM.md)

- `next.config.js` ignores TS and ESLint errors during build
- Console.error calls in production code (page.tsx, page-client.tsx)
- No React Error Boundaries on main pages
- WalletConnect project ID falls back to hardcoded `'demo'`
- `@hanzo/ui` has no exported types (shimmed in `types/hanzo-ui.d.ts`)

## Related Skills

- `hanzo/hanzo-desktop.md` -- Desktop app that consumes the store
- `hanzo/hanzo-mcp.md` -- MCP servers listed in the store
- `hanzo/hanzo-extension.md` -- Browser/IDE extensions

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: store, marketplace, mcp, agents, tools, desktop
**Prerequisites**: Node.js, npm, WalletConnect project ID
