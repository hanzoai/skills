# Hanzo Docs - Multi-Brand Documentation Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/python-sdk.md`, `hanzo/hanzo-brand.md`

## Overview

Hanzo Docs is a **multi-brand documentation platform** — fork of fumadocs with 27 packages and 16 apps. Serves documentation for Hanzo, Lux, Zoo, and Zen brands from a single monorepo with shared components and per-brand theming.

**NOTE**: This is a **fumadocs fork** (not a plain Next.js MDX site). It's a monorepo with **27 packages** and **16 apps**, supporting multi-brand documentation (hanzo.ai/docs, docs.lux.network, docs.zoo.ngo, docs.zenlm.org).

### Why Hanzo Docs?

- **fumadocs fork**: Full-featured documentation framework
- **Multi-brand**: 4 brands from one codebase (Hanzo, Lux, Zoo, Zen)
- **27 packages**: Shared components, MDX plugins, search, themes
- **16 apps**: Brand-specific documentation sites
- **MDX-powered**: Markdown with React components (live code, diagrams)
- **Auto-generated API docs**: From OpenAPI spec
- **Full-text search**: Across all documentation sites
- **Versioned**: Documentation tied to SDK versions

## When to use

- Writing or updating documentation for any Hanzo brand
- Adding API reference pages
- Creating tutorials or guides
- Modifying the documentation platform itself
- Adding a new brand/product documentation site

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/docs` |
| Upstream | fumadocs fork |
| Framework | Next.js 14+ with MDX |
| Packages | 27 |
| Apps | 16 |
| Dev | `pnpm dev` |
| Build | `pnpm build` |
| Port | 3000 (dev) |

## Brand Sites

| Brand | URL | App |
|-------|-----|-----|
| Hanzo | `hanzo.ai/docs` | `apps/hanzo/` |
| Lux | `docs.lux.network` | `apps/lux/` |
| Zoo | `docs.zoo.ngo` | `apps/zoo/` |
| Zen | `docs.zenlm.org` | `apps/zen/` |

## Project Structure

```
docs/
├── apps/                    # 16 brand-specific apps
│   ├── hanzo/               # hanzo.ai/docs
│   ├── lux/                 # docs.lux.network
│   ├── zoo/                 # docs.zoo.ngo
│   ├── zen/                 # docs.zenlm.org
│   └── ...                  # Other product-specific doc sites
├── packages/                # 27 shared packages
│   ├── core/                # Core fumadocs engine
│   ├── mdx/                 # MDX processing and plugins
│   ├── ui/                  # Shared UI components
│   ├── search/              # Full-text search
│   ├── openapi/             # OpenAPI → docs generator
│   ├── theme-hanzo/         # Hanzo brand theme
│   ├── theme-lux/           # Lux brand theme
│   ├── theme-zoo/           # Zoo brand theme
│   ├── theme-zen/           # Zen brand theme
│   ├── components/          # Shared MDX components
│   │   ├── CodeBlock.tsx
│   │   ├── ApiEndpoint.tsx
│   │   ├── Callout.tsx
│   │   └── SDKTabs.tsx
│   └── ...                  # More packages
├── content/                 # Shared content
│   ├── api/                 # API reference (auto-generated from OpenAPI)
│   ├── guides/              # Cross-brand guides
│   └── sdks/                # SDK documentation
├── pnpm-workspace.yaml
└── package.json
```

## Development

```bash
git clone https://github.com/hanzoai/docs.git
cd docs
pnpm install

# Dev all brands
pnpm dev

# Dev specific brand
pnpm dev --filter @hanzo/docs-hanzo

# Build for production
pnpm build

# Lint MDX
pnpm lint

# Check broken links
pnpm check-links
```

## Writing Documentation

### MDX Page

```mdx
---
title: "Chat Completions"
description: "Create chat completions with the Hanzo API"
---

import { CodeBlock, ApiEndpoint, Callout, SDKTabs } from "@hanzo/docs-components"

# Chat Completions

<Callout type="info">
  This endpoint is OpenAI-compatible.
</Callout>

<ApiEndpoint method="POST" path="/v1/chat/completions" />

## Request

<SDKTabs>
  <SDKTabs.Tab lang="python" title="Python">
    {`from hanzoai import Hanzo
client = Hanzo()
response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello"}],
)`}
  </SDKTabs.Tab>
  <SDKTabs.Tab lang="typescript" title="TypeScript">
    {`import Hanzo from "hanzoai"
const client = new Hanzo()
const response = await client.chat.completions.create({
    model: "zen-70b",
    messages: [{ role: "user", content: "Hello" }],
})`}
  </SDKTabs.Tab>
</SDKTabs>
```

### Adding a New Brand

```bash
# 1. Create new app
cp -r apps/hanzo apps/new-brand

# 2. Create brand theme package
cp -r packages/theme-hanzo packages/theme-new-brand

# 3. Configure brand colors, logo, navigation
# 4. Add to pnpm-workspace.yaml
# 5. Add to CI/CD
```

## Redirects

Service-specific docs have vanity URLs:
- `orm.hanzo.ai` → `hanzo.ai/docs/services/orm`
- `hanzo.ai/docs/api` → Full API reference
- `hanzo.ai/docs/sdks/python` → Python SDK guide

## Related Skills

- `hanzo/python-sdk.md` - Python SDK (documented here)
- `hanzo/js-sdk.md` - JS SDK (documented here)
- `hanzo/hanzo-brand.md` - Brand guidelines for docs styling
- `hanzo/hanzo-cloud.md` - Cloud dashboard (links to docs)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: documentation, fumadocs, mdx, nextjs, multi-brand
**Prerequisites**: MDX, Next.js basics
