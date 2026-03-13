# Hanzo Docs - Documentation Site

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/python-sdk.md`, `hanzo/hanzo-brand.md`

## Overview

Hanzo Docs is the **official documentation site** covering API reference, SDK guides, tutorials, and deployment instructions. Built with Next.js and MDX for rich, interactive documentation.

### Why Hanzo Docs?

- **MDX-powered**: Markdown with React components (live code, diagrams)
- **Auto-generated API docs**: From OpenAPI spec
- **SDK docs**: Python, TypeScript, Go, Rust with runnable examples
- **Search**: Full-text search across all documentation
- **Versioned**: Documentation tied to SDK versions

## When to use

- Writing or updating Hanzo documentation
- Adding API reference pages
- Creating tutorials or guides
- Modifying the documentation site itself

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://hanzo.ai/docs` |
| Framework | Next.js 14+ with MDX |
| Repo | `github.com/hanzoai/docs` |
| Dev | `pnpm dev` |
| Build | `pnpm build` |
| Port | 3000 (dev) |

## Content Structure

```
docs/
├── pages/
│   ├── api/              # API reference (auto-generated from OpenAPI)
│   │   ├── chat.mdx      # Chat completions
│   │   ├── embeddings.mdx # Embeddings
│   │   ├── models.mdx    # Models
│   │   └── files.mdx     # File management
│   ├── guides/           # How-to guides
│   │   ├── getting-started.mdx
│   │   ├── authentication.mdx
│   │   ├── streaming.mdx
│   │   └── function-calling.mdx
│   ├── sdks/             # SDK documentation
│   │   ├── python.mdx
│   │   ├── typescript.mdx
│   │   ├── go.mdx
│   │   └── rust.mdx
│   ├── services/         # Individual service docs
│   │   ├── chat.mdx
│   │   ├── platform.mdx
│   │   ├── kms.mdx
│   │   └── orm.mdx
│   └── tutorials/        # Step-by-step tutorials
│       ├── build-chatbot.mdx
│       ├── deploy-app.mdx
│       └── agent-workflow.mdx
├── components/           # MDX components
│   ├── CodeBlock.tsx
│   ├── ApiEndpoint.tsx
│   ├── Callout.tsx
│   └── SDKTabs.tsx
├── public/              # Static assets
│   └── images/
├── next.config.mjs
└── package.json
```

## Development

```bash
git clone https://github.com/hanzoai/docs.git
cd docs
pnpm install
pnpm dev     # http://localhost:3000

# Build for production
pnpm build
pnpm start

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

import { CodeBlock, ApiEndpoint, Callout } from "@/components"

# Chat Completions

<Callout type="info">
  This endpoint is OpenAI-compatible.
</Callout>

<ApiEndpoint method="POST" path="/v1/chat/completions" />

## Request

<CodeBlock lang="python" title="Python">
{`from hanzoai import Hanzo
client = Hanzo()
response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello"}],
)`}
</CodeBlock>
```

### Redirects

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
**Related**: documentation, mdx, nextjs, api-reference
**Prerequisites**: MDX, Next.js basics
