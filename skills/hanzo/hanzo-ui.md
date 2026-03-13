# Hanzo UI - Component Library (shadcn/ui Fork)

**Category**: Hanzo Ecosystem
**Related Skills**: `frontend/react-component-patterns.md`, `frontend/nextjs-app-router.md`, `hanzo/hanzo-desktop.md`

## Overview

Hanzo UI is a **React component library** forked from shadcn/ui, extended with 161+ components (3x more than upstream), two themes, multi-framework support, AI components, 3D components, and a visual page builder. Published as `@hanzo/ui` on npm. The CLI is published as `shadcn` v3.8.4.

Built on **React 19**, **Tailwind CSS 4**, **Radix UI** primitives, and **motion** (Framer Motion successor). Uses **pnpm** workspaces with **Turborepo** for builds.

### OSS Info

Repo: `github.com/hanzoai/ui`. Private workspace name `ui` v0.0.1. Docs: `ui.hanzo.ai`. Dev port: 3003.

## When to use

- Building UIs for Hanzo ecosystem applications
- Adding pre-built React components via CLI (`npx @hanzo/ui add button`)
- Creating white-label applications (Zoo, Lux brand configs)
- Using AI, 3D, or animation components not available in upstream shadcn/ui
- Visual page building with drag-and-drop block assembly
- E-commerce / checkout / commerce flows

## Hard requirements

1. **pnpm 9+** (not npm or yarn -- enforced by workspace config)
2. **Node.js 20+**
3. **React 19** (peer dependency)
4. **Tailwind CSS 4** (OKLCH color system)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/ui` |
| npm package | `@hanzo/ui` |
| CLI | `shadcn` v3.8.4 |
| Components | 161+ (~127 implemented, ~34 stubs) |
| Themes | 2 (Default, New York) |
| Blocks | 24+ production-ready templates |
| Docs | `ui.hanzo.ai` |
| Framework | React 19, Tailwind CSS 4, Radix UI |
| Build | Turborepo + pnpm |
| Test | Vitest + Playwright |
| License | MIT |

## Project structure

```
ui/
├── package.json          # Workspace root (name: "ui", v0.0.1)
├── pnpm-workspace.yaml   # Workspace definition
├── turbo.json            # Turborepo config
├── brand.config.ts       # Brand configuration
├── components.json       # shadcn component config
├── registries.json       # 35+ external registry sources
├── tailwind.config.cjs
├── vitest.config.ts
├── playwright.config.ts
├── app/                  # Documentation site (Next.js 15.3, React 19)
│   └── registry/         # Component registry (SOURCE OF TRUTH)
│       ├── default/ui/   # 150+ components (default theme)
│       ├── default/example/  # Usage demos
│       ├── default/blocks/   # 24+ full-page sections
│       └── new-york/     # Alternative theme
├── apps/                 # Additional apps
├── packages/
│   ├── shadcn/           # CLI tool (published as shadcn v3.8.4)
│   ├── og/               # OpenGraph image generation
│   └── tests/            # Shared test utilities
├── pkg/
│   ├── ui/               # Core component library (@hanzo/ui on npm)
│   ├── agent-ui/         # Agent interface components
│   ├── brand/            # Branding system (white-label support)
│   ├── checkout/         # Checkout flow components
│   ├── commerce/         # E-commerce components
│   ├── react/            # React utilities
│   └── shop/             # Shop/storefront components
├── demo/                 # Demo applications
├── deprecated/           # Deprecated components
├── docs/                 # Documentation source
├── template/             # Project templates
├── templates/            # Additional templates
└── tests/                # E2E and visual tests
```

## Three-layer architecture

1. **Components** (`app/registry/{style}/ui/`) -- Single primitives (Button, Card, Dialog). CLI-installable via `npx @hanzo/ui add`.
2. **Examples** (`app/registry/{style}/example/`) -- Usage demos displayed in docs via `<ComponentPreview />`.
3. **Blocks** (`app/registry/{style}/blocks/`) -- Full-page sections (Dashboard, Login). Docs only, NOT CLI-installable.

## Component categories (from LLM.md)

- 9 3D components
- 12 AI components
- 13 animation components
- 15 navigation variants
- Standard shadcn/ui primitives (Button, Card, Dialog, Form, Table, etc.)
- Commerce components (checkout, cart, product)
- Brand system components

## Package exports

```typescript
import { Button, Card } from '@hanzo/ui'
import { Button } from '@hanzo/ui/components'
import * as Dialog from '@hanzo/ui/primitives/dialog'
import { cn } from '@hanzo/ui/lib/utils'
```

## CLI usage

```bash
# Install components into your project
npx @hanzo/ui add button
npx @hanzo/ui add card dialog

# Install from external registries (35+ sources in registries.json)
npx @hanzo/ui add @aceternity/spotlight
```

## Key dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| react | 19.2.0 | UI framework |
| tailwindcss | 4.1.16 | Styling (OKLCH colors) |
| motion | 12.12.1 | Animations (Framer Motion successor) |
| lucide-react | 0.544.0 | Icons |
| recharts | 3.3.0 | Charts |
| @tanstack/react-table | 8.21.3 | Data tables |
| @dnd-kit/core | 6.3.1 | Drag and drop (page builder) |
| turbo | 2.6.0 | Build orchestration |
| vitest | 4.0.6 | Unit testing |
| @playwright/test | 1.56.1 | E2E testing |

## Development

```bash
git clone https://github.com/hanzoai/ui.git
cd ui
pnpm install
pnpm dev           # Dev server on :3003
```

### Build (critical order)

```bash
pnpm build:registry    # MUST run first -- generates JSON for CLI
pnpm build             # Then build the app
```

### Commands

```bash
pnpm dev               # Dev server
pnpm build             # Build all
pnpm lint              # Lint all workspaces
pnpm typecheck         # Type checking
pnpm test              # Unit tests (Vitest)
pnpm test:e2e          # E2E tests (Playwright)
pnpm changeset         # Create changeset for publishing
pnpm health-check      # Component health check
```

## White-label / branding

The repo supports white-label forks via brand configs in `brand.config.ts`. Zoo and Lux brands are configured. Create a `{BRAND}.brand.ts` file to customize colors, logos, and typography.

## Adding a new component

1. Create in BOTH themes: `app/registry/{default,new-york}/ui/my-component.tsx`
2. Create example: `app/registry/default/example/my-component-demo.tsx`
3. Create docs: `app/content/docs/components/my-component.mdx`
4. Update nav: `app/config/docs.ts`
5. Build: `pnpm build:registry`

## Import path transformation

Registry files use `@/registry/default/ui/button`. After CLI install into a user project, paths are rewritten to `@/components/ui/button`.

## Known gotchas (from LLM.md)

- Registry index is `Index[style][name]`, NOT `Index[name]` -- caused silent block render failures
- Shiki `getHighlighter` incompatible with static export -- replaced with basic pre/code
- Some blocks have Server Component issues with event handlers
- Firebase auth split to optional `@hanzo/auth-firebase` package
- `@hanzo/auth` v2.6.0 uses pluggable provider registry

## Related Skills

- `hanzo/hanzo-desktop.md` - Desktop app (has its own internal hanzo-ui lib, separate from this)
- `frontend/react-component-patterns.md` - React fundamentals
- `frontend/nextjs-app-router.md` - Next.js integration
- `hanzo/hanzo-node.md` - Backend infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: frontend, react, components, tailwind, shadcn, design-system
**Prerequisites**: React 19, TypeScript, Tailwind CSS 4, pnpm
