# Hanzo UI - React Component Library

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-billing.md`, `hanzo/hanzo-brand.md`, `hanzo/hanzo-chat.md`

## Overview

Hanzo UI is a **React component library** forked from shadcn/ui, extended with 161+ components (3x upstream), two themes, multi-framework support, AI components, 3D components, and a visual page builder. Published as `@hanzo/ui` on npm. Used across all Hanzo frontend apps.

## When to use

- Building UIs for Hanzo ecosystem applications
- Adding pre-built React components via CLI (`npx @hanzo/ui add button`)
- Creating white-label applications (Hanzo, Lux, Zoo brand configs)
- Using AI, 3D, or animation components
- Visual page building with drag-and-drop block assembly
- E-commerce, checkout, and commerce flows

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
| Related | `@hanzo/auth`, `@hanzo/commerce`, `@hanzo/agent-ui`, `@hanzo/react` |
| CLI | `shadcn` v3.8.4 |
| Components | 161+ (~127 implemented, ~34 stubs) |
| Themes | 2 (Default, New York) |
| Blocks | 24+ production-ready templates |
| Docs | `ui.hanzo.ai` |
| Framework | React 19, Tailwind CSS 4, Radix UI |
| Build | Turborepo + pnpm |
| Test | Vitest + Playwright |
| Dev port | 3003 |

## Package ecosystem

| Package | Purpose |
|---------|---------|
| `@hanzo/ui` | Core component library (161+ components) |
| `@hanzo/auth` | Authentication components and flows |
| `@hanzo/commerce` | E-commerce components (cart, checkout, pricing) |
| `@hanzo/agent-ui` | AI agent interface components |
| `@hanzo/react` | React hooks and utilities |
| `@luxfi/ui` | Lux ecosystem UI preset (extends @hanzo/ui) |

## Install and use

```bash
# Add a component to your project
npx @hanzo/ui add button
npx @hanzo/ui add dialog
npx @hanzo/ui add ai-chat

# Or install the package
pnpm add @hanzo/ui
```

## Component categories

| Category | Count | Examples |
|----------|-------|---------|
| **Primitives** | 40+ | Button, Input, Select, Dialog, Popover |
| **Data Display** | 20+ | Table, DataTable, Card, Badge, Avatar |
| **Navigation** | 10+ | Tabs, Breadcrumb, Sidebar, Command |
| **Feedback** | 10+ | Toast, Alert, Progress, Skeleton |
| **AI** | 8+ | AIChat, AICompletion, AICodeBlock, AIMessage |
| **3D** | 5+ | Spline, Canvas3D, Scene |
| **Commerce** | 15+ | Cart, Checkout, PricingCard, ProductCard |
| **Layout** | 10+ | Page, Section, Grid, Stack |
| **Animation** | 5+ | Motion, AnimatePresence, Transition |

## Tailwind configuration

Projects using @hanzo/ui configure Tailwind to scan the library:

```typescript
// tailwind.config.ts
import { hanzoUIPreset } from '@hanzo/ui/tailwind'

export default {
  presets: [hanzoUIPreset],
  content: [
    'src/**/*.tsx',
    './node_modules/@hanzo/ui/**/*.{ts,tsx}',
    './node_modules/@hanzo/auth/**/*.{ts,tsx}',
    './node_modules/@hanzo/commerce/**/*.{ts,tsx}',
  ]
}
```

## Development

```bash
cd ~/work/hanzo/ui
pnpm install

# Development server (docs site)
pnpm dev          # Port 3003

# Build all packages
pnpm build

# Run tests
pnpm test

# Add component to registry
pnpm shadcn add <component-name>
```

## White-label theming

```typescript
// Brand configuration for different orgs
import { createTheme } from '@hanzo/ui/theme'

const hanzoTheme = createTheme({
  brand: 'hanzo',
  colors: { primary: 'oklch(0.7 0.15 250)' }
})

const luxTheme = createTheme({
  brand: 'lux',
  colors: { primary: 'oklch(0.8 0.12 45)' }
})
```

## Pattern: Static-exported Next.js app with @hanzo/ui

Used by billing.hanzo.ai, lux.id, and other sites:

```tsx
// app/layout.tsx
import '@hanzo/ui/globals.css'

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-black text-white">{children}</body>
    </html>
  )
}
```

## Related Skills

- `hanzo/hanzo-billing.md` -- Billing portal (uses @hanzo/ui)
- `hanzo/hanzo-brand.md` -- Brand guidelines and design tokens
- `hanzo/hanzo-chat.md` -- Chat UI
- `hanzo/hanzo-id.md` -- Login UI

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: ui, react, components, shadcn, tailwind, design-system
**Prerequisites**: React 19, Tailwind CSS 4, pnpm
