# Hanzo Brand Guidelines

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-docs.md`, `hanzo/hanzo-ui.md`

## Overview

Brand assets, color palette, typography, and guidelines for the Hanzo ecosystem. Consistent branding across all products: hanzo.ai, chat.hanzo.ai, platform.hanzo.ai, console.hanzo.ai, and all open-source repos.

## Key Assets

| Asset | Location | npm Package |
|-------|----------|-------------|
| Logo (all formats) | `github.com/hanzoai/logo` (`dist/` directory) | `@hanzo/logo` |
| Design tokens | `github.com/hanzoai/tokens` | `@hanzo/tokens` |
| Fonts (Zen / Zen Mono) | `github.com/hanzoai/font` | `@hanzo/font` |
| Brand guidelines | `github.com/hanzoai/brand` | `@hanzo/brand` |
| UI components | `github.com/hanzoai/ui` | `@hanzo/ui` |

## Colors

**Monochrome only — black and white, light and dark following the device.** There is **no Hanzo Red** (and no brand accent color, neon, or gradient glow). All surfaces consume `@hanzo/tokens`.

| Token | Light | Dark | Usage |
|-------|-------|------|-------|
| Background (`surface1`) | `#ffffff` | `#000000` | Primary page canvas |
| Elevated (`surface2`) | `#fafafa` | `#111111` | Cards, panels, modals |
| Foreground (`neutral1`) | `#0a0a0a` | `#fafafa` | Primary text and marks |
| Primary CTA (`accent1`) | `#000000` | `#ffffff` | Primary buttons and active states |
| Muted (`neutral2`) | `rgba(10,10,10,0.6)` | `rgba(250,250,250,0.6)` | Secondary text |
| Border | `rgba(10,10,10,0.12)` | `#262626` | Hairline dividers and card borders |

**NOTE**: Do not use `#d81c33`, `#fd4444`, cyan/purple glows, or custom brand accent colors. Hanzo is strictly monochrome across marketing, apps, chat, IAM, and docs.

### CSS Variables

```css
/* Prefer importing semantic tokens directly from @hanzo/tokens */
:root {
  --background: #ffffff;
  --surface: #fafafa;
  --foreground: #0a0a0a;
  --accent: #000000;
  --muted: rgba(10, 10, 10, 0.6);
  --border: rgba(10, 10, 10, 0.12);
}

.dark {
  --background: #000000;
  --surface: #111111;
  --foreground: #fafafa;
  --accent: #ffffff;
  --muted: rgba(250, 250, 250, 0.6);
  --border: #262626;
}
```

### Tailwind Config

```typescript
// tailwind.config.ts — consume @hanzo/tokens semantic variables
export default {
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        muted: "var(--muted)",
        border: "var(--border)",
      },
    },
  },
}
```

## Logo

Geometric **H** mark (SVG). Available variants:

| Variant | File | Use Case |
|---------|------|----------|
| Full logo | `hanzo-logo.svg` | Marketing, headers |
| H mark | `hanzo-mark.svg` | Favicons, small spaces |
| White | `hanzo-logo-white.svg` | Dark backgrounds |
| Monochrome | `hanzo-logo-mono.svg` | Single-color contexts |

### npm Packages

```bash
# Install brand assets
pnpm add @hanzo/brand # Colors, typography, guidelines
pnpm add @hanzo/logo # SVG logos and marks
```

```typescript
// Usage in code
import { colors, typography } from "@hanzo/brand"
import { HanzoLogo, HanzoMark } from "@hanzo/logo"
```

### Logo Usage Rules

- Minimum clear space: 1x height of H mark on all sides
- Minimum size: 24px height for digital, 10mm for print
- Never stretch, rotate, or apply effects
- Always use provided SVG — never recreate

## Typography

| Context | Font | Weight |
|---------|------|--------|
| Headings | Zen | 600 (Semibold) |
| Body | Zen | 400 (Regular) |
| Code | Zen Mono | 400 |
| Marketing | Zen | 700 (Bold) |

## Brand Policy: No Upstream References

**CRITICAL**: All public-facing docs present Zen models as Hanzo's own family.
- Never reference upstream model names (GLM-5, Kimi K2.5, Qwen3, Moonshot, Zhipu, etc.)
- Brand methodology: **Zen MoDE (Mixture of Diverse Experts)**
- Keep factual specs (params, context, architecture type) accurate
- Private upstream mapping in `github.com/hanzoai/zen` (`gateway/config.yaml`)
- Applies to: hanzo.industries, hanzo.ai, zenlm.org, GitHub READMEs, HuggingFace model cards

## Rebranding Checklist

When rebranding a fork for the Hanzo ecosystem:

1. **Display text only**: Replace user-visible strings, NOT code identifiers
2. **Logo swap**: Direct SVG replacement, don't modify JS class names
3. **Colors**: Map Tailwind config and CSS variables to monochrome `@hanzo/tokens` (black and white, light + dark)
4. **Translations**: Update i18n/localization files
5. **URLs**: Update documentation links and API endpoints
6. **Internal names**: Keep internal package names if they're npm dependencies
7. **Function names**: Keep internal function names (they're code, not brand)
8. **NEVER**: Blanket `sed` on minified JS — breaks class definitions, imports, property assignments

### Common Rebranding Pitfalls

| Mistake | Result | Correct Approach |
|---------|--------|------------------|
| `sed 's/ComfyUI/Hanzo Studio/g'` on minified JS | Broken app | Targeted display text replacement |
| Renaming npm package imports | Build failure | Keep internal package names |
| Changing CSS class names | Broken styles | Only change Tailwind theme values |
| sed on i18n keys | Missing translations | Only change i18n values, not keys |
| Using `#d81c33`, `#fd4444`, or neon accents | Off-brand | Monochrome black & white via `@hanzo/tokens` |

## Product Names

| Product | Name | NOT |
|---------|------|-----|
| AI models | Zen (family) | Not "Hanzo LLM" |
| Chat app | Hanzo Chat | Not "LibreChat" |
| Platform | Hanzo Platform | Not "Dokploy" |
| Search | Hanzo Search | Not "Meilisearch" |
| KMS | Hanzo KMS | Not "Hanzo KMS" |
| IAM | Hanzo ID (hanzo.id) | Not "Hanzo IAM" |
| Studio | Hanzo Studio | Not "ComfyUI" |
| Flow | Hanzo Flow | Not "Langflow" |
| Insights | Hanzo Insights | Not "PostHog" |

## Related Skills

- `hanzo/hanzo-ui.md` - React component library (implements brand)
- `hanzo/hanzo-docs.md` - Documentation site (brand applied)
- `hanzo/zenlm.md` - Zen model branding

---

**Last Updated**: 2026-11-26
**Category**: Hanzo Ecosystem
**Related**: brand, design, colors, logo, guidelines
