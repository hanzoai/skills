# Hanzo Brand Guidelines

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-docs.md`, `hanzo/hanzo-ui.md`

## Overview

Brand assets, color palette, typography, and guidelines for the Hanzo ecosystem. Consistent branding across all products: hanzo.ai, chat.hanzo.ai, platform.hanzo.ai, console.hanzo.ai, and all open-source repos.

## Key Assets

| Asset | Location | npm Package |
|-------|----------|-------------|
| Logo (all formats) | `github.com/hanzoai/logo` (`dist/` directory) | `@hanzo/logo` |
| Design tokens | `github.com/hanzoai/ui` (`packages/tokens`) | `@hanzo/tokens` |
| Fonts (Zen / Zen Mono) | `github.com/hanzoai/font` | `@hanzo/font` |
| Brand guidelines | `github.com/hanzoai/brand` | `@hanzo/brand` |
| UI components | `github.com/hanzoai/ui` | `@hanzo/ui` |

## Colors

Hanzo is **pure monochrome**: black, white, and greys. There is no brand colour: no red, no accent hue, no gradient text. Colour appears only inside Hanzo's own media (art, video) and semantic status tokens (`--state-error`, `--state-warning`, `--state-success`, `--state-online`, `--destructive`), never as brand UI chrome. `@hanzo/brand` states `"pure monochrome (black + white), no color accent"`; all values come from `@hanzo/tokens`.

Every Hanzo surface ships **light and dark**, following the device (`prefers-color-scheme`).

| Role | Dark | Light | Token |
|------|------|-------|-------|
| Page | `#0a0a0a` (`#000000` on `--pure-black` pages) | `#f7f7f7` | `--background` / `--pure-black` |
| Card / surface | `#0f0f0f` | `#f2f2f2` | `--card` |
| Text | `#e5e5e5` (`#fafafa` on `--text-primary`) | `#0a0a0a` | `--foreground` / `--text-primary` |
| Muted text | `#a3a3a3` | `#525252` | `--muted-foreground` |
| Border | `rgba(255,255,255,.10)` | `rgba(0,0,0,.10)` | `--border` |
| Primary button | `#fafafa` fill, `#0a0a0a` text | `#0a0a0a` fill, `#fafafa` text | `--primary` / `--primary-foreground` |

**Retired, do not reintroduce:** "Hanzo Red" `#d81c33`, and the old Hanzo Chat accent `#fd4444`.

### CSS

Use the tokens rather than hand-written colour variables:

```css
@import '@hanzo/tokens/css'; /* dark on :root; the .light class switches the palette */
```

To follow the device, a tiny script toggles the `light` class on `<html>` from `matchMedia('(prefers-color-scheme: light)')` before first paint.

### Tailwind

```typescript
// Tailwind 3
import { hanzoTheme } from "@hanzo/tokens/tailwind"
export default { theme: { extend: hanzoTheme } }
```

On Tailwind 4, map the semantic tokens in `@theme inline` (`--color-background: var(--background)`, `--color-foreground: var(--foreground)`, and so on).

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
3. **Colors**: Replace the fork's accent with Hanzo's monochrome tokens (`@hanzo/tokens`). No red, no brand hue; preserve semantic status tokens only.
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
| Adding a brand colour (e.g. `#d81c33`, `#fd4444`, neon) | Off-brand | Monochrome `@hanzo/tokens` only; colour only inside media and semantic status tokens |

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
