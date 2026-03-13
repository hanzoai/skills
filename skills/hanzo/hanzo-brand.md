# Hanzo Brand Guidelines

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-docs.md`, `hanzo/hanzo-ui.md`

## Overview

Brand assets, color palette, typography, and guidelines for the Hanzo ecosystem. Consistent branding across all products: hanzo.ai, chat.hanzo.ai, platform.hanzo.ai, console.hanzo.ai, and all open-source repos.

## Key Assets

| Asset | Location | npm Package |
|-------|----------|-------------|
| Logo (all formats) | `github.com/hanzoai/logo` (`dist/` directory) | `@hanzo/logo` |
| Brand guidelines | `github.com/hanzoai/brand` | `@hanzo/brand` |
| UI components | `github.com/hanzoai/ui` | `@hanzo/ui` |

## Colors

| Color | Hex | Usage |
|-------|-----|-------|
| **Hanzo Red** | `#d81c33` | **Primary brand**, CTAs, active states |
| Dark | `#0a0a0a` | Backgrounds |
| Light | `#fafafa` | Text on dark backgrounds |
| Muted | `#737373` | Secondary text |
| Border | `#262626` | Card/container borders |
| Chat accent | `#fd4444` | Chat UI accent (rebrand of OpenAI green `#10a37f`) |

**NOTE**: The primary brand color is `#d81c33`, NOT `#fd4444`. The `#fd4444` red is used specifically in Hanzo Chat (LibreChat rebrand) as an accent color.

### CSS Variables

```css
:root {
  --hanzo-red: #d81c33;
  --hanzo-dark: #0a0a0a;
  --hanzo-light: #fafafa;
  --hanzo-muted: #737373;
  --hanzo-border: #262626;
}
```

### Tailwind Config

```typescript
// tailwind.config.ts
export default {
  theme: {
    extend: {
      colors: {
        hanzo: {
          red: "#d81c33",
          dark: "#0a0a0a",
        },
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
pnpm add @hanzo/brand    # Colors, typography, guidelines
pnpm add @hanzo/logo     # SVG logos and marks
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
| Headings | Inter | 600 (Semibold) |
| Body | Inter | 400 (Regular) |
| Code | JetBrains Mono | 400 |
| Marketing | Inter | 700 (Bold) |

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
3. **Colors**: Update Tailwind config and CSS variables to `#d81c33`
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
| Using `#fd4444` as primary | Wrong red | Use `#d81c33` (primary brand) |

## Product Names

| Product | Name | NOT |
|---------|------|-----|
| AI models | Zen (family) | Not "Hanzo LLM" |
| Chat app | Hanzo Chat | Not "LibreChat" |
| Platform | Hanzo Platform | Not "Dokploy" |
| Search | Hanzo Search | Not "Meilisearch" |
| KMS | Hanzo KMS | Not "Infisical" |
| IAM | Hanzo ID (hanzo.id) | Not "Casdoor" |
| Studio | Hanzo Studio | Not "ComfyUI" |
| Flow | Hanzo Flow | Not "Langflow" |
| Insights | Hanzo Insights | Not "PostHog" |

## Related Skills

- `hanzo/hanzo-ui.md` - React component library (implements brand)
- `hanzo/hanzo-docs.md` - Documentation site (brand applied)
- `hanzo/zenlm.md` - Zen model branding

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: brand, design, colors, logo, guidelines
