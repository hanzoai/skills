# Hanzo Brand Guidelines

**Category**: Hanzo Ecosystem

## Overview

Brand assets and guidelines for the Hanzo ecosystem.

## Key Assets

| Asset | Location |
|-------|----------|
| Logo (all formats) | ``github.com/hanzoai/logo`dist/` |
| Brand guidelines | ``github.com/hanzoai/brand`` |

## Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Hanzo Red | `#fd4444` | Primary brand, CTAs, links |
| Dark | `#0a0a0a` | Backgrounds |
| Light | `#fafafa` | Text on dark |

## Logo

Geometric **H** mark (SVG). Available as:
- `hanzo-logo.svg` — Full logo
- `hanzo-mark.svg` — H mark only
- `hanzo-logo-white.svg` — White variant

## Brand Policy: No Upstream References

**CRITICAL**: All public-facing docs present Zen models as our own family.
- Never reference upstream model names (GLM-5, Kimi K2.5, Qwen3, etc.)
- Brand methodology: **Zen MoDE (Mixture of Diverse Experts)**
- Private upstream mapping in ``github.com/hanzoai/zen`gateway/config.yaml`
- Applies to: hanzo.industries, hanzo.ai, zenlm.org, GitHub, HuggingFace

## Rebranding Checklist

When rebranding a fork:
1. Use targeted display text replacement (NOT blanket sed on minified JS)
2. Replace logos directly (SVG swap)
3. Update colors in Tailwind config and CSS variables
4. Update translations/i18n files
5. Never rename internal package names unless publishing separately

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
