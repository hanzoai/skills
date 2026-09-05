# Probe contract

Use this when live DOM access is available.

## Stable identities

Prefer semantic keys over brittle CSS selectors:

```html
<header data-parity-key="app.header">
<section data-parity-key="trade.panel">
<button data-parity-key="trade.primary-action">
```

Do not assign the same key to nodes that only look similar but serve different roles.

## Computed-style probe

Capture this shape for each parity key:

```js
const r = el.getBoundingClientRect();
const s = getComputedStyle(el);
({
  rect: { x: r.x, y: r.y, width: r.width, height: r.height },
  display: s.display,
  position: s.position,
  overflow: s.overflow,
  zIndex: s.zIndex,
  padding: s.padding,
  margin: s.margin,
  gap: s.gap,
  gridTemplateColumns: s.gridTemplateColumns,
  gridTemplateRows: s.gridTemplateRows,
  alignItems: s.alignItems,
  justifyContent: s.justifyContent,
  fontFamily: s.fontFamily,
  fontSize: s.fontSize,
  fontWeight: s.fontWeight,
  lineHeight: s.lineHeight,
  letterSpacing: s.letterSpacing,
  fontVariantNumeric: s.fontVariantNumeric,
  color: s.color,
  backgroundColor: s.backgroundColor,
  border: s.border,
  borderRadius: s.borderRadius,
  boxShadow: s.boxShadow,
  opacity: s.opacity,
});
```

Also record text content, line count, scroll size, pseudo-element presence, icon box and selected/disabled/focus state where relevant.

## Dependency rule

Do not interpret descendant x/y differences before common ancestor geometry matches. Mark them `blocked_by:<ancestor-key>` and remeasure after the ancestor is fixed.

## Typography rule

When font families differ intentionally, compare:

- actual rendered text width for identical strings
- line count and wrap points
- baseline position inside controls
- apparent weight at target size
- cap/x-height impression

The same numeric font weight across different variable faces is not automatically equivalent.
