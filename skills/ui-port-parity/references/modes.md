# Input Modes

## Live reference + target with source

Use browser capture, DOM/style probes, source inspection, and automatic target edits. Audit before the first edit. Preserve target logic.

## Live URLs without source

Measure and capture both. Produce a defect ledger and implementation-level CSS/component guidance. Do not claim to have changed code.

## Repos/worktrees

Start the apps using their existing documented commands. Avoid introducing a new build system. Freeze route, viewport, theme, data, locale, and feature flags before capture.

## Storybook/component pages

Compare primitives first. Build a component state matrix, then move to compound components and full pages. Prefer identical story fixtures.

## Screenshots or PDFs plus target source

Use the image as visual evidence. Measure image-space geometry when helpful, but label it as image-derived rather than DOM-measured. Inspect the target live when possible. Mark hidden/interactive reference states `UNMEASURED`.

## Screenshot-only

Produce:

- visible page-shell differences
- region geometry differences
- apparent type/spacing/radius/color differences
- missing/extra visible regions
- likely systemic causes labeled as hypotheses
- a prioritized remediation ledger

Do not invent CSS values, breakpoints, focus states, hover states, DOM structure, or data behavior.

## DOM/computed-style dumps

Compare exact values first. Group recurring differences by shared ancestor/token/component. Treat values as evidence, but still verify rendered screenshots where possible because different fonts/renderers can produce visual differences from similar computed CSS.

## Partial or failing backend data

Record data-dependent areas as `UNMEASURED`. Do not redesign the target around an error state that may not represent the reference's populated rendering. Build deterministic fixture hooks while waiting for a trustworthy reference payload.
