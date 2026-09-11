# Screen convergence format

Create one section per route and state. Do this before page-specific code changes.

## [screen / route / state]

### Reference topology

Write the visible primary region sequence from top to bottom. Example:

`header -> trade tabs -> sell card -> swap control -> buy card -> CTA`

### Target topology

Write the same sequence for the target. Include extra regions even if they are useful.

### Structural delta

Classify each difference:

- `MISSING`: reference region not represented
- `EXTRA`: target region absent from reference
- `ORDER`: same regions, wrong order
- `MODE`: modal/drawer/page mismatch
- `STATE`: wrong default/active/empty state
- `FOOTPRINT`: same concept but materially different size/position
- `HIERARCHY`: target makes secondary information primary
- `COPY_ONLY`: content differs but geometry can remain
- `DECORATION`: border/elevation/color/radius only

Any `MISSING`, `EXTRA`, `ORDER`, `MODE`, `STATE`, `FOOTPRINT` or `HIERARCHY` issue that changes the primary workflow is P0/P1 and must be solved before pixel polish.

### Required convergence

Describe changes in implementation order:

1. page/overlay mode
2. region ordering
3. anchors and widths
4. major component dimensions
5. target-only functionality relocation
6. empty/error behavior
7. footer/disclosure placement
8. residual styling

### Functional preservation

For each target-only capability, specify where it moves:

- advanced accordion
- secondary panel
- inline details after user action
- subsequent step
- contextual popover
- separate route linked from the parity screen

Never delete working functionality merely for visual parity unless the user explicitly asks.

### Acceptance criteria

Use measurable conditions. Example:

- same primary region sequence
- same active tab
- primary form top within 2 px of reference at canonical viewport
- primary form width within 2 px
- no extra primary card before CTA
- footer begins below the reference workspace boundary
- target-only details hidden until user requests them

### Evidence

Record screenshot IDs, route, viewport, parity keys and diff output.

## Screen-level stop rule

A screen cannot pass if its topology is still different, even if a pixel score appears favorable because of whitespace.
