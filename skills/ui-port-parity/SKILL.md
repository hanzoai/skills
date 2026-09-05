---
name: ui-port-parity
description: Port or migrate a UI from an old/reference implementation to a new system while preserving new-system functionality and achieving high visual parity. Use when comparing reference and target URLs, routes, repos/worktrees, Storybook stories, screenshots, PDFs, DOM/computed-style probes, or any combination; when asked to make a rewrite look like an older UI; when a visual regression is "close" but still reads as a different product; or when agents need a repeatable audit-fix-measure loop for screen-by-screen convergence.
---

# UI Port Parity

Treat the reference UI as an executable visual contract, not inspiration. Preserve the target system's working business logic, data model, semantics, accessibility, and explicitly approved divergences while making equivalent visible states converge to the reference.

## Core rule

Never begin by polishing individual components. Converge in dependency order:

1. capture conditions
2. shell geometry
3. typography metrics
4. spacing and control primitives
5. screen topology / information architecture
6. compound components
7. state families and data-dependent regions
8. responsive layouts
9. pixel-level cleanup

A lower-level mismatch invalidates measurements below it. If the shell shifts every child, fix the shell and remeasure. If typography changes wrapping, fix type before vertical spacing. If a screen has the wrong regions or order, do not compensate with margins.

## Supported input modes

Use the highest-confidence inputs available. Combine modes when possible.

- **Live reference + live target:** inspect routes, DOM, computed styles, screenshots, states.
- **Reference repo + target repo/worktree:** run both, inspect implementation and patch target automatically when authorized.
- **Storybook/component explorer:** compare equivalent stories and states before page composition.
- **Screenshots or PDFs:** perform visual topology and geometry analysis; mark computed-style facts as unmeasured.
- **DOM/computed-style probes:** use stable semantic selectors or `data-parity-key` attributes to compare rectangles and styles.
- **Mixed mode:** e.g. reference screenshot + target repo, or reference URL + target Storybook.

If only screenshots are available, do not invent DOM, CSS, or interaction facts. Produce a visual remediation plan and clearly mark what requires a live probe.

## Before editing: establish the parity contract

Create or update a parity manifest with:

- canonical viewport(s), DPR, browser engine, zoom, theme, locale and fonts
- route/state mapping between reference and target
- deterministic fixture/data requirements
- intentional divergences
- shared defects that should not be copied
- unmeasured states

Classify every finding as exactly one of:

- `PARITY`: target differs and should converge.
- `EXEMPT`: intentional difference; do not repeatedly "fix" it.
- `SHARED-BUG`: both systems are wrong; fix separately from parity.
- `UNMEASURED`: insufficient rendered state/data; do not guess.
- `ABSENCE`: reference does not render the element/state; absence alone is not proof target should remove semantics or functionality.

Do not copy accessibility regressions, broken landmarks, unsafe behavior, misleading controls, or shared defects merely because the reference has them.

## Phase 1: deterministic capture

Before comparison, normalize:

- exact viewport and device scale
- browser engine and font loading
- light/dark theme selected explicitly
- locale/timezone/number formatting
- animations/transitions disabled or captured at stable time
- identical scroll positions
- identical route parameters
- identical wallet/account state where possible
- identical market/token/data fixture where possible

Live market or timestamp noise must not dominate screenshot diffs. Add fixture injection or request interception before judging data-dependent components.

If the backend is unavailable, continue on static regions and mark data-dependent regions `UNMEASURED`. Build the fixture injection mechanism now; do not fabricate reference rows or charts.

## Phase 2 gate: system primitives

Do not evaluate page-specific parity until these pass closely enough that descendant measurements are meaningful.

### Shell

Measure page/app root, header, gutters, max-widths, first-content x/y, main content width and major columns. Fix parent geometry before children.

### Typography

Compare rendered metrics, not just CSS names:

- body size
- integer/fractional size ramp
- variable font axes / apparent weight
- line-height strategy
- letter spacing
- x-height/cap-height appearance
- text width and wrapping
- numeric features

Different font families may be an exemption. If so, tune the target face to visual equivalence rather than forcing equal numeric `font-weight` values.

### Spacing

Build a histogram of padding, margin and gap values. Detect contamination such as tiny padding plus forced `min-height`, arbitrary fractional scale multipliers, or one-off compensation values. Fix the generating primitive/token rather than replacing hundreds of local declarations.

### Controls

Compare Button, IconButton, Input, Select, Tabs, Checkbox, Toggle, Badge, Menu, Popover and Modal. Record height, padding, radius, border/elevation, font metrics, icon size, baseline and all visible states.

Do not pass controls because their outer height matches if inner text/icon geometry still differs.

See `references/probe-contract.md` for probe fields and semantic selectors.

## Phase 3 gate: screen topology

This gate is mandatory. A screen that contains different primary regions is not visually close even if tokens match.

For every route/state, create a **screen contract** from top to bottom. Record:

1. persistent shell elements
2. primary content anchor and width
3. ordered visible regions
4. each region's role and approximate bounding box
5. primary vs secondary information
6. overlays/modals vs standalone pages
7. empty/error/loading treatment
8. footer visibility within the canonical viewport

Compare reference and target as region sequences. Example:

`tabs -> sell card -> buy card -> CTA`

is not equivalent to:

`tabs -> H1 -> sell card -> buy card -> helper link -> CTA -> disclosure -> footer`

Fix the topology before pixel styling.

### Screen-topology rules

- Remove redundant page titles if the reference relies on tabs or context instead.
- Do not turn quiet reference states into large cards, dashboards or explanatory blocks.
- Do not preserve target-only primary sections merely because they contain useful information. Move target-only functionality to progressive disclosure, a secondary panel, an advanced section, or a subsequent state when that preserves function without changing the reference geometry.
- Preserve target-only functionality; change its prominence and placement when necessary.
- Do not fake parity with `transform`, negative margins or absolute-position patches that leave the wrong document flow.
- A reference modal must remain modal in the target parity state; do not replace it with a full page unless explicitly exempted.
- A reference wizard/sidebar layout must remain a wizard/sidebar layout; two equivalent fields in a centered mini-form are not structural parity.

See `references/screen-convergence.md` for the required per-screen audit format and `references/exchange-example.md` for a worked example.

## Phase 4: compound components

Once topology matches, compare compound components in place. Use semantic component identities such as:

- market selector
- trade tabs
- swap/limit/send/buy forms
- metric strip
- order-book rows
- tables
- chart toolbar
- wallet prompt
- position card
- asset picker
- wizard stepper

Prefer `data-parity-key="domain.component"` on equivalent nodes. Reuse the same key across reference and target only when the elements are semantically equivalent.

Fix shared primitives at the highest reusable level. Use page-local CSS only for genuinely page-local geometry.

## Phase 5: state families

A component is not ported until its relevant state family converges. Capture at least:

- default
- hover
- active/pressed
- focus-visible
- disabled
- loading
- selected
- empty
- error
- populated

For transaction/product flows also capture disconnected/connected wallet, valid/invalid input, menu open, modal open, and successful/blocked states as applicable.

Target-only safety/compliance behavior is allowed, but it must fit the reference hierarchy and footprint when possible. Never create a fake enabled control to imitate functionality the target intentionally does not provide.

## Phase 6: visual measurement

For matching elements capture at least:

- bounding rect: x, y, width, height
- display, position, overflow, z-index
- flex/grid tracks, alignment and gaps
- padding and margins
- font family, size, weight, line-height, letter spacing
- color and background
- border, radius and shadow
- opacity
- icon box/stroke and text baseline
- wrapping/truncation

Compare screenshots with overlay and pixel diff, but do not rely on one global percentage. Report independent gates:

- shell geometry
- typography
- primitive spacing
- controls
- screen topology
- screen geometry
- state parity
- data regions

Large blank regions must not hide a bad header or form in the global score.

## Phase 7: fix loop

For each screen:

1. capture reference and target at the same state
2. write the screen contract
3. classify all extra/missing/reordered regions
4. fix topology
5. recapture
6. compare major bounding boxes
7. fix compound components
8. recapture
9. inspect overlay/diff hotspots
10. fix typography/spacing/style residuals
11. repeat until the screen gate passes

After any shared primitive change, recapture all previously passing screens to detect regressions.

## Automatic failure conditions

Fail parity if any of these remain without an explicit exemption:

- equivalent screens use different primary region order
- a reference modal is a target page or vice versa
- target adds a prominent H1/hero/card absent from the reference and shifts the primary workflow
- target uses a different active/default tab
- equivalent controls differ materially in height or typography
- an extra footer/disclosure appears inside a viewport where the reference is still application workspace, unless explicitly required and placed without changing primary geometry
- target uses cards where reference is intentionally flat and the difference materially changes hierarchy
- a parent layout error is compensated in descendants
- screenshot diff is accepted without checking the largest residual regions
- an agent declares "close enough" without a measured screen contract

## Output artifacts

Maintain these files in the target repo or parity workspace when possible:

- `VISUAL_PARITY_MANIFEST.md` - environments, route mapping, exemptions, shared bugs
- `VISUAL_PARITY_AUDIT.md` - defect ledger with IDs and statuses
- `VISUAL_PARITY_SCREENS.md` - one contract and convergence checklist per route/state
- deterministic screenshot fixtures and baselines
- a repeatable parity runner such as `bin/parity.mjs`

The defect ledger should include:

`id | screen | region | class | priority | reference | target | root cause | fix location | status | evidence`

Use P0 for shell/topology mismatches that invalidate many descendants; P1 for shared primitives and major component geometry; P2 for local styling; P3 for minor residual polish.

## Definition of done

Do not finish at "same design system." Finish when:

- persistent shell geometry converges
- text has equivalent visual density and wrapping
- spacing rhythm and control geometry converge
- each screen has the same primary topology and hierarchy
- target-only functionality remains available without dominating reference geometry
- representative state families converge
- responsive layouts converge at agreed breakpoints
- data-dependent regions are fixture-tested rather than guessed
- intentional divergences are documented and stable
- automated captures reproduce the result

Use `scripts/check_ledger.py` as a lightweight completion gate when the audit is represented as JSON.

## Bundled resources

Load only what the current parity task needs:

- `references/modes.md` - choose workflow for URLs, repos, Storybook, screenshots/PDFs, DOM dumps or failing backends.
- `references/ledger.md` - defect classes, severity and ledger rules.
- `references/acceptance.md` - phase gates, tolerances and stop conditions.
- `references/probe-contract.md` - stable selectors and computed-style measurement.
- `references/screen-convergence.md` - mandatory per-screen topology contract.
- `references/exchange-example.md` - worked multi-screen example showing page-vs-modal, wizard-vs-mini-form and primary-vs-secondary hierarchy failures.
- `references/report-template.md` - standard report structure.
- `scripts/probe-page.mjs` - optional Playwright DOM/style probe using host-project Playwright.
- `scripts/diff-probes.mjs` - compare two parity probe JSON files.
- `scripts/image-diff.py` - simple equal-size screenshot diff helper.
- `scripts/check_ledger.py` - fail when unresolved blocking parity findings remain.
