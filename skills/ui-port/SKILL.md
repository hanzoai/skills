---
name: ui-port
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

## Two rules that outrank every heuristic below

Both are false positives that let an agent declare victory on a screen a human
rejects at a glance. Neither is caught by a CSS check.

### Never validate only the state that happened to render

A screenshot that looks correct because a backend failed is not a pass. An
Explore screen scored PASS on five metric tiles, four tabs, no chart cards and
one quiet failure line; every one of those was the 404 state. Fixture data
restored the chart panels, their headings, and the rows — and the screen was
then wrong on three gates.

So a data-driven screen is scored once per state, and a state nobody exercised
reports NOT EVALUATED, never inheriting its neighbour's verdict:

```
loading · empty · failure · populated-minimal · populated-realistic
```

Only states the reference actually has must match, but each must be entered
deliberately, by a deterministic fixture, and served identically to both
builds. A populated reference is captured separately — never inferred from the
failure state.

The state goes in the row key, so a screen is not representable as one verdict:

```
explore:error:view          positions:disconnected:view
explore:populated:view      positions:connected:view
explore:populated:page      buy:default:view
send:open:view              buy:unsupported:view
```

`explore:error:view` passing says nothing about `explore:populated:view`. A key
that named only the screen is what let the false green exist.

### Never equate DOM presence with visual presence

`getBoundingClientRect()` is not a definition of visible. A `VisuallyHidden`
heading is clipped to one pixel and its child still reports a box: "Swap" in a
1px parent wraps one letter to a line and returns 21×118. A closed `<details>`
keeps its contents laid out and paints none of them.

Ask the question of the text, not the element. Take the Range client rects over
the node and require at least one that survives every clipping ancestor:

```
visible(node) =
     hasRenderedTextRect(node)
  && intersectsScope(node)
  && intersectsAllClippingAncestors(node)
  && !hiddenByAncestor(node)          // display, visibility, ~0 opacity
  && !insideClosedDetails(node)
  && !accessibilityOnlyPattern(node)
```

And geometric visibility is still not perceptual visibility — black text on a
black ground passes every test above. DOM extraction counts semantic material;
the screenshot is the final authority. Neither alone is sufficient.

### Never let a screen converge by hiding

Invention has an inverse, and a census cannot tell them apart: material removed
from the count because it was suppressed, rather than because the composition
was reproduced. Every reviewer asks it directly —

> Did this become closer by hiding or clipping material, rather than by
> reproducing the reference composition?

Fail the screen when material the reference genuinely shows was pushed behind
`opacity: 0`, an offscreen transform, a zero-height or one-pixel clip, an
overflow the viewport never reaches, a closed disclosure, or a condition that
suppresses it only under the conditions the census runs in. The test is
intent-free and mechanical: if the reference paints it and the target does not,
that is a gap, whatever moved it out of view.

### Never measure against a host that now serves the target

The reference address is a fact that changes. The moment the port went live at
the reference's own domain, every capture compared the target to itself —
identical rows, identical strings, 0px, a table reading ten routes at perfect
parity — and it took thirteen independent reviewers to notice. Two safeguards,
both mechanical:

- the harness refuses two builds whose regions agree on every band's y, x,
  width and height, and names the fix (point the reference at the other build);
- the reference is the other build served from its own source where the
  production host can no longer be trusted, on a hostname that build treats as
  its own — some SPAs turn their browser router on only for hosts they know and
  draw the landing for every path elsewhere.

### The reference's language has three homes, and the exception channel is not one

A reference's surface is its string catalog plus whatever its source hardcodes
— a standing legal disclosure under every trade card, a screen's own three
words — and the second kind is audited exactly as much as the first. Capture it
from the rendered reference as leaf sentences (an ancestor's text run together
is a sentence nobody paints) and give it one home beside the catalog, keyed by
your ids because the runtime needs a key and the reference gave none, held to
the capture by a test. The gate then accepts three things and nothing else:

```
catalog id      the reference's catalog, verbatim
painted         sentences the reference draws outside its catalog, verbatim
override        the owner's own decisions — a handful, each one signed
```

The failure mode is letting the override channel become the second catalog:
agents register the reference's own words as "overrides" because nothing else
would hold them, the channel grows to dozens, and a rewrite hides among the
no-ops. Prune it to decisions and the rest either has a captured source or was
never the reference's.

## Scope is part of every assertion

Three scopes, three different facts about one page, never compared to each other:

| scope | what it answers |
|---|---|
| `viewport` | the reference screenshot's contract — the first frame, before anyone scrolls |
| `region` | one named part, for matching a single panel's anatomy |
| `full-page` | the whole document, for a port of an entire site |

A landing page measured 36 words at `viewport` and 292 at `full-page`; scored
against a reference number taken at the other scope, either reading is nonsense.
Record the scope on every row. Below-fold material must not contaminate a
viewport score, and a viewport score must not be reported as a page total.

Two of these are separate deliverables, not two readings of one:

**View parity** is what a person sees in the specified capture. It is what the
reference screenshot contracts for and what the anti-invention rule governs.

**Page parity** is the route's complete information architecture.

`landing:view` converging while `landing:page` stays far apart is not a
contradiction and not a regression. It means the hero port is done and the rest
of the route is still its own migration. Say which one a claim is about.

## Equal counts are not convergence

Send reached exactly 42 words against the reference's 42 and was still visibly a
different screen: the reference modal is a large sending-value area, a token
selector, a destination block and a bottom CTA; the target was a compact
amount/recipient form. The same words in a different arrangement.

So a count never produces a green by itself. Counts detect *invention* — material
present that the reference does not carry. Anatomy detects *the port*. When the
target has fewer words than the reference, the missing material is a region or a
component that was not built, and the fix is never to write sentences to close
the gap.

## Semantics are not visual material

A census that counts `h1` counts a fact about markup, not a fact about the
screen. Where the reference draws its title in a `div` and the target draws the
same title, in the same place, at the same size, in an `h1`, the target is
visually identical and semantically better. Reporting that as *over on h1* asks
a correct port to delete a correct heading to satisfy a screenshot harness.

Split them, and never let the second penalise the first:

```
VISUAL MATERIAL          SEMANTICS
painted title boxes      h1 / h2 / nav / main / table / list
painted prose            landmark roles
painted controls         label associations
painted cards            heading order
marks and icons
```

A difference that is visible scores. A difference that is only in the tag is
recorded on a **semantic exemption** channel:

```
same visible title, reference div → target h1
  visual material   MATCH
  semantics         SEMANTIC_IMPROVEMENT — zero visual penalty
```

The exemption runs one way. A target that drops to a `div` where the reference
uses a heading is a semantic regression, and that is a finding.

## Marks are material, and their absence is a failure

Token marks, action icons and product illustrations carry much of a dense
interface's density, and a census that folds them into "controls" or ignores
them reports a screen of bare labels as converged. They get their own gate:

```
MARKS                    reference / target
token marks
action icons
status icons
decorative product marks
```

Automatic failures:

- a known token's mark is absent where the reference draws one
- text is substituted for a control the reference draws as an icon alone
- a generic placeholder stands in where the token metadata names a canonical mark
- the target invents a badge to explain state where the reference draws none

One component owns the asset control, and screens do not each assemble their
own:

```
AssetChoice
├── mark            always drawn for a known token
├── symbol
├── balance/value   optional
└── chevron

known token, mark available   → draw the mark
known token, no mark          → the canonical achromatic fallback
unselected                    → a generic placeholder, never a fabricated logo
reference draws a mark        → the target may not omit it
```

## Implementation leakage

A port's characteristic failure is not a missing feature — it is the engineer
explaining the new build on the surface of the product. It reads as
microcopy, badges, technical qualifiers and state labels the reference never
carries.

Flag any visible string matching, among others:

```
projected · derived · decimals · browser · unavailable · endpoint · indexer
request · failed to fetch · what this does · what we disclose
```

None of those words is forbidden. Each is evidence worth checking, and the
check is one question:

> Is this phrase something a person needs in order to operate the reference
> interface, or something the author wanted to explain about how the target
> works?

If the latter it leaves the default state. It does not have to be deleted —
a tooltip, a closed disclosure, a settings surface, the result of an
interaction, or an accessibility label all keep the information without putting
it in a composition the reference draws without it. A required disclosure keeps
its access and loses its inline restatement; a required marking can live in an
`aria-label` rather than a visible badge.

The rule, stated once:

```
THE REFERENCE DOES NOT SAY IT
            ↓
THE DEFAULT SCREEN DOES NOT SAY IT
```

## A delta needs a region

The largest displacement is only useful if it names something worth fixing. A
missing 20px help control in a fixed corner will out-measure every real defect
on a nearly converged form, and a report whose headline is that number sends
the next pass to the wrong place.

Label each band and report two numbers:

```
header · primary · secondary · footer · fixed-affordance · overlay

limit:
  largest overall    y860  40px   fixed-affordance
  largest primary    y312  14px   sell field
```

Fix the fixed-affordance gaps once, globally, precisely because they are cheap
and they distort everything measured after them.

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

Decompose each one. A height is assembled, and only the decomposition says out
of what:

```
outer height          padding-top / bottom
content-box height    border-top / bottom
font-size             min-height
computed line-height  icon box
text Range height     icon baseline offset
                      gap
                      align-items
```

The stop condition is the **same rendered tier**, not the same CSS mechanism.
Where the reference lets content size a control — `line-height: normal`, no
minimum, the face's own metrics deciding — that mechanism is not portable: one
face at 16px lands on 27 and another does not. Copying the mechanism copies the
arithmetic and misses the answer. Name the tier, state the number, and let each
build reach it its own way.

Do not close a control-height gap by moving the global type ramp or setting
`line-height: 1`. If the ramp already agrees with the reference, the remaining
difference is local composition, and a global change to fix a local difference
breaks every screen that was right.

A primitive the classifier finds on one build and not on the other is **NOT
COMPARABLE**, never DIFFERENT. Reference tabs that are plain anchors and target
tabs that carry `role="tab"` are a markup difference; reporting a gap there
reports something nobody measured.

The frequent cause of a build having fewer tiers than the reference is a
control geometry with no single home — the same constant redefined per file,
drifting to two or three values, so every control lands on one of them. Give it
one home before comparing tiers.

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

Compare screenshots with overlay and pixel diff, but do not rely on one global percentage. Report the four gates independently, each once per state:

| gate | question |
|---|---|
| TOPOLOGY | same kind of screen, same major regions in the same order? |
| VISIBLE MATERIAL | same headings, prose, controls, cards, affordances actually painted? |
| GEOMETRY | same place, size, alignment, density, rhythm? |
| STATE MATRIX | does parity hold in every reference-relevant state? |

Beneath them, the primitive gates:

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
- a data-driven screen is scored from whichever state happened to render, or a populated state is inferred from a failure state
- visibility is asserted from an element box rather than surviving text rects, so clipped, collapsed or accessibility-only material is counted as shown
- material is moved into a closed disclosure, a visually-hidden wrapper or below the fold to lower a count, where the reference genuinely shows it
- a number measured at one scope is compared against a number measured at another
- word or element counts are reported as convergence without an anatomy comparison
- material the reference paints is suppressed rather than reproduced: `opacity: 0`, an offscreen transform, a zero-height or one-pixel clip, an overflow the viewport never reaches, or a condition that hides it only under the conditions the census runs in
- a control-height gap is closed by moving the global type ramp or forcing `line-height: 1`
- a primitive present on one build and absent on the other is scored as a difference rather than as not comparable
- a semantic difference with no visual difference (reference `div`, target `h1`) is scored as a visual-material failure
- a token mark the reference draws is absent, or a generic placeholder stands in where the metadata names a canonical mark
- explanatory microcopy, a status badge, a technical qualifier or an implementation-state label appears in the default state with no visible counterpart in the reference
- a largest-delta figure is reported without its region, so a fixed affordance outranks the primary content

## What a converged screen's report looks like

Counts are diagnostics. They catch invention and they catch hiding. They stop
being the headline the moment a screen's archetype converges — after that the
remaining signal is anchor geometry and component anatomy:

```
ADD
topology       PASS
material       PASS
geometry       82%
states         PASS
largest delta  SelectPair width +38px

SEND
topology       PASS
material       PARTIAL
geometry       68%
states         PASS
largest delta  modal anatomy
```

A geometry percentage alone is not a report. Pair it with the largest single
displacement, because a percentage says how much agrees and never says what
does not.

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

`scripts/see.mjs` is the reference implementation of the visibility rule above:
`census(scope)` runs inside `page.evaluate` and returns painted headings, words,
prose nodes, controls and cards at `view`, `page` or a region selector, counting
only text whose Range rects survive every clipping ancestor. `lines(marks)`
returns the y of named anchor rows, for matching one composition's rhythm to
another.


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
