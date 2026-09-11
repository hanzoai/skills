# Worked example: exchange1 -> exchange2

This example captures a common failure mode: the shell is close, but the new system still reads as a different product because screen topology, typography, control geometry and information hierarchy diverge.

Canonical comparison shown in the example: 1440x900, light theme, reference on the left and target on the right.

## Global findings

### Already close enough to stop touching first

- Main shell horizontal anchor is aligned closely enough to make page-level comparison meaningful.
- The target intentionally retains its own house font, a real dark theme, semantic landmarks, achromatic asset marks, selected elevation treatment and pinned disclosure text. Treat these as `EXEMPT` unless the project changes direction.

### Still blocking visual parity

- Reference body type is visibly larger/heavier; target type is smaller/lighter and uses a fractional ramp.
- Spacing/control construction differs; target controls are materially taller and have a different inner rhythm.
- Most important: multiple routes have different primary information architecture. Fixing colors or radii will not solve them.

## Swap

### Reference topology

`header -> trade tabs -> sell card -> circular swap control -> buy card -> primary CTA`

No standalone page H1 between tabs and form. The form is the visual subject.

### Target currently

`header -> trade tabs -> H1 "Swap" -> sell card -> swap control -> buy card -> helper link -> CTA -> disclosure links -> footer`

The target also shows different fixture values, an inline "predicted" badge and an always-visible decimals/helper treatment.

### Required convergence

1. Remove the redundant `Swap` H1 from the baseline parity state. The active tab is sufficient context.
2. Normalize fixture state so both sides show the same sell amount, buy amount and token selections before geometry is judged.
3. Align the form card top, width and inter-card overlap to the reference after typography/control primitives are fixed.
4. Keep target-only prediction/explanation behavior, but hide it until it is relevant or place it in a compact details affordance that does not expand the baseline form.
5. Make the primary action occupy the same full form width and vertical position as the reference.
6. Keep disclosure/compliance copy, but prevent it from collapsing the application workspace: place it below the primary viewport flow or in the approved disclosure region without shifting the trade form.
7. Recheck header extras (`network`, `sign in`) so they do not change the search/nav alignment.

### Acceptance

- same visible sequence from tabs through CTA
- no extra H1/helper row before CTA
- form width/top within tolerance
- same fixture values/state
- disclosure does not enter the reference's empty workspace above the fold

## Explore

### Reference topology

`header -> 5-column metric strip -> tabs + right toolbar -> quiet content/empty state`

The failure state is sparse. It does not become a dashboard of failed cards.

### Target currently

`header -> H1 + explanatory paragraph -> 4 metrics -> two failed chart cards -> tabs -> full-width failure card -> endpoint diagnostic -> footer`

Tabs are also different in ordering/content and the target defaults to a different primary section.

### Required convergence

1. Remove the Explore hero/H1/explainer from the primary reference state. Put explanation behind help/info if it must remain.
2. Rebuild the top area as the same flat horizontal metric strip. Match number of visible slots, widths, labels/values hierarchy and separators for the mapped data.
3. Match reference tab order and default active tab when the corresponding capabilities exist. Do not silently substitute a different default.
4. Restore the right-side toolbar footprint next to tabs. Map available target filters into that geometry.
5. Collapse failed chart shells in the no-data/error state. Do not show two large empty/failure cards when the reference shows quiet whitespace.
6. Replace the full-width failure card with the reference's restrained empty/error line. Put raw endpoint/debug information behind `Details` or a developer-only disclosure.
7. Keep footer/disclosures below the reference workspace boundary.

### Acceptance

- metric strip is the first content region after header
- no hero text or failed chart cards in the baseline error state
- tabs/order/default match mapped reference behavior
- empty/error message is quiet and occupies comparable footprint
- right toolbar exists at the same anchor

## Positions

### Reference topology

`header -> "Your positions" heading -> disconnected-wallet prompt card -> three compact informational cards`

Before connection, no address explorer dominates the page.

### Target currently

`header -> "Positions" heading -> holder address input -> name-holder action -> empty-holder card -> pair-address card -> footer`

The target's address-based explorer is valid functionality but is primary when the reference makes wallet connection primary.

### Required convergence

1. Make the disconnected-wallet state the default parity state.
2. Match the reference heading wording/scale/position if product copy permits.
3. Recreate the compact wallet prompt card footprint, icon area, explanatory line and two actions (`New position`, `Connect wallet` or mapped equivalents).
4. Restore the three compact informational/feature cards beneath it, using target-safe copy/assets if exact reference content is unavailable.
5. Preserve holder/pair address lookup, but move it behind a secondary `View by address` action, expandable panel, or subsequent route so it does not replace the disconnected state.
6. Do not show large empty lookup cards until the user activates lookup mode.

### Acceptance

- disconnected first viewport has the same region sequence and approximate occupied area
- address tools are not primary before user requests them
- no footer rises into the reference's open workspace

## Add / new position

### Reference topology

`header -> breadcrumb -> H1 "New position" + top-right utility controls -> two-column wizard: stepper sidebar | main selection card`

Main card contains pair selection, fee-tier section and a full-width Continue action.

### Target currently

`header -> centered H1 "Add to a range" -> two inline asset pickers -> wallet message -> explanatory links -> footer`

This is not the same screen shape even though both expose two asset pickers.

### Required convergence

1. Convert the page to the reference wizard composition rather than restyling the centered mini-form.
2. Add breadcrumb and left-aligned `New position` hierarchy.
3. Add the left stepper/sidebar with Step 1 active and Step 2 inactive.
4. Move asset pickers into the main right card with the same two-field horizontal composition.
5. Add/match the fee-tier region when supported. If target cannot supply a valid fee-tier choice, mark the region `EXEMPT` or show a truthful disabled/not-applicable state without inventing functionality.
6. Match Reset/version/settings utility control placement when target has equivalents.
7. Move `Add to a pair` / `Add under a hook` explanations out of the primary wizard footprint; surface them as secondary choices after or beside the mapped workflow.
8. Match Continue button position and disabled state.

### Acceptance

- breadcrumb + H1 + utility row match reference hierarchy
- visible two-column wizard structure exists
- stepper and main card anchors match
- asset selection lives inside main card, not as a centered standalone row
- target-only explanatory flows do not replace the wizard

## Limit

### Reference topology

`header -> trade tabs -> limit-price card with quick offsets -> sell card -> swap control -> buy card -> CTA -> compact warning`

No standalone H1. Quick price actions live inside the first card.

### Target currently

`header -> trade tabs -> H1 "Limit" -> sell card -> buy card -> separate Price field -> transfer-duration 2x2 cards -> helper link -> blocked CTA`

### Required convergence

1. Remove redundant H1 from the baseline form.
2. Reorder the primary controls to match the reference: limit price first, then sell, swap control, buy, CTA.
3. If the target cannot compute reference Market/+1/+5/+10 presets because the chain lacks the required AMM spot price, do not fake them. Preserve a visually compatible price region and mark the unavailable preset behavior as an explicit functional exemption.
4. Move transfer-duration controls out of the middle of the core trade stack. If required, place them in a compact advanced section below the reference-equivalent stack or reveal them after core order inputs are valid.
5. Keep target explanations progressive rather than permanently expanding the form.
6. Match the compact warning region at the bottom if a mapped warning is needed.

### Acceptance

- same primary input order
- duration UI no longer splits the reference trade stack
- unavailable preset behavior is documented, not simulated
- form footprint closely matches reference before optional advanced controls open

## Send

### Reference topology

`dimmed trade screen -> centered Send crypto modal`

Modal contains header + close button, large sending value, token selector, recipient field and primary CTA.

### Target currently

A standalone Send page with H1, amount card, recipient field, parsed transaction facts, helper links, CTA and footer.

### Required convergence

1. This is a `MODE` mismatch: implement Send as a modal/drawer state over the trade screen for parity.
2. Direct `/send` navigation should deep-link into the same modal-open state rather than render a separate page, if routing architecture permits.
3. Match modal width, vertical anchor, overlay opacity, header/close layout and stacked field order.
4. Put the target's recipient validation/transaction facts inside the modal only after input exists, or behind a compact details disclosure. They must not enlarge the untouched baseline modal.
5. Keep the same target validation/business logic; only change its presentation mode and hierarchy.

### Acceptance

- background trade screen remains visible and dimmed
- Send is not a standalone baseline page
- modal region sequence matches reference
- extra diagnostics are conditional

## Buy

### Reference topology

`header -> trade tabs -> buy card with large $ value + preset chips -> provider/secondary row -> primary CTA`

### Target currently

`header -> trade tabs -> H1 "Buy" -> compliance/safety explanation card -> links -> footer`

The target intentionally does not offer a fiat purchase flow. This is a functional divergence, not a reason to fabricate one.

### Required convergence

1. Keep the target's no-fiat/no-registered-firm behavior. Do not create enabled purchase controls merely to imitate the screenshot.
2. Remove the redundant H1 if the active tab provides sufficient context.
3. Place the safety explanation in a compact panel occupying approximately the reference buy-card footprint rather than a separate page narrative.
4. If useful, render reference-like preset/value affordances only when they are clearly disabled/non-actionable and cannot mislead; otherwise preserve the reference card footprint using truthful explanatory content.
5. Keep disclosure links compact and below the main card rather than allowing the page to become a document.
6. Keep the footer below the application workspace.

### Acceptance

- page reads as the same trade-tab layout and density
- no fake enabled fiat path
- safety/compliance difference is explicit but does not create a different page architecture

## Landing

### Reference topology

`header -> large centered hero composition -> headline -> elevated swap card -> short supporting line`

The reference also has subtle blurred decorative marks and a landing-specific header action.

### Target currently

The same headline and swap concept exist, but they start much higher, the card is flatter, compliance/helper content expands below it and the next page section begins inside the captured viewport.

### Required convergence

1. Match hero vertical rhythm: push the headline and trade card to the same visual center/lower center of the viewport.
2. Match headline width, size, weight and line box after the global typography fix.
3. Give the landing trade card the reference-specific elevation/border treatment and footprint; do not assume the inner app form and landing card must share identical outer treatment.
4. Reintroduce subtle decorative background marks if they are part of the approved brand treatment; keep them non-interactive and low contrast.
5. Match the short supporting line position under the card using truthful target copy.
6. Prevent the next major section from entering the canonical 900px viewport if the reference keeps the hero isolated.
7. Map landing-specific header actions without breaking the global header alignment.

### Acceptance

- headline and card vertical centers closely match reference
- hero remains isolated within first viewport
- card outer elevation/footprint matches the landing reference
- supporting copy occupies the same small role

## Cross-screen rule from this example

The target is not primarily failing because of palette. It is failing because many routes are different *screen archetypes*: full page vs modal, mini-form vs wizard, explorer vs wallet prompt, diagnostic dashboard vs quiet empty state. Fix those archetypes before asking for another pixel-polish pass.
