# Acceptance and Stop Conditions

## Canonical state matrix

Choose states appropriate to the app. Typical coverage:

- default page
- alternate primary mode/tab
- hover/pressed/focus on shared controls
- populated and empty forms
- open dropdown/combobox
- loading/error/empty data
- connected/disconnected account where relevant
- populated/empty tables
- modal/popover/drawer open
- desktop/tablet/mobile

## Phase gates

### Gate A - System

- shell geometry passes
- typography rhythm passes or has declared font exemption
- spacing model no longer contains systemic contamination
- control geometry passes

Do not begin page pixel polishing before Gate A.

### Gate B - Primitives

- shared controls match across relevant state families
- no repeated local compensating overrides remain

### Gate C - Pages

- equivalent regions appear in equivalent order
- major rectangles align
- target-only functionality is visually native to the reference system

### Gate D - Regression

- deterministic captures are repeatable
- per-region screenshot/geometry thresholds pass
- unresolved states are explicitly `UNMEASURED`, not silently ignored

## Suggested default tolerances

Use tighter project-specific values when practical.

- static major geometry: <= 1px
- control height/radius/type size: exact when same visual primitive is expected
- stable color tokens: exact
- screenshot diff: <= 0.5-1% unexplained differing pixels per stable region

A global screenshot score cannot override a failed shell or primitive gate.

## Automatic failures

- wrong container model
- wrong default route/tab/state
- materially different section ordering
- systemic type scale mismatch
- systemic spacing/control sizing mismatch
- descendant offsets compensating for ancestor defects
- removed target functionality without instruction
- accessibility regression for visual imitation
- guessed rendering for unavailable reference state
- completion without rendered comparison
