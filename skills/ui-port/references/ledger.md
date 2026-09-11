# Visual Parity Ledger

Use one row per independently actionable issue.

| Field | Meaning |
|---|---|
| ID | Stable identifier such as VP-001 |
| Class | PARITY, EXEMPT, SHARED-BUG, UNMEASURED, ABSENCE |
| Region | Shell, Header, Typography, Primitive, Page, Data, Responsive, etc. |
| Severity | P0 systemic, P1 major, P2 local, P3 polish |
| State | Route + UI state + breakpoint |
| Reference evidence | Measured or observed reference behavior |
| Target evidence | Measured or observed target behavior |
| Delta | Concrete geometric/style/structural difference |
| Root cause | Highest plausible shared cause |
| Fix location | Token, primitive, component, layout, route, fixture |
| Validation | How completion will be verified |
| Status | open, blocked, fixed, exempted |

## Severity

- **P0**: ancestor/system defect that invalidates downstream measurements, such as container model, base type scale, root spacing mechanism.
- **P1**: major primitive/region mismatch visible across a page or many states.
- **P2**: local component or state mismatch.
- **P3**: final polish with small visual impact.

## Rules

- Fix P0 before trusting descendant geometry.
- Re-measure descendants after ancestor fixes.
- Do not close `UNMEASURED` by assumption.
- Put user-approved differences in the divergence registry and reference the exemption ID.
- Reclassify shared defects out of parity instead of forcing both implementations to remain wrong.
