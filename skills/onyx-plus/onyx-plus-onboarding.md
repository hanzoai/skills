---
name: onyx-plus-onboarding
description:  onboarding monorepo -- BD step-2 biometric step; forked from ~/work/liquidity/id
---

#  Onboarding

**Category**: 
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-verify.md`, `onyx-plus/onyx-plus-onyxd.md`

## Overview

`onboarding/` is the  user-facing identity-enrollment UI. Forked from `~/work/liquidity/id` (BD's onboarding monorepo) and adapted for 's biometric capture step.

- Repo: `github.com/onyx-plus/onboarding`
- Hosts: `onboarding.{env}.`

Customers hitting any Liquidity tenant's onboarding flow (BD step 2 identity) land here for the touchless thumbprint capture. The flow submits directly to the  backend at `-api.{env}./v1/onyx/enrollments/*` -- biometric PII never routes through BD.

## Stack

- Vite + React 19 + TanStack Router (inherited from liquidity/id)
- `@/onboarding` package wiring
- OAuth2/OIDC via Hanzo IAM at `iam.{env}.`
- IAM client id: `-onboarding`

## Multi-tenant

`config/organizations.ts` carries entries for the tenants this onboarding monorepo serves -- forked from `liquidity/id` which is itself multi-tenant. Hostname-based tenant resolution in `getOrgByHost(host)`:

- `*.` / `*.` / `id.liquidity` → liquidity
- `mlc.*` → MLC
- `vcc.*` / `vccross*` → VCC (Venture Cross Capital)
- (and others -- legacy from the fork)

The `lux` tenant entry exists from the upstream fork but is not used by any  deployment; it's data only and does not affect 's stack branding.

## Key env

| Var | Purpose |
|---|---|
| `VITE_BD_URL` | BD base URL |
| `VITE_ONYX_URL` |  backend URL |
| `VITE_IAM_URL` | IAM base URL |
| `VITE_ORG_ID` | defaults to `` (this is the  org, not the per-tenant brand) |

## When `onboarding` is used vs `verify`

| Surface | Flow |
|---|---|
| `onboarding.{env}.` | BD onboarding step 2 -- user is in the middle of a BD/KYC flow on `id.{env}.` and gets redirected for the biometric step |
| `verify.{env}.` | Standalone IDV -- user lands here directly from an RP that needs  attestations |

Both ultimately submit to `onyxd /v1/onyx/enrollments/*`.

## Source pointers

| Subject | Source |
|---|---|
| Repo root | `~/work//onboarding/` |
| Tenant config | `~/work//onboarding/config/organizations.ts` |
| App entry | `~/work//onboarding/apps/web/` |
| CLAUDE.md | `~/work//onboarding/CLAUDE.md` |

## Related skills

- `onyx-plus/onyx-plus-verify.md` - standalone IDV variant
- `onyx-plus/onyx-plus-onyxd.md` - daemon (submission target)
- `onyx-plus/onyx-plus-iam.md` - OAuth issuer
- `liquidity/liquidity-app.md` - upstream (BD onboarding flow)

---

**Last Updated**: 2026-05-12
**Category**: 
**Related**: onboarding, bd, kyc, biometric, vite
