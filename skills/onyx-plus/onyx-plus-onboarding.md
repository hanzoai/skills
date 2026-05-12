---
name: onyx-plus-onboarding
description: OnyxPlus onboarding monorepo -- BD step-2 biometric step; forked from ~/work/liquidity/id
---

# OnyxPlus Onboarding

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-verify.md`, `onyx-plus/onyx-plus-onyxd.md`

## Overview

`onboarding/` is the OnyxPlus user-facing identity-enrollment UI. Forked from `~/work/liquidity/id` (BD's onboarding monorepo) and adapted for OnyxPlus's biometric capture step.

- Repo: `github.com/onyx-plus/onboarding`
- Hosts: `onboarding.{env}.satschel.com`

Customers hitting any Liquidity tenant's onboarding flow (BD step 2 identity) land here for the touchless thumbprint capture. The flow submits directly to the OnyxPlus backend at `onyxplus-api.{env}.satschel.com/v1/onyx/enrollments/*` -- biometric PII never routes through BD.

## Stack

- Vite + React 19 + TanStack Router (inherited from liquidity/id)
- `@liquidityio/onboarding` package wiring
- OAuth2/OIDC via Hanzo IAM at `iam.{env}.satschel.com`
- IAM client id: `onyxplus-onboarding`

## Multi-tenant

`config/organizations.ts` carries entries for the tenants this onboarding monorepo serves -- forked from `liquidity/id` which is itself multi-tenant. Hostname-based tenant resolution in `getOrgByHost(host)`:

- `*.liquidity.io` / `*.satschel.com` / `id.liquidity` → liquidity
- `mlc.*` → MLC
- `vcc.*` / `vccross*` → VCC (Venture Cross Capital)
- (and others -- legacy from the fork)

The `lux` tenant entry exists from the upstream fork but is not used by any OnyxPlus deployment; it's data only and does not affect OnyxPlus's stack branding.

## Key env

| Var | Purpose |
|---|---|
| `VITE_BD_URL` | BD base URL |
| `VITE_ONYX_URL` | OnyxPlus backend URL |
| `VITE_IAM_URL` | IAM base URL |
| `VITE_ORG_ID` | defaults to `onyxplus` (this is the OnyxPlus org, not the per-tenant brand) |

## When `onboarding` is used vs `verify`

| Surface | Flow |
|---|---|
| `onboarding.{env}.satschel.com` | BD onboarding step 2 -- user is in the middle of a BD/KYC flow on `id.{env}.satschel.com` and gets redirected for the biometric step |
| `verify.{env}.satschel.com` | Standalone IDV -- user lands here directly from an RP that needs OnyxPlus attestations |

Both ultimately submit to `onyxd /v1/onyx/enrollments/*`.

## Source pointers

| Subject | Source |
|---|---|
| Repo root | `~/work/onyxplus/onboarding/` |
| Tenant config | `~/work/onyxplus/onboarding/config/organizations.ts` |
| App entry | `~/work/onyxplus/onboarding/apps/web/` |
| CLAUDE.md | `~/work/onyxplus/onboarding/CLAUDE.md` |

## Related skills

- `onyx-plus/onyx-plus-verify.md` - standalone IDV variant
- `onyx-plus/onyx-plus-onyxd.md` - daemon (submission target)
- `onyx-plus/onyx-plus-iam.md` - OAuth issuer
- `liquidity/liquidity-app.md` - upstream (BD onboarding flow)

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: onboarding, bd, kyc, biometric, vite
