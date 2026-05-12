---
name: onyx-plus-verify
description: OnyxPlus standalone IDV flow -- selfie, ID document, thumbprint, proof-of-life; posts to onyxd
---

# OnyxPlus Verify (IDV Flow)

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-onboarding.md`

## Overview

`verify/` is the standalone OnyxPlus identity verification flow. Selfie + ID document + touchless thumbprint + proof-of-life capture, posting directly to `onyxd` at `onyxplus-api.{env}.satschel.com`. Biometric PII never routes through BD.

- Repo: `github.com/onyx-plus/verify`
- Host: `verify.{env}.satschel.com`
- IAM client: `onyxplus-verify`

## When `verify` is used vs `onboarding`

| Surface | Flow | Tenant |
|---|---|---|
| `verify.{env}.satschel.com` | Standalone IDV -- user lands here directly from an RP that needs OnyxPlus attestations | Cross-tenant; user signs in with their OnyxPlus identity |
| `onboarding.{env}.satschel.com` | BD onboarding step 2 -- user is in the middle of a BD/KYC flow on `id.{env}.satschel.com` and gets redirected for the biometric step | BD-driven; user is already in a BD session |

Both surfaces ultimately submit to `onyxd /v1/onyxplus/enrollments/{id}/{liveness,id-document,passkey,submit}`.

## Capture pipeline

1. **Selfie** -- WebRTC camera capture; client-side face detector ensures one face, eyes open, centred
2. **ID document** -- camera capture of ID front + back; OCR client-side; hash stored on enrollment row
3. **Touchless thumbprint** -- face liveness signal (anti-spoofing); pose-entropy scoring on the daemon side after upload
4. **Proof-of-life** -- short head-motion sequence; uploaded as low-fps video; daemon scores
5. **WebAuthn passkey** -- biometric (Face ID / Touch ID) or platform passkey registered via the MPC mesh; credential id stored on enrollment row

After all five steps complete, `POST /v1/onyxplus/enrollments/{id}/submit` triggers the eager 1:1 resolve (`ResolveOnchainIdentity` via BD → TA → MPC + IdentityFactory) and the per-topic ERC-735 claim signing.

## Auth

OAuth 2.1 + PKCE against Hanzo IAM:
- IAM client ID: `onyxplus-verify`
- Redirect URI: `https://verify.{env}.satschel.com/auth/callback`
- Scope: `openid profile email`

The IDV flow can be entered with or without an existing IAM session -- new users register inline.

## Required env (Vite SPA, manifest-substituted)

| Var | Value (dev) |
|---|---|
| `VITE_ONYX_URL` | `https://onyxplus-api.dev.satschel.com` |
| `VITE_IAM_URL` | `https://iam.dev.satschel.com` |
| `VITE_ORG_ID` | `onyxplus` |

## Source pointers

| Subject | Source |
|---|---|
| Repo root | `~/work/onyxplus/verify/` |
| Dockerfile (Vite + nginx or hanzoai/spa) | `~/work/onyxplus/verify/Dockerfile` |
| Capture components | `~/work/onyxplus/verify/apps/web/src/` |

## Related skills

- `onyx-plus/onyx-plus-onyxd.md` - daemon (where submissions land)
- `onyx-plus/onyx-plus-onboarding.md` - BD onboarding variant
- `onyx-plus/onyx-plus-attestation.md` - what gets signed after submit
- `onyx-plus/onyx-plus-iam.md` - OAuth issuer

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: idv, verify, biometric, webauthn, passkey, liveness
