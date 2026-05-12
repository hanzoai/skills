---
name: onyx-plus
description: OnyxPlus -- in-house EVM-native identity attestation issuer. Replaces Simplici. Multi-tenant, topic-atomic selective disclosure, on-chain pre-trade gate.
---

# OnyxPlus -- Biometric Identity Attestation

**Category**: OnyxPlus (Satschel)
**Related Skills**: `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-iam.md`, `onyx-plus/onyx-plus-kms.md`, `onyx-plus/onyx-plus-mpc.md`, `liquidity/liquidity-app.md`

## Overview

OnyxPlus replaces the third-party Simplici KYC SaaS with an in-house, EVM-native attestation issuer. The slide framing "decentralized vs centralized" is wrong -- both issuers are centralized (single regulated entity, by SEC/FINRA design). The actual axis is **3rd-party SaaS vs in-house EVM-native**. OnyxPlus wins for regulated securities because ERC-3643 transfers must read identity in Solidity -- W3C VC cannot run inside `transfer()`.

### Architecture

```
liquidity/id (BD onboarding)        verify/ (standalone IDV)
        │                                    │
        └────────► onyxd ◄───────────────────┘
                     │
                     ├─ secp256k1 ECDSA root key (in KMS, ERC-735 signer)
                     ├─ WebAuthn passkeys (UV) registered via the MPC mesh
                     ├─ MPC default wallet per user (provisioned via BD → TA → MPC, option-a)
                     ├─ OnchainID contract per user (deployed via BD relay)
                     ├─ on-chain IdentityRegistry binding (wallet → OnchainID, country)
                     └─ admin/ UI (org isolation, audit, overrides)
```

## When to use

- Implementing OnyxPlus daemon features (enrollment, attestation, claims)
- Wiring a new tenant SPA to the OnyxPlus backend
- Understanding the 1:1 (orgID, userID) → wallet → OnchainID invariant
- Debugging an attestation that fails verification on chain
- Reasoning about the ERC-734/735 + ERC-3643 stack

## Hard requirements

1. **1:1 invariant per `(orgID, userID)`**: exactly one MPC wallet, exactly one OnchainID
2. **Eager creation**: `ResolveOnchainIdentity` runs in `handleSubmit` BEFORE `issueAttestation`; failure aborts with 502 and a critical admin event
3. **Canonical CREATE2 salt**: `keccak256(orgID || 0x00 || userID)` -- pinned by tests in both `liquidity/bd/abi/identity_factory_test.go::TestSaltFromOrgUserCanonical` and `onyxplus/onyxd/onchainid_test.go::TestCanonicalSaltDeterministic`
4. **Option-a management key**: the user's MPC wallet is the OnchainID `MANAGEMENT_KEY` -- OnyxPlus never holds it
5. **Zero chain-write authority in onyxd**: all chain writes route through BD's `REGISTRAR_ROLE` key

## Current vs OnyxPlus

| Aspect | Simplici (current) | OnyxPlus (new) |
|---|---|---|
| Issuer | Simplici SaaS | onyxd (our daemon, our GKE) |
| Tenancy | Multi-tenant SaaS (Simplici-owned) | Multi-tenant in-house (`<org>-<app>` IAM) |
| Carrier | W3C VC JSON-LD in wallet | ERC-734/735 OnchainID contract |
| Auth | OIDC + OID4VP | OAuth 2.1 + PKCE + WebAuthn passkeys (MPC-backed) |
| Verify path | RP → Simplici OID4VP | `ecrecover` offline OR on-chain `IdentityRegistry` read |
| Selective disclosure | Attribute-level (SD-JWT/BBS) | Topic-level (4 ERC-735 claims) |
| Pre-trade gate | Off-chain only -- cannot run in `transfer()` | On-chain -- ATS reads `IdentityRegistry` synchronously in EVM |
| Revocation | Status list (advisory pull) | Atomic bit flip in registry → next `transfer()` fails |
| PII custody | Simplici holds | onyxd Base, encrypted per-org, in our GCP |
| User key custody | Simplici wallet | MPC + WebAuthn passkeys -- Onyx+ never holds wallet authority |

## Selective disclosure -- topic-atomic ERC-735

Onyx+ issues four separately signed claims to the user's OnchainID:

| Topic | Name | Data |
|---|---|---|
| 10 | IDVerified | `(uint64 verifiedAt, bytes32 docDigest, bytes32 selfieDigest)` |
| 11 | Liveness | `(uint64 enrolledAt, uint8 score)` |
| 12 | BiometricUnique | `(uint64 verifiedAt, bytes32 faceHash)` |
| 13 | Jurisdiction | `bytes2 ISO 3166-1 alpha-2` (FHE-encrypted in future PR) |

A relying party that needs only jurisdiction reads the topic-13 claim. It learns nothing about liveness scores or biometric hashes. Same privacy property as W3C VC selective disclosure, different cryptographic primitive.

BD claims (KYC/AML/Accredited, topics 1-3) are issued separately by BD's `REGISTRAR_ROLE` key after onboarding approval.

## Layout

| Dir | Repo | Purpose |
|---|---|---|
| `admin/` | `onyx-plus/admin` | Vite SPA operator dashboard + universe k8s manifests |
| `onyxd/` | `onyx-plus/onyxd` | Go daemon (Base + enrollment + attestation) |
| `internal/` | `onyx-plus/internal` | Next.js + @hanzo/docs engineering docs site (satschel.com-gated) |
| `verify/` | `onyx-plus/verify` | Standalone IDV flow |
| `onboarding/` | `onyx-plus/onboarding` | BD/KYC onboarding monorepo |
| `papers/` | `onyx-plus/papers` | LaTeX papers (onyx-plus, iam, kms, mpc, old-vs-new) |
| `proofs/` | `onyx-plus/proofs` | Formal proofs (future) |
| `brand/` | `onyx-plus/brand` | Tailwind theming |
| `logo/` | `onyx-plus/logo` | Logo SVGs |

## Sibling integrations

- `~/work/liquidity/id` -- BD onboarding frontend; calls `onyxd` directly during step 2
- `~/work/liquidity/bd/identity_ensure.go` -- `/v1/identity/ensure` + `/v1/identity/register` HTTP relay (the BD → TA → MPC + IdentityFactory bridge)
- `~/work/liquidity/bd/abi/identity_factory.go::SaltFromOrgUser` -- canonical salt formula
- `~/work/liquidity/contracts/lib/onchain-id/contracts/factory/IdFactory.sol` -- upstream CREATE2 deployer
- `~/work/liquidity/contracts/lib/erc-3643/contracts/registry/implementation/IdentityRegistry.sol` -- on-chain pre-trade gate
- `~/work/liquidity/ats/identity_check.go` -- ATS reads the chain indexer's `IdentityValid(orgID, userID)` projection

## Open work

1. **Biometric-skip tier policy** -- emit topics 10+13 only when liveness is absent; tag attestation tier; ATS / JurisdictionModule applies tighter limits per tier
2. **Schema migration tooling** -- Hanzo Base auto-migration semantics need per-env verification before mainnet rollout
3. **MPC-resident issuer key** -- move `ONYX_SIGNING_KEY` from process memory into MPC; sign via `MPC.sign()` instead of holding raw secp256k1 in onyxd
4. **Post-quantum claim verifier** -- ERC-735 `scheme=2` ML-DSA-65 once the PQ verifier precompile is on chain; hybrid sign in transition window
5. **FHE jurisdiction** -- replace plaintext `bytes2` country code with FHE ciphertext via the FHE precompile

## Companion documentation

- `~/work/onyxplus/LLM.md` -- engineering reference (workspace root)
- `~/work/onyxplus/internal/content/docs/*.mdx` -- internal docs site
- `~/work/onyxplus/papers/onyx-plus/onyx-plus.pdf` -- long-form paper

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: identity, attestation, erc-735, erc-3643, biometric, oauth, mpc
