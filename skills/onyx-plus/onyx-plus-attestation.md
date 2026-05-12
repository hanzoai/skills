---
name: onyx-plus-attestation
description: ERC-735 per-topic claim signer in onyxd -- secp256k1 + EIP-191; 4 topics; canonical OnchainID binding
---

# OnyxPlus ERC-735 Attestation

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-mpc.md`

## Overview

`onyxd/attestation.go` issues four separately-signed ERC-735 claims per enrollment. Each claim binds to the user's OnchainID contract address; BD calls `IIdentity.addClaim(...)` to write them on-chain.

Signature scheme: ECDSA secp256k1 (ERC-735 `scheme=1`) over EIP-191 wrapper, `ecrecover`-compatible.

## Topics

| Topic | Name | ABI shape | Data |
|---|---|---|---|
| 10 | IDVerified | `abi.encode(uint64 verifiedAt, bytes32 documentDigest, bytes32 selfieDigest)` | 96 bytes |
| 11 | Liveness | `abi.encode(uint64 enrolledAt, uint8 score)` | 64 bytes |
| 12 | BiometricUnique | `abi.encode(uint64 verifiedAt, bytes32 faceHash)` | 64 bytes |
| 13 | Jurisdiction | `bytes2 iso3166_1_alpha_2` (left-aligned in 32-byte slot) | 32 bytes |

Topic IDs are pinned by gateway tests on the BD/TA side. ERC-735 schemes:
- `scheme=1` ECDSA secp256k1 (current)
- `scheme=2` ML-DSA-65 (post-quantum, future)

## Signing recipe

```go
// inner = keccak256(abi.encode(onchainID, topic, data))
inner := crypto.Keccak256(addrBuf, topicBuf, data)

// EIP-191 prefix
prefixed := crypto.Keccak256(
    []byte("\x19Ethereum Signed Message:\n32"),
    inner,
)

sig, _ := crypto.Sign(prefixed, signingKey)
sig[64] += 27  // secp256k1 Sign returns v as 0/1; ecrecover wants 27/28
```

The signature is 65 bytes: r (32) || s (32) || v (1, 27 or 28) -- the value `ecrecover` expects in Solidity.

## Required fields on the enrollment row

| Field | Required | Source |
|---|---|---|
| `wallet_address` | yes | `EnsureOnchainID` via BD relay |
| `onchain_id_address` | yes | `EnsureOnchainID` via BD relay |
| `face_hash` | yes (today) | Liveness step (32-byte hex) |
| `id_document_hash` | optional | ID document step (zero-filled if absent) |
| `liveness_score` | yes | Liveness step (float 0..1, scaled to uint8 0..100) |
| `jurisdiction_iso` | yes | Onboarding step (ISO 3166-1 alpha-2) |

`issueAttestation` refuses to sign if `wallet_address` or `onchain_id_address` is missing. The eager-resolve in `handleSubmit` populates both before this runs.

## ClaimsResponse shape

```
GET /v1/onyx/claims/{enrollmentId}?onchain_id=0x...

{
  "enrollment_id": "abc123",
  "user_id": "u_001",
  "org_id": "onyxplus",
  "onchain_id_address": "0x...",
  "issuer_address": "0x...",       // ClaimIssuer contract
  "signer_address": "0x...",       // EOA recovered from sig
  "claims": [
    {
      "topic": 10,
      "scheme": 1,
      "issuer": "0x...",           // ClaimIssuer contract
      "signature": "0x...",        // 65-byte hex
      "data": "0x...",             // ABI-encoded payload
      "uri": "https://onyxplus-api.{env}.satschel.com/v1/onyxplus/enrollments/{id}/verify/{topic}"
    },
    // ... topics 11, 12, 13
  ]
}
```

BD pulls this endpoint after deploying the user's OnchainID, then submits each claim via `IIdentity.addClaim(...)` from its `REGISTRAR_ROLE` key.

## Audit row

Per-topic data digests (not signatures -- those are built on-demand at claim issuance) land on `attestations` collection rows for the legacy `/attestation` endpoint and regulator review. Body fields:

| Field | Notes |
|---|---|
| `version` | `4` (current) |
| `wallet_address` | required |
| `onchain_id_address` | required |
| `face_quality_score` | float |
| `liveness_passed` | bool |
| `dedup_passed` | bool |
| `passkey_credential_id` | from MPC register response |
| `jurisdiction_iso` | uppercase ISO 3166-1 alpha-2 |
| `accredited` | bool |
| `issued_at` / `expires_at` | RFC 3339 (365-day default expiry) |
| `digest` | hex-encoded keccak256 of the body |

## Signing key lifecycle

- Loaded at boot from KMS via `secrets.go::loadFromKMS`
- Held in `signingKey` package-level pointer with `signingOnce` `sync.Once`
- The well-known endpoint exposes the derived Ethereum address so BD can verify `ecrecover(claim) == issuer.signer` offline (no chain lookup)

Future post-soak work: move signing into the MPC mesh so onyxd never holds raw secp256k1. The claim shape is unchanged -- only the `Sign()` implementation moves.

## BD claims vs OnyxPlus claims

| Issuer | Topics | Authority |
|---|---|---|
| OnyxPlus (`onyxd`) | 10, 11, 12, 13 (biometric + jurisdiction) | `ONYX_SIGNING_KEY` |
| BD (compliance) | 1 KYC, 2 AML, 3 Accredited | BD's `REGISTRAR_ROLE` key |

Two separate `ClaimIssuer` contracts; both register their signers on the user's OnchainID. ATS reads all required topics atomically via `IIdentity.getClaimIdsByTopic(...)` during pre-trade gate.

## Biometric-skip path -- OPEN

Today `buildClaims` hard-requires `face_hash` and errors otherwise. For biometric-skipped enrollments the plan is:

1. Emit topics 10 (IDVerified, selfie hash zeroed) + 13 (Jurisdiction)
2. Omit topics 11 (Liveness) + 12 (BiometricUnique)
3. Tag the attestation `tier: "id_only"` so ATS / JurisdictionModule applies tighter limits (per-trade caps, settlement-only, etc.)
4. Still register the digest on-chain via `IdentityRegistry`

**Decision pending** on tier policy (allow stables only / cap securities / block securities).

## Source pointers

| Subject | Source |
|---|---|
| Claim signer | `~/work/onyxplus/onyxd/attestation.go::signERC735Claim` |
| Claims endpoint | `~/work/onyxplus/onyxd/attestation.go::handleGetClaims` |
| Issuer keys well-known | `~/work/onyxplus/onyxd/attestation.go::handleAttestationKeys` |
| TA verifier | `~/work/liquidity/ta/onyx_attestation.go` |
| OnchainID contract | `~/work/liquidity/contracts/lib/onchain-id/contracts/Identity.sol` |
| ClaimIssuer contract | `~/work/liquidity/contracts/src/identity/ClaimIssuer.sol` |

## Related skills

- `onyx-plus/onyx-plus-onyxd.md` - daemon embedding
- `onyx-plus/onyx-plus-mpc.md` - future home of the signing key
- `liquidity/liquidity-app.md` - BD-side compliance claims

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: erc-735, attestation, secp256k1, eip-191, ecrecover, claim, onchainid
