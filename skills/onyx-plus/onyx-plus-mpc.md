---
name: onyx-plus-mpc
description: OnyxPlus MPC consumer -- wallet provision via BD → TA → MPC; JWT-claims invariant; passkey direct path
---

# OnyxPlus MPC Consumer

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `hanzo/hanzo-mpc.md`, `onyx-plus/onyx-plus-iam.md`, `onyx-plus/onyx-plus-attestation.md`

## Overview

OnyxPlus does not run its own MPC. `liquid-mpc-{0,1,2}` is a 3-of-3 quorum (threshold `t=2`) shared across BD, ATS, TA, AML, and onyxd. The mesh holds threshold-signed keys for every user's default EVM wallet -- the same wallet that becomes the `MANAGEMENT_KEY` on each OnchainID contract (option-a, self-sovereign -- onyxd holds zero MPC authority).

Mesh:

| Node | HTTP | ZAP |
|---|---|---|
| mpc-0 | `:8081` | `:9653` |
| mpc-1 | `:8081` | `:9663` |
| mpc-2 | `:8081` | `:9673` |

## Critical invariant -- JWT claims, NOT headers

The MPC server resolves `(orgID, userID)` from JWT claims, not from `X-Org-Id` / `X-User-Id` headers. Upstream `pkg/api/middleware.go::authMiddleware` pins this; a test in `handlers_kms_test.go` asserts a client-supplied `X-Org-ID` header must NOT influence the result.

**Implication**: onyxd cannot provision a user's wallet by calling MPC directly -- its JWT carries the daemon's service identity, not the user's. Wallet provisioning must go through TA, which holds `MPC_SERVICE_KEY` (per CTO directive 2026-04-22) and is the sole authority for wallet creation.

## Wallet provision path

```
1. onyxd POSTs to BD /v1/identity/ensure {org_id, user_id}
 with Authorization: Bearer ${BD_INTERNAL_SERVICE_KEY}.

2. BD calls requestMPCWallet(ctx, userID, orgID) — POST to
 TA /v1/ta/mpc/wallets {for_user_id, label, set_default}.
 TA holds MPC_SERVICE_KEY and is the sole authority for
 wallet creation; BD never touches the MPC credential.

3. TA proxies to MPC with its own MPC_SERVICE_KEY, returns
 the user's default EVM wallet address.

4. BD calls IdentityFactory.identityOf(wallet); if zero,
 calls createIdentity(wallet, canonicalSalt), waits for
 the IdentityCreated event, returns (wallet, onchainID,
 txHash) to onyxd.

5. onyxd persists wallet_address + onchain_id_address +
 identity_factory_tx on the enrollment row, then issues
 the per-topic ERC-735 claims.
```

Single round trip from onyxd's perspective: one POST, four legs of internal routing. Idempotent on `(orgID, userID)`.

## What onyxd calls MPC directly for

| Endpoint | Caller path | Why direct |
|---|---|---|
| `POST /v1/passkeys/register` | onyxd → MPC | Per-credential auth, no `(orgID, userID)` JWT-claim requirement |
| `POST /v1/passkeys/challenge` | onyxd → MPC | Same -- challenge is bound to credentialID, not user |
| `POST /v1/ta/mpc/wallets` | onyxd → BD → TA → MPC | Wallet creation needs `(orgID, userID)` in JWT; only TA has it |

## Threshold profile

- **Curve**: ECDSA secp256k1 via CGGMP21 (`luxfi/threshold`)
- **Quorum**: 3 nodes, threshold `t = 2`
- **EdDSA path**: Ed25519 via FROST (for Solana / TON; not the OnyxPlus hot path)
- **Identity**: Each node has an Ed25519 identity key (age-encrypted)
- **Storage**: BadgerDB, AES-256 encrypted key shares
- **HSM hookable**: file, AWS KMS, GCP KMS, Azure HSM, Zymbit, or Hanzo KMS

## ConsensusKV discovery

MPC nodes find each other through ConsensusKV -- a private BFT chain running alongside the mesh. Two certificates per block:
- **Ed25519** for fast finality
- **ML-DSA-65** (lattice) for post-quantum durability

Dual-cert means the chain is durable against a future quantum computer that breaks Ed25519.

## Trust anchor

Same IAM JWKS that gates KMS gates the MPC ZAP listener. There is **no separate trust anchor** for service-to-MPC calls. Every JWT follows the same `iss`/`aud` invariants documented in `onyx-plus-iam.md`.

## Smoke test (compose)

```bash
# 1. Spin local stack
docker compose -f ~/work/liquidity/universe/compose.yml up -d

# 2. Get a TA service JWT
TA_CID=$(docker exec liquid-iam sqlite3 -readonly /data/iam/iam.db \
 "SELECT client_id FROM application WHERE name='liquidity-ta';")
TA_SECRET=$(docker exec liquid-iam sqlite3 -readonly /data/iam/iam.db \
 "SELECT client_secret FROM application WHERE name='liquidity-ta';")

JWT=$(curl -sS -X POST \
 -H "Content-Type: application/x-www-form-urlencoded" \
 -d "grant_type=client_credentials&client_id=${TA_CID}&client_secret=${TA_SECRET}" \
 http://localhost:8000/login/oauth/access_token | jq -r .access_token)

# 3. Provision a wallet via TA (forwards to MPC with TA's MPC_SERVICE_KEY)
curl -sS -X POST \
 -H "Authorization: Bearer ${JWT}" \
 -H "Content-Type: application/json" \
 -d '{"for_user_id":"u_test_001","label":"smoke","set_default":true}' \
 http://localhost:8090/v1/ta/mpc/wallets
```

Successful response: `{"walletId":"...","address":"0x...", ...}`. Same `(orgID, userID)` repeated yields the same `address` (default-wallet idempotence).

## Source pointers

| Subject | Source |
|---|---|
| MPC auth middleware (JWT claims) | MPC upstream `pkg/api/middleware.go::authMiddleware` |
| Handler test pinning header-ignore | `pkg/api/handlers_kms_test.go` |
| onyxd → BD ensure relay | `~/work/onyxplus/onyxd/onchainid.go::EnsureOnchainID` |
| BD → TA wallet provision | `~/work/liquidity/bd/onboarding.go::requestMPCWallet` |
| BD `/v1/identity/ensure` endpoint | `~/work/liquidity/bd/identity_ensure.go` |
| onyxd → MPC passkey register | `~/work/onyxplus/onyxd/enrollment.go::handlePasskey` |
| onyxd → MPC passkey challenge | `~/work/onyxplus/onyxd/enrollment.go::mpcPasskeyChallenge` |
| Threshold lib (CGGMP21 + FROST) | `luxfi/threshold` upstream |
| Internal docs page | `~/work/onyxplus/internal/content/docs/mpc.mdx` |
| Long-form paper | `~/work/onyxplus/papers/onyx-plus-mpc/onyx-plus-mpc.pdf` |

## Related skills

- `hanzo/hanzo-mpc.md` - MPC upstream (threshold-signing)
- `onyx-plus/onyx-plus-iam.md` - JWT issuer
- `onyx-plus/onyx-plus-attestation.md` - ERC-735 claim signer (post-soak: moves into MPC)
- `liquidity/liquidity-app.md` - sibling tenant (BD/TA hold MPC_SERVICE_KEY)

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: mpc, threshold-signing, wallet, jwt-claims, custody, cggmp21
