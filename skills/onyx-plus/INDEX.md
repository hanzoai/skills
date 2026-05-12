---
name: onyx-plus-index
description: OnyxPlus -- biometric identity attestation product (Satschel). Repos, components, decision tree.
---

# OnyxPlus Skills Index

**Biometric identity attestation by Satschel. EVM-native, ERC-734/735 OnchainIDs, 1:1 (orgID,userID) ↔ wallet ↔ OnchainID invariant, on-chain pre-trade gate.**

OnyxPlus is paired with Liquidity as the in-house replacement for the third-party Simplici KYC SaaS. Multi-tenant by `<org>-<app>` IAM, white-label by hostname. Privacy is topic-atomic via ERC-735 (four separately signed claims), not attribute-atomic via SD-JWT/BBS+.

## Critical Rules

- **OnyxPlus is a SATSCHEL/LIQUIDITY surface** -- never mention Lux/luxd/lux-prefixed compounds in OnyxPlus docs or code (image refs to upstream OSS `luxfi/*` modules are fine; brand-prefixed identifiers are not)
- **Repo org**: `github.com/onyx-plus` (private, SSH host `github-zatsch`)
- **GCP**: `onyxplus-{devnet,testnet,mainnet}` projects; clusters `dev`, `test`, `main` in `us-central1`
- **Registry**: `us-docker.pkg.dev/onyxplus-registry/onyx-plus/{name}:{semver}` -- separate from `liquidity-registry` and `ghcr.io/hanzoai`
- **Auth**: shared Liquidity IAM (`liquid-iam`, Casdoor fork) -- the `onyxplus` org is a peer tenant
- **Secrets**: shared Liquidity KMS (`liquid-kms`, ZapDB) under path `secret/data/onyxplus/{env}/*`
- **MPC**: shared `liquid-mpc-{0,1,2}` mesh; onyxd holds zero MPC authority -- BD → TA → MPC is the wallet provision path
- **Daemon convention**: `<x>d` (`onyxd`, `hanzod`); no `luxd` in OnyxPlus context

## Network facts

| Item | Value |
|------|-------|
| Backend | `onyxd` (Go + Hanzo Base) on `onyxplus-api.{env}.satschel.com` |
| Operator UI | `admin` (Vite SPA) on `onyxplus.{env}.satschel.com` |
| Internal docs | `internal` (Next.js + @hanzo/docs) on `internal.onyxplus.{env}.satschel.com` |
| IDV flow | `verify` (Vite SPA) on `verify.{env}.satschel.com` |
| BD onboarding | `onboarding` (Vite SPA) on `onboarding.{env}.satschel.com` |
| Build | Cloud Build (`cloudbuild.yaml`) → `us-docker.pkg.dev/onyxplus-registry/onyx-plus/*` |
| Deploy | `kubectl set image` against `dev`/`test`/`main` GKE clusters in strict order |

## Skills in this category

| Skill | Topic |
|---|---|
| `onyx-plus/onyx-plus.md` | Umbrella overview, identity model, 1:1 invariant |
| `onyx-plus/onyx-plus-onyxd.md` | Go daemon (Hanzo Base + enrollment + attestation) |
| `onyx-plus/onyx-plus-admin.md` | Operator dashboard (Vite SPA + hanzoai/spa) |
| `onyx-plus/onyx-plus-internal.md` | Internal docs site (Next.js + @hanzo/docs + satschel.com gate) |
| `onyx-plus/onyx-plus-verify.md` | Standalone IDV flow (selfie + ID + thumbprint + proof-of-life) |
| `onyx-plus/onyx-plus-onboarding.md` | BD/KYC onboarding monorepo |
| `onyx-plus/onyx-plus-iam.md` | IAM tenant (onyxplus org, onyxplus-{admin,verify,onyxd} apps) |
| `onyx-plus/onyx-plus-kms.md` | KMS secret paths + ZAP transport |
| `onyx-plus/onyx-plus-mpc.md` | Wallet provision via BD → TA → MPC; JWT-claims invariant |
| `onyx-plus/onyx-plus-attestation.md` | ERC-735 per-topic claim signer (secp256k1) |
| `onyx-plus/onyx-plus-deploy.md` | Cloud Build + manifest bump + kubectl set image |

## Decision tree

```
Question                                  → Skill
--------                                  ---------
"How does (orgID,userID) → wallet?"       → onyx-plus-mpc.md
"How does onyxd read secrets?"            → onyx-plus-kms.md
"How does the verify SPA authenticate?"   → onyx-plus-iam.md
"What ERC-735 topics does onyxd issue?"   → onyx-plus-attestation.md
"How do I bump onyxd from 0.2.2 to 0.2.3?"→ onyx-plus-deploy.md
"What's in the admin dashboard?"          → onyx-plus-admin.md
"How is the internal docs site gated?"    → onyx-plus-internal.md
"Why is OnyxPlus better than Simplici?"   → onyx-plus.md
```

## Related skills

- `hanzo/hanzo-iam.md` - IAM upstream (Casdoor fork)
- `hanzo/hanzo-kms.md` - KMS upstream (ZapDB)
- `hanzo/hanzo-mpc.md` - MPC upstream (CGGMP21 + FROST)
- `hanzo/hanzo-base.md` - Backend framework onyxd embeds
- `hanzo/hanzo-docs.md` - Docs framework internal uses
- `hanzo/hanzo-id.md` - Client OAuth flow (verify + admin)
- `liquidity/liquidity-app.md` - Sibling tenant (BD/ATS/TA on same IAM)
- `liquidity/liquidity-universe.md` - Sibling deployment infra

---

**Last Updated**: 2026-05-12
