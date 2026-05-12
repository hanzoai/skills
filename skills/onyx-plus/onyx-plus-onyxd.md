---
name: onyx-plus-onyxd
description: OnyxPlus daemon (Go + Hanzo Base) -- enrollment, attestation, ERC-735 claim signing
---

# onyxd -- OnyxPlus Daemon

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-attestation.md`, `hanzo/hanzo-base.md`, `onyx-plus/onyx-plus-kms.md`, `onyx-plus/onyx-plus-mpc.md`

## Overview

`onyxd` is the OnyxPlus backend daemon. Go binary, port 8090, embeds Hanzo Base, lives at `onyxplus-api.{env}.satschel.com`.

- Module: `github.com/onyx-plus/onyxd`
- Data dir: `/data/onyxd`
- Routes: `/v1/onyxplus/*` (no `/api/` prefix)
- Image: `us-docker.pkg.dev/onyxplus-registry/onyx-plus/onyxd:{semver}`
- Live tag: `0.2.2` in dev/test/main as of 2026-05-12

## Hard requirements

1. **Hanzo Base embedded** -- `github.com/hanzoai/base` provides the auth, file storage, REST/GraphQL, admin dashboard
2. **Eager 1:1 resolve** -- `ResolveOnchainIdentity` runs in `handleSubmit` before `issueAttestation`; failure → 502 + critical admin event
3. **Zero MPC authority** -- onyxd never holds `MPC_SERVICE_KEY`; wallet provisioning goes through BD → TA → MPC
4. **Zero chain-write authority** -- all chain writes (createIdentity, registerIdentity, addClaim) go via BD's `/v1/identity/{ensure,register}` relay
5. **secp256k1 signing key in KMS** -- loaded at boot via `secrets.go::loadFromKMS` over ZAP; never written to disk

## Routes

| Method+Path | Purpose | Auth |
|---|---|---|
| `POST /v1/onyxplus/enrollments` | Create new enrollment row | Bearer JWT (verify/admin SPA) |
| `GET /v1/onyxplus/enrollments/{id}` | Read enrollment state | Bearer JWT |
| `POST /v1/onyxplus/enrollments/{id}/liveness` | Submit liveness capture | Bearer JWT |
| `POST /v1/onyxplus/enrollments/{id}/id-document` | Submit ID document hash | Bearer JWT |
| `POST /v1/onyxplus/enrollments/{id}/passkey` | Register WebAuthn passkey (proxies to MPC) | Bearer JWT |
| `POST /v1/onyxplus/enrollments/{id}/submit` | Finalise: resolve wallet+OnchainID, issue attestation | Bearer JWT |
| `GET /v1/onyx/claims/{enrollmentId}?onchain_id=0x...` | Return 4 ERC-735 claims for BD's `addClaim` calls | Bearer JWT |
| `GET /v1/onyxplus/users/{user_id}/attestation` | Legacy audit row | Bearer JWT |
| `POST /v1/onyxplus/users/{user_id}/verify` | Verify attestation by enrollment row | Bearer JWT |
| `GET /v1/onyxplus/.well-known/attestation-keys.json` | Issuer pubkey + ClaimIssuer address | Public |
| `GET /v1/onyxplus/healthz` | Health probe | Public |
| `GET /healthz` | Health probe (root) | Public |

## Required env vars

| Var | Required | Purpose |
|---|---|---|
| `MPC_URL` | yes | MPC mesh base for WebAuthn passkey register/challenge (wallet provisioning is via BD, not direct) |
| `BD_IDENTITY_ENSURE_URL` | yes | BD `/v1/identity/ensure` -- single round trip: wallet provision + OnchainID deploy |
| `IDENTITY_REGISTRY_URL` | yes | BD `/v1/identity/register` -- binds `(wallet, OnchainID, country)` on the on-chain `IdentityRegistry` |
| `BD_INTERNAL_SERVICE_KEY` | yes | shared bearer with BD; default-deny when unset |
| `ONYX_SIGNING_KEY` | yes | secp256k1 ECDSA root for ERC-735 claim signing (loaded from KMS at boot) |
| `ONYX_CLAIM_ISSUER_ADDR` | yes | deployed ClaimIssuer contract |
| `ONYX_PUBLIC_BASE_URL` | yes | claim `uri` field base, e.g. `https://onyxplus-api.{env}.satschel.com` |
| `IAM_ENDPOINT` | yes | shared Hanzo IAM URL |
| `IAM_CLIENT_ID` | yes | `onyxplus-onyxd` client id (substituted by operator from K8s Secret) |
| `KMS_ZAP_ADDR` | recommended | `<host>:9999` for ZAP boot-secret fetch |

## Boot order

```
1. enforceProdSafety()           — refuse boot if ONYX_DEMO_MODE set in prod-like env
2. mkdir DATA_DIR /data/onyxd
3. loadSecrets(ctx) over ZAP     — pull ONYX_SIGNING_KEY, IAM_CLIENT_SECRET, MPC_WEBHOOK_SECRET from KMS
4. base.NewWithConfig(...)       — Hanzo Base bootstrap
5. platform.MustRegister(...)    — IAM + KMS plugin
6. registerEnrollmentRoutes
7. registerAttestationRoutes
8. registerWebhookRoutes
9. registerAuditExportRoutes
10. registerAdminActionRoutes
11. app.Execute()                 — serve on :8090
```

## Build + deploy

```bash
# Local
go build -o /tmp/onyxd ./
go test -race ./...

# Image (NEVER on a laptop — always Cloud Build)
gcloud builds submit --config=cloudbuild.yaml --project=onyxplus-registry \
    --gcs-source-staging-dir=gs://onyxplus-registry_cloudbuild/source

# Deploy (dev → test → main)
kubectl --context=onyxplus-dev -n onyxplus set image statefulset/onyxd \
    onyxd=us-docker.pkg.dev/onyxplus-registry/onyx-plus/onyxd:0.2.2
kubectl --context=onyxplus-dev -n onyxplus rollout status statefulset/onyxd
```

`VERSION` file is the immutable semver tag. Cloud Build refuses to push when the tag already exists in GAR -- so each release bumps `VERSION` and commits before the build runs.

## Key source files

| File | Purpose |
|---|---|
| `main.go` | Boot orchestration, middleware, header-strip, CORS |
| `secrets.go` | ZAP KMS client; pulls signing key, IAM client secret, MPC webhook secret |
| `enrollment.go` | Enrollment CRUD + liveness + ID document + passkey + submit + on-chain register |
| `attestation.go` | ERC-735 per-topic claim signer (secp256k1, EIP-191) |
| `onchainid.go` | `EnsureOnchainID` (BD relay) + `CanonicalSalt` + `PredictedOnchainID` debug helper |
| `webhooks.go` | Inbound MPC challenge callbacks (HMAC-verified) |
| `audit_export.go` | SEC/FINRA/IRS subpoena response endpoint |
| `admin_actions.go` | Reject / override-liveness / re-enroll / revoke |
| `cloudbuild.yaml` | Cloud Build pipeline definition |
| `Dockerfile` | golang:1.26-alpine multi-stage; CGO for SQLite |

## Testing

```bash
# Unit + integration
go test -race -count=1 ./...

# Specific: canonical salt drift detection
go test -run TestCanonicalSalt -v ./

# E2E (compose stack required)
docker compose -f ~/work/liquidity/universe/compose.yml up -d
go test -tags=e2e ./...
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `wallet_address / onchain_id_address missing -- ResolveOnchainIdentity must run before issueAttestation` | submit handler skipped the eager-resolve | Re-trace `handleSubmit` -- `ResolveOnchainIdentity` MUST run before `issueAttestation`; do not move it into a goroutine |
| `mpc wallet provision failed` | BD `/v1/identity/ensure` unreachable or TA missing `MPC_SERVICE_KEY` | Check BD logs; verify TA env carries `MPC_SERVICE_KEY` |
| `ONYX_SIGNING_KEY must be 64 hex chars` | KMS returned malformed value | Re-seed `secret/data/onyxplus/{env}/onyx_signing_key` with `openssl rand -hex 32` |
| `face_hash missing` | Liveness step incomplete or biometric-skip path triggered | Either complete liveness or implement biometric-skip tier policy (still open) |

## Related skills

- `onyx-plus/onyx-plus-attestation.md` - ERC-735 claim signing detail
- `onyx-plus/onyx-plus-kms.md` - boot-time secret fetch
- `onyx-plus/onyx-plus-mpc.md` - wallet provision path
- `onyx-plus/onyx-plus-iam.md` - JWT validation
- `onyx-plus/onyx-plus-deploy.md` - Cloud Build + GKE rollout
- `hanzo/hanzo-base.md` - embedded backend

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: onyxd, daemon, go, attestation, erc-735
