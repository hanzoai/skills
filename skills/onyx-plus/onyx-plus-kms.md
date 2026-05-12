---
name: onyx-plus-kms
description: OnyxPlus KMS consumer -- ZAP boot-secret fetch; signing key, IAM client secret, MPC webhook secret
---

# OnyxPlus KMS Consumer

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `hanzo/hanzo-kms.md`, `onyx-plus/onyx-plus-iam.md`

## Overview

OnyxPlus does not run its own KMS. `liquid-kms` (image `liquidityio/kms`) holds every secret for BD, ATS, TA, AML, the gateway, and onyxd. Two ports:

- `:8443` -- HTTPS REST (`/v1/kms/secrets/...`, JWT-gated)
- `:9999` -- ZAP binary transport (one JWT validation on `INIT`, then long-lived stream)

Encrypted store: ZapDB at `/data/kms`.

## OnyxPlus secret paths

```
secret/data/onyxplus/{env}/onyx_signing_key   # secp256k1 ECDSA, ERC-735 signer
secret/data/onyxplus/{env}/iam_client_secret  # onyxplus-onyxd OAuth2 client_credentials secret
secret/data/onyxplus/{env}/mpc_webhook_secret # HMAC for inbound MPC callbacks
```

Bootstrap env on the onyxd pod carries exactly one credential (`KMS_ZAP_ADDR`); every other secret lands from KMS at boot.

## Boot path

```go
// onyxd/secrets.go
func loadFromKMS(ctx context.Context, addr string) (Secrets, error) {
    env := envStr("IAM_ENV", "dev")
    path := fmt.Sprintf("secret/data/onyxplus/%s", env)

    c, err := zapclient.Dial(dialCtx, addr, path)
    if err != nil { /* ... */ }
    defer c.Close()

    sign, _ := c.Get(getCtx, "SIGNING_KEY_PEM", env)
    iam, _  := c.Get(getCtx, "IAM_CLIENT_SECRET", env)
    mpc, _  := c.Get(getCtx, "MPC_WEBHOOK_SECRET", env)
    // ...
}
```

If `KMS_ZAP_ADDR` is empty the loader falls back to env vars -- production deploys never trip the fallback.

## Canonical KMS environment

```
KMS_IAM_URL              = https://iam.{env}.satschel.com   # http://iam:8000 in compose
KMS_JWKS_URL             = ${KMS_IAM_URL}/.well-known/jwks
KMS_EXPECTED_ISSUER      = ${KMS_IAM_URL}                   # MUST match IAM's ORIGIN
KMS_EXPECTED_AUDIENCE    = liquidity-bd,liquidity-ats,liquidity-kms,liquidity-ta,onyxplus-onyxd
KMS_ZAP_AUDIENCE         = liquidity-kms
ZAP_PORT                 = 9999
KMS_ZAP_AUTH_ENABLED     = true
MPC_ADDR                 = mpc-0:9653,mpc-1:9663,mpc-2:9673
MPC_VAULT_ID             = liquidity-{env}                  # liquidity-local in compose
```

## ZAP transport

```
client                                   KMS / MPC
  |                                          |
  | ZAP INIT { jwt }              ----->     validate JWT once on connect
  | ZAP INIT_ACK { capability id} <-----     (iss/aud/JWKS, same path)
  |                                          |
  | ZAP PUSH method/params         ----->    authorise (capability)
  | ZAP RESOLVE result             <-----
  | (long-lived stream)                       |
```

`KMS_ZAP_AUTH_ENABLED=true` activates JWT validation on the ZAP listener. JWT verified at connect; per-call authorisation rides the capability id returned in `INIT_ACK`.

## Failure modes

| Symptom | Cause | Where it shows |
|---|---|---|
| `401 token has invalid issuer` | IAM `ORIGIN` differs from `KMS_EXPECTED_ISSUER` | onyxd boot log |
| `403 audience not allowed` | onyxplus-onyxd JWT missing `liquidity-kms` audience | KMS access logs |
| `KMS dial: connection refused` | ZAP listener not ready (boot race) | onyxd boot log; compose `depends_on` should prevent in dev/test/main |
| `KMS get NAME: not found` | Secret path not seeded for this env | KMS access logs -- populate before pod rollout |

## Seeding (compose / first deploy)

```bash
# In compose, the KMS container seeds itself from /init_data.json; for new
# paths, use the REST API:
ADMIN_JWT="..."  # liquidity-kms service JWT
curl -sS -X PUT \
    -H "Authorization: Bearer ${ADMIN_JWT}" \
    -H "Content-Type: application/json" \
    -d '{"value":"'"$(openssl rand -hex 32)"'"}' \
    https://localhost:8443/v1/kms/secrets/onyxplus/local/onyx_signing_key
```

In K8s, the `kms-operator` reconciles `KMSSecret` CRDs into the underlying KMS. Per `~/.claude/CLAUDE.md`, `KMSSecret` apiGroup `secrets.lux.network/v1alpha1` is the upstream OSS API surface -- the CRD's apiGroup is a fact about the cluster, not a brand prefix on our stack.

## Bootstrap order

```
iam → kms → mpc-{0,1,2} → ats / bd / ta / aml → onyxd → exchange / id / verify / swap / pay / ...
```

onyxd blocks on `KMS_ZAP_ADDR` until the listener is healthy (`GET /healthz` 200), then fetches its secret bundle. A missing secret aborts the boot -- better a CrashLoop than a daemon running with empty signing key.

## Source pointers

| Subject | Source |
|---|---|
| Caller (onyxd) | `~/work/onyxplus/onyxd/secrets.go::loadFromKMS` |
| Caller (BD, for reference) | `~/work/liquidity/bd/secrets.go` |
| KMS embed config | `~/work/hanzo/kms/pkg/kms/embed.go::EmbedConfig` |
| ZAP server | `~/work/hanzo/kms/pkg/zap/server.go` |
| Compose service block | `~/work/liquidity/universe/compose.yml` (`liquid-kms` section) |
| Internal docs page | `~/work/onyxplus/internal/content/docs/kms.mdx` |
| Long-form paper | `~/work/onyxplus/papers/onyx-plus-kms/onyx-plus-kms.pdf` |

## Related skills

- `hanzo/hanzo-kms.md` - KMS upstream
- `onyx-plus/onyx-plus-iam.md` - JWT issuer that gates KMS
- `onyx-plus/onyx-plus-onyxd.md` - boot-time secret fetch caller

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: kms, secrets, zap, zapdb, jwt, infisical
