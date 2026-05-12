---
name: onyx-plus-iam
description: OnyxPlus IAM tenant -- Hanzo IAM fork shared with Liquidity; onyxplus org + onyxplus-{admin,verify,onyxd} apps
---

# OnyxPlus IAM Tenant

**Category**: OnyxPlus
**Related Skills**: `onyx-plus/onyx-plus.md`, `hanzo/hanzo-iam.md`, `onyx-plus/onyx-plus-kms.md`

## Overview

OnyxPlus does not run its own IAM. It is a tenant in the shared Liquidity IAM (`liquid-iam` at `v1.14.16+`) that already gates BD, ATS, TA, AML, the gateway, and the SPAs. The same JWKS that lets BD read KMS lets `onyxd` read KMS -- there is no separate trust anchor.

Hosts:
- local (compose): `http://iam:8000`
- dev: `https://iam.dev.satschel.com`
- test: `https://iam.test.satschel.com`
- main: `https://iam.main.satschel.com`

JWKS at `${IAM_URL}/.well-known/jwks`. Every JWT-validating consumer (KMS, MPC, gateway, onyxd) caches it (atomic-pointer cache, 50k+ RPS per core after `v1.14.15`).

## Seed (init_data.json)

`k8s/iam/init_data.json` mounted read-only at `/init_data.json`. `initDataNewOnly=true` so re-mount never overwrites.

### Organisations
- `liquidity` -- canonical for BD/ATS/TA/AML + exchange SPAs
- `onyxplus` -- peer tenant for OnyxPlus admin + verify SPAs + onyxd service account

### SPA applications (`authorization_code`)

| Name | Tenant | Surface |
|---|---|---|
| `onyxplus-admin` | onyxplus | `onyxplus.{env}.satschel.com` |
| `onyxplus-verify` | onyxplus | `verify.{env}.satschel.com` |
| `liquidity-exchange-client-id` | liquidity | `exchange.{env}.satschel.com` |
| `liquidity-superadmin` | liquidity | superadmin SPA |
| `liquidity-id` | liquidity | BD onboarding (`id.{env}.satschel.com`) |
| `liquidity-pay-client-id` | liquidity | pay SPA |

### Service applications (`client_credentials`)

| Name | `aud` claim |
|---|---|
| `liquidity-kms` | `liquidity-kms` |
| `liquidity-bd` | `liquidity-bd,liquidity-kms` |
| `liquidity-ats` | `liquidity-ats,liquidity-kms` |
| `liquidity-ta` | `liquidity-ta,liquidity-kms` |
| `onyxplus-onyxd` | `onyxplus-onyxd,liquidity-kms` |

`onyxplus-onyxd` needs both: `liquidity-kms` so it reads secrets, `onyxplus-onyxd` so inbound callers can scope to the daemon's surface.

## Invariants (failure modes)

| Invariant | Symptom on miss | Fix |
|---|---|---|
| `iss` byte-for-byte = `KMS_EXPECTED_ISSUER` | `401 token has invalid issuer` (silent logout on refresh) | Set IAM `ORIGIN` env to the public URL the JWT will carry. Without `ORIGIN`, Hanzo IAM mints `iss` from the inbound `Host` header -- yields a token KMS rejects |
| `aud` ∈ `KMS_EXPECTED_AUDIENCE` (comma-separated) | `403 audience not allowed` | Service IAM app must list every audience it needs. Per-tenant audiences coexist on one KMS instance |
| JWKS cached with strong ETag (`v1.14.15+`) | JWT validation slow | IAM serves `/.well-known/jwks` from a pre-marshalled bytes cache. `If-None-Match` returns `304`. Throughput went from ~565 RPS to >50 000 RPS |
| Init data run with `initDataNewOnly=true` | First boot wipes existing rows on PVC re-mount | Container env must include `INIT_DATA_NEW_ONLY=true` |

## End-user OAuth flow

```
1. User clicks Login on verify.{env}.satschel.com or onyxplus.{env}.satschel.com
2. SPA → iam.{env}.satschel.com/login/oauth/authorize?
 client_id=onyxplus-{verify|admin}&redirect_uri=...&scope=openid&state=...
3. IAM authenticates (password / Google / phone+OTP; compose sandbox: OTP 999999 always succeeds)
4. IAM → redirect back with ?code=...
5. SPA exchanges code at /login/oauth/access_token for a JWT
6. SPA calls onyxd with Authorization: Bearer <JWT>
7. onyxd validates JWT against IAM's JWKS (cached); trusts sub (user id) and org claims
```

## Service-to-service round trip

```
1. onyxd holds (clientId, clientSecret) in env (substituted by operator from K8s Secret)
2. onyxd POSTs grant_type=client_credentials to IAM /login/oauth/access_token
 → IAM signs JWT with cert-superuser key; returns access token carrying
 iss=https://iam.{env}.satschel.com/, aud=onyxplus-onyxd,liquidity-kms, exp=now+24h
3. onyxd → GET /v1/kms/secrets/onyxplus/{env}/onyx_signing_key on KMS with Bearer JWT
4. KMS verifies signature, asserts iss/aud, returns {value}
5. onyxd loads value into ONYX_SIGNING_KEY in process memory
```

## Manual reproduction (compose)

```bash
# Read live credentials for the onyxd service account
ONYX_CID=$(docker exec liquid-iam sqlite3 -readonly /data/iam/iam.db \
 "SELECT client_id FROM application WHERE name='onyxplus-onyxd';")
ONYX_SECRET=$(docker exec liquid-iam sqlite3 -readonly /data/iam/iam.db \
 "SELECT client_secret FROM application WHERE name='onyxplus-onyxd';")

# Get a JWT
JWT=$(curl -sS -X POST \
 -H "Content-Type: application/x-www-form-urlencoded" \
 -d "grant_type=client_credentials&client_id=${ONYX_CID}&client_secret=${ONYX_SECRET}" \
 http://localhost:8000/login/oauth/access_token | jq -r .access_token)

# Read OnyxPlus signing key from KMS
curl -sS -H "Authorization: Bearer ${JWT}" \
 https://localhost:8443/v1/kms/secrets/onyxplus/local/onyx_signing_key
```

`200` (or `404` if the path is not seeded yet) confirms the chain is healthy. `401` means IAM and KMS disagree on `iss`/`aud`/JWKS.

## Source pointers

| Subject | Source |
|---|---|
| IAM compose service block | `~/work/liquidity/universe/compose.yml` |
| Seed file | `~/work/liquidity/universe/k8s/iam/init_data.json` |
| JWKS cache + serving | `~/work/hanzo/iam/object/jwks_cache.go` and `controllers/wellknown_oidc_discovery.go::serveJwksBytes` |
| Password hashing (argon2id default) | `~/work/hanzo/iam/cred/cred.go::GetCredManager` |
| Init seed code | `~/work/hanzo/iam/object/init_data.go::initDefinedUser` |
| Internal docs page | `~/work/onyxplus/internal/content/docs/iam.mdx` |
| Long-form paper | `~/work/onyxplus/papers/onyx-plus-iam/onyx-plus-iam.pdf` |

## Related skills

- `hanzo/hanzo-iam.md` - IAM upstream (full Hanzo IAM surface)
- `onyx-plus/onyx-plus-kms.md` - KMS consumer
- `onyx-plus/onyx-plus-mpc.md` - MPC consumer
- `liquidity/liquidity-app.md` - sibling tenant

---

**Last Updated**: 2026-05-12
**Category**: OnyxPlus
**Related**: iam, oauth, iam, jwks, jwt, audiences
