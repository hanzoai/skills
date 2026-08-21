---
name: hanzo-api-kms
description: Secret custody: your org's secrets sealed at rest, plus threshold signing.
---

# Hanzo KMS

**Scope**: Secret custody: your org's secrets sealed at rest, plus threshold signing.
**Surface**: 7 operations on `/v1/kms`
**Lines**: ~123
**Last Updated**: 2026-08-20
**Source**: https://api.hanzo.ai/v1/openapi.json (`x-app: kms`) — generated, do not hand-edit

## When to Use This Skill

Activate this skill when:
- Your code needs a credential and you refuse to put it in a manifest or an env file
- Rotating a secret without redeploying whatever reads it
- You need a signature from a key the caller is never allowed to hold
- Auditing what secrets an org holds without revealing any value
- Calling `/v1/kms` (7 operations)

## Authentication

Every operation takes a Hanzo IAM bearer token. `hanzo auth token` prints the
active short-lived access token for the identity you are signed in as; the org
is carried in the token's claim, never in a path or a query parameter.

```bash
TOKEN=$(hanzo auth token)
curl -sS https://api.hanzo.ai/v1/... -H "Authorization: Bearer $TOKEN"
```

The document declares one scheme, `bearer`, at the top level: `Authorization:
Bearer <token>`. `X-Authorization` and HTTP Basic carrying the token as the
password are accepted spellings of the same header; a browser presents the
session cookie instead. There is no product-specific key scheme.

## Surface

| Family | Operations | Addresses under it |
|---|---|---|
| `/v1/kms` | 7 | `auth`, `config`, `health`, `secrets` |

## Operations

### `/v1/kms`

| Method | Path | Summary |
|---|---|---|
| `POST` | `/v1/kms/auth/login` | Exchange a machine credential for an IAM bearer token |
| `GET` | `/v1/kms/config` | Runtime configuration for the KMS console |
| `GET` | `/v1/kms/health` | Whether this broker can actually serve secrets |
| `GET` | `/v1/kms/secrets` | List the secrets your org holds, without their values |
| `POST` | `/v1/kms/secrets` | Store or replace one secret in your org |
| `DELETE` | `/v1/kms/secrets/{wildcard1}` | Delete one secret from your org |
| `GET` | `/v1/kms/secrets/{wildcard1}` | Read one secret's value |

## Example Calls

```bash
TOKEN=$(hanzo auth token)
```

**Store or replace one secret in your org**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/kms/secrets" \
  -H "Authorization: Bearer $TOKEN"
```

Upserts one secret under the caller's own org. The value is sealed before it is written — a fresh per-secret data key, itself wrapped by the master key — so plaintext never reaches disk. The receipt confirms the name and environment that were written and does not echo the value.

`env` is REQUIRED on a write and has no default, which is the rule most easily got wrong here: reads and deletes still fall back to the default environment for older callers, but a write must not, because the environment is part of the storage key. A silently defaulted write lands in a bucket the readers that resolve project, environment and path never look in, and the stale value keeps being served — so the write…

`name` is required, `path` is an optional subpath beneath the org root, and the org is taken from the validated claim rather than the body.

Requires ADMIN authority over the org — a member reads, an admin writes. A machine credential holds no membership and so is never an org admin: it can read the secrets it was issued for and cannot replace one. Fail-closed admission, in order: admin of the org, well-formed org, master key present — 403, 400 and 503, all decided before any record is touched.

**Exchange a machine credential for an IAM bearer token**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/kms/auth/login" \
  -H "Authorization: Bearer $TOKEN"
```

Takes a tenant's machine credential — a client id and client secret — and returns an owner-scoped IAM access token with its lifetime, which is the bearer the caller then carries on the org-scoped secret operations.

It is deliberately public and unauthenticated, because it IS the credential exchange and runs before any principal exists. That makes it the one route in this subsystem rate-limited PER SOURCE IP, keyed on the real TCP peer rather than on any caller-supplied header.

The submitted secret is never logged and never echoed, and failures collapse to one clean status with no upstream detail: 401 when the credential does not authenticate, 502 when the identity provider is unreachable, 503 when no issuer is configured. That is on purpose — a richer error would be a validity oracle for guessed credentials.

**List the secrets your org holds, without their values**

```bash
curl -sS "https://api.hanzo.ai/v1/kms/secrets" \
  -H "Authorization: Bearer $TOKEN"
```

Returns the METADATA of the caller's own secrets: each one's name, path, environment and sealing scheme. No value and no ciphertext is included — this operation exists to enumerate what is held, and reading a value is a separate, per-secret call.

## Composition

- Everything that takes a third-party credential reads it from here — `integrations`, `notify`, `settings`
- `iam` decides who the caller is; this decides what that caller may unseal
- `audit` records the read; the two together are the evidence trail
- `platform` and `functions` inject secrets into a deployment by reference, never by value

## Related Skills

- `hanzo-api/wallet.md` — Blockchain key custody: create wallets, rotate their keys, and sign with them.
- `hanzo-api/security.md` — Secret scanning for your code: submit sources, get findings, masked never raw.
- `hanzo-api/sbom.md` — What is inside a container image: every component, resolvable by digest or image ref.
- `hanzo-api/audit.md` — Your org's tamper-evident audit trail: every security-relevant event, hash-chained and…
- `hanzo/hanzo-kms.md` — the repo and how it is operated, not the API surface
- `hanzo-cloud-architecture/SKILL.md` — how one binary serves all of this
- `hanzo-api/INDEX.md` — every Hanzo product, by domain

---

**Last Updated**: 2026-08-20
**Generated from**: https://api.hanzo.ai/v1/openapi.json
