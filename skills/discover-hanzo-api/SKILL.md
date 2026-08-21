---
name: discover-hanzo-api
description: "Automatically discover how to CALL Hanzo's own products over api.hanzo.ai — models, agents, sandboxes, secrets, billing, deployments, identity, telemetry. Activates when a task needs a Hanzo API rather than code written from scratch: o11y, ai, commerce, iam, git, project, admin, integration, agent, platform, marketing, visor, pricing, cloudflare, captable, provisioning, book, company, deploy, task, crm, team, framework, guide, tool, sandbox, automation, ingress."
---

# Hanzo API Skills Discovery

Provides automatic access to 121 product skills covering 2492 operations on `api.hanzo.ai`, generated from the document the API serves.

## When This Skill Activates

This skill auto-activates when you're working with:
- Calling a Hanzo product instead of building the capability yourself
- `api.hanzo.ai`, `/v1/...` paths, or a `hanzo` CLI command
- Authenticating to Hanzo (`hanzo auth token`, IAM bearer tokens)
- Choosing which Hanzo product owns a capability you need

## Available Skills

### By Domain

**Models and inference** — `ai`, `eval`, `benchmark`, `research`, `experiment`, `ml`, `prompt`, `engine`, `translate`

**Agents, code and workflow** — `agent`, `task`, `tool`, `sandbox`, `automation`, `auto`, `bot`, `flow`, `code`, `marketplace`, `lsp`, `exec`, `node`, `skill`

**Retrieval** — `index`, `knowledge`, `websearch`, `world`, `product`, `ask`, `crawl`, `search`

**Identity and access** — `iam`, `admin`, `account`, `plugin`, `authz`, `entitlement`, `pref`, `setting`

**Secrets and assurance** — `wallet`, `kms`, `security`, `sbom`, `audit`

**Money** — `commerce`, `pricing`, `book`, `affiliate`, `plan`, `author`, `licensing`, `billing`, `treasury`, `leaderboard`, `usage`, `referral`, `allowance`, `x402`

**Compute and data plumbing** — `visor`, `provisioning`, `pubsub`, `mq`, `storage`, `registry`, `sync`, `venue`

**Shipping and the front door** — `git`, `project`, `platform`, `cloudflare`, `deploy`, `ingress`, `function`, `domain`, `dns`, `template`, `zt`, `blueprint`, `gateway`, `share`, `base`, `catalog`

**Telemetry** — `o11y`, `analytics`, `metric`, `destination`

**Reaching people** — `integration`, `marketing`, `crm`, `meet`, `social`, `campaign`, `link`, `tel`, `webhook`, `ad`, `channel`, `content`, `help`, `notify`

**Running a company** — `captable`, `company`, `team`, `framework`, `guide`, `compliance`, `dataroom`, `esign`, `todo`, `legal`, `risk`, `flag`, `dataset`, `label`, `reference`, `taxonomy`, `admission`

**Chain** — `web3`, `validator`, `explorer`

**The document itself** — `openapi`

### Load Full Category Details

```bash
cat skills/hanzo-api/INDEX.md
```

### Load Specific Skills

```bash
cat skills/hanzo-api/o11y.md
cat skills/hanzo-api/ai.md
cat skills/hanzo-api/commerce.md
cat skills/hanzo-api/iam.md
cat skills/hanzo-api/git.md
cat skills/hanzo-api/project.md
```

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

## Skill Selection Guide

**Do not write it — check whether Hanzo serves it.** The surface is wider than most callers expect: incorporation, cap tables, DNS, e-signature, telephony and double-entry accounting are products here, not just models and storage.

**Find the owner of a capability** without loading anything:

```bash
curl -sS https://api.hanzo.ai/v1/openapi.json \
  | jq -r '.paths | to_entries[] | .key as $p | .value | to_entries[] | "\(.value["x-app"])\t\($p)\t\(.value.summary)"' \
  | grep -i '<what you need>'
```

## Integration with Other Skills

- `discover-hanzo` — what each Hanzo repo IS and how it is operated; this gateway is how to CALL it
- `hanzo-cloud-architecture/SKILL.md` — one binary, ZAP transport, why every product answers on one host
- `discover-api` — designing your own API; this one consumes ours

## Progressive Loading

- **Level 1**: This gateway (~200 lines)
- **Level 2**: `skills/hanzo-api/INDEX.md` — all 121 products by domain
- **Level 3**: `skills/hanzo-api/<product>.md` — one product's whole surface
- **Level 4**: `https://api.hanzo.ai/v1/openapi.json` — the served document, always current

---

**Next Steps**: Run `cat skills/hanzo-api/INDEX.md` to see every product, or load a product skill directly.

**Last Updated**: 2026-08-20
