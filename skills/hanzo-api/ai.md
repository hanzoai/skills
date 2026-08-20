---
name: hanzo-api-ai
description: Hanzo AI — the model API on /v1 (/v1/chat/completions, /v1/messages, /v1/models and the rest of hanzoai/ai's surface) — mounted into a cloud binary with the money, ingest and telemetry callbacks…
---

# Hanzo AI

**Scope**: Hanzo AI — the model API on /v1 (/v1/chat/completions, /v1/messages, /v1/models and the rest of hanzoai/ai's surface) —…
**Surface**: 302 operations on `/v1/ai`, `/v1/audio`, `/v1/chat`, `/v1/completions`, `/v1/dev-bridge`, `/v1/docs`, `/v1/documents`, `/v1/embeddings` and 17 more
**Lines**: ~511
**Last Updated**: 2026-08-20
**Source**: https://api.hanzo.ai/v1/openapi.json (`x-app: ai`) — generated, do not hand-edit

## When to Use This Skill

Activate this skill when:
- Calling a model — chat, messages, embeddings, images, audio, video, rerank
- You want one credential and one base URL across every provider
- Porting code written against the OpenAI or Anthropic SDK: point base_url here
- Choosing a model: /v1/models carries context window, modalities and price
- Calling `/v1/ai` (208 operations)
- Calling `/v1/router` (30 operations)
- Calling `/v1/org` (10 operations)
- Calling `/v1/finetune` (9 operations)
- Calling `/v1/memory` (7 operations)
- Calling `/v1/audio` (5 operations)
- Calling any of this product's 19 smaller families (see Surface below)

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
| `/v1/ai` | 208 | `account`, `activities`, `answer`, `articles`, `assets`, `chats`, `connections`, `dashboards`, +30 |
| `/v1/audio` | 5 | `foley`, `music`, `speech`, `transcriptions`, `voice` |
| `/v1/chat` | 3 | `completions`, `public` |
| `/v1/completions` | 1 | the family root |
| `/v1/dev-bridge` | 1 | the family root |
| `/v1/docs` | 1 | `ingest` |
| `/v1/documents` | 2 | the family root |
| `/v1/embeddings` | 1 | the family root |
| `/v1/feedback` | 1 | the family root |
| `/v1/finetune` | 9 | `cancel`, `deploy`, `hf`, `job`, `jobs`, `presets` |
| `/v1/images` | 1 | `generations` |
| `/v1/install-patch` | 1 | the family root |
| `/v1/memory` | 7 | `delete`, `facts`, `list`, `recall`, `remember`, `search`, `update` |
| `/v1/messages` | 2 | `count_tokens` |
| `/v1/models` | 4 | `providers` |
| `/v1/org` | 10 | `settings` |
| `/v1/query` | 1 | the family root |
| `/v1/query_multiple` | 1 | the family root |
| `/v1/rag` | 5 | `context`, `delete`, `embed`, `query`, `query-multiple` |
| `/v1/rerank` | 1 | the family root |
| `/v1/responses` | 1 | the family root |
| `/v1/router` | 30 | `artifact-meta`, `data`, `defaults`, `history`, `judge-panel`, `ledger`, `policy`, `rewards`, +1 |
| `/v1/traffic` | 1 | `globe` |
| `/v1/videos` | 3 | `generations` |
| `/v1/wecom-bot` | 2 | `callback` |

## Operations

302 operations — listed by address. Read any one's prose from the document:

```bash
curl -sS https://api.hanzo.ai/v1/openapi.json \
  | jq '.paths | to_entries[] | select(.value[]."x-app" == "ai") | {path: .key, ops: (.value | map_values(.summary))}'
```

### `/v1/ai`

- `GET /v1/ai/account`
- `GET /v1/ai/activities`
- `GET /v1/ai/answer`
- `GET /v1/ai/articles`
- `POST /v1/ai/articles`
- `GET /v1/ai/articles/global`
- `DELETE /v1/ai/articles/{owner}/{name}`
- `GET /v1/ai/articles/{owner}/{name}`
- `PATCH /v1/ai/articles/{owner}/{name}`
- `PUT /v1/ai/articles/{owner}/{name}`
- `GET /v1/ai/assets`
- `POST /v1/ai/assets`
- `POST /v1/ai/assets/scan`
- `DELETE /v1/ai/assets/{owner}/{name}`
- `GET /v1/ai/assets/{owner}/{name}`
- `PATCH /v1/ai/assets/{owner}/{name}`
- `PUT /v1/ai/assets/{owner}/{name}`
- `POST /v1/ai/assets/{owner}/{name}/scan`
- `GET /v1/ai/chats`
- `POST /v1/ai/chats`
- `GET /v1/ai/chats/global`
- `DELETE /v1/ai/chats/{owner}/{name}`
- `GET /v1/ai/chats/{owner}/{name}`
- `PATCH /v1/ai/chats/{owner}/{name}`
- `PUT /v1/ai/chats/{owner}/{name}`
- `GET /v1/ai/connections`
- `POST /v1/ai/connections`
- `DELETE /v1/ai/connections/{provider}`
- `POST /v1/ai/connections/{provider}`
- `GET /v1/ai/connections/{provider}/authorize`
- `GET /v1/ai/connections/{provider}/callback`
- `GET /v1/ai/connections/{provider}/usage`
- `GET /v1/ai/dashboards/agents`
- `GET /v1/ai/dashboards/vm`
- `GET /v1/ai/deployments`
- `POST /v1/ai/deployments`
- `DELETE /v1/ai/deployments/{owner}/{name}`
- `GET /v1/ai/deployments/{owner}/{name}`
- `PATCH /v1/ai/deployments/{owner}/{name}`
- `PUT /v1/ai/deployments/{owner}/{name}`
- `POST /v1/ai/deployments/{owner}/{name}/deploy`
- `POST /v1/ai/deployments/{owner}/{name}/undeploy`
- `GET /v1/ai/files`
- `POST /v1/ai/files`
- `POST /v1/ai/files/activate`
- `GET /v1/ai/files/active`
- `GET /v1/ai/files/global`
- `POST /v1/ai/files/upload`
- `DELETE /v1/ai/files/{owner}/{name}`
- `GET /v1/ai/files/{owner}/{name}`
- `PATCH /v1/ai/files/{owner}/{name}`
- `PUT /v1/ai/files/{owner}/{name}`
- `POST /v1/ai/files/{owner}/{name}/vectors`
- `GET /v1/ai/forms`
- `POST /v1/ai/forms`
- `GET /v1/ai/forms/data`
- `GET /v1/ai/forms/global`
- `DELETE /v1/ai/forms/{owner}/{name}`
- `GET /v1/ai/forms/{owner}/{name}`
- `PATCH /v1/ai/forms/{owner}/{name}`
- `PUT /v1/ai/forms/{owner}/{name}`
- `GET /v1/ai/graphs`
- `POST /v1/ai/graphs`
- `GET /v1/ai/graphs/global`
- `DELETE /v1/ai/graphs/{owner}/{name}`
- `GET /v1/ai/graphs/{owner}/{name}`
- `PATCH /v1/ai/graphs/{owner}/{name}`
- `PUT /v1/ai/graphs/{owner}/{name}`
- `GET /v1/ai/k8s-status`
- `GET /v1/ai/mcp/tools`
- `GET /v1/ai/messages`
- `POST /v1/ai/messages`
- `GET /v1/ai/messages/global`
- `DELETE /v1/ai/messages/welcome`
- `DELETE /v1/ai/messages/{owner}/{name}`
- `GET /v1/ai/messages/{owner}/{name}`
- `PATCH /v1/ai/messages/{owner}/{name}`
- `PUT /v1/ai/messages/{owner}/{name}`
- `GET /v1/ai/messages/{owner}/{name}/answer`
- `GET /v1/ai/nodes`
- `POST /v1/ai/nodes`
- `DELETE /v1/ai/nodes/{owner}/{name}`
- `GET /v1/ai/nodes/{owner}/{name}`
- `PATCH /v1/ai/nodes/{owner}/{name}`
- `PUT /v1/ai/nodes/{owner}/{name}`
- `GET /v1/ai/nodes/{owner}/{name}/tunnel`
- `POST /v1/ai/nodes/{owner}/{name}/tunnel`
- `PATCH /v1/ai/preferences`
- `PUT /v1/ai/preferences`
- `GET /v1/ai/prometheus`
- `GET /v1/ai/providers`
- `POST /v1/ai/providers`
- `GET /v1/ai/providers/global`
- `POST /v1/ai/providers/mcp-tools`
- `DELETE /v1/ai/providers/{owner}/{name}`
- `GET /v1/ai/providers/{owner}/{name}`
- `PATCH /v1/ai/providers/{owner}/{name}`
- `PUT /v1/ai/providers/{owner}/{name}`
- `GET /v1/ai/records`
- `POST /v1/ai/records`
- `POST /v1/ai/records/batch`
- `POST /v1/ai/records/commit`
- `POST /v1/ai/records/commit-second`
- `GET /v1/ai/records/query`
- `GET /v1/ai/records/query-second`
- `DELETE /v1/ai/records/{owner}/{name}`
- `GET /v1/ai/records/{owner}/{name}`
- `PATCH /v1/ai/records/{owner}/{name}`
- `PUT /v1/ai/records/{owner}/{name}`
- `GET /v1/ai/remote-connections`
- `POST /v1/ai/remote-connections`
- `DELETE /v1/ai/remote-connections/{owner}/{name}`
- `GET /v1/ai/remote-connections/{owner}/{name}`
- `PATCH /v1/ai/remote-connections/{owner}/{name}`
- `PUT /v1/ai/remote-connections/{owner}/{name}`
- `POST /v1/ai/remote-connections/{owner}/{name}/start`
- `POST /v1/ai/remote-connections/{owner}/{name}/stop`
- `GET /v1/ai/routes`
- `POST /v1/ai/routes`
- `DELETE /v1/ai/routes/{owner}/{name}`
- `GET /v1/ai/routes/{owner}/{name}`
- `PATCH /v1/ai/routes/{owner}/{name}`
- `PUT /v1/ai/routes/{owner}/{name}`
- `GET /v1/ai/scales`
- `POST /v1/ai/scales`
- `GET /v1/ai/scales/global`
- `GET /v1/ai/scales/public`
- `DELETE /v1/ai/scales/{owner}/{name}`
- `GET /v1/ai/scales/{owner}/{name}`
- `PATCH /v1/ai/scales/{owner}/{name}`
- `PUT /v1/ai/scales/{owner}/{name}`
- `GET /v1/ai/scans`
- `POST /v1/ai/scans`
- `DELETE /v1/ai/scans/{owner}/{name}`
- `GET /v1/ai/scans/{owner}/{name}`
- `PATCH /v1/ai/scans/{owner}/{name}`
- `PUT /v1/ai/scans/{owner}/{name}`
- `POST /v1/ai/signin`
- `GET /v1/ai/signin-sessions`
- `POST /v1/ai/signin-sessions`
- `GET /v1/ai/signin-sessions/duplicated`
- `DELETE /v1/ai/signin-sessions/{owner}/{name}`
- `GET /v1/ai/signin-sessions/{owner}/{name}`
- `PATCH /v1/ai/signin-sessions/{owner}/{name}`
- `PUT /v1/ai/signin-sessions/{owner}/{name}`
- `POST /v1/ai/signout`
- `GET /v1/ai/stores`
- `POST /v1/ai/stores`
- `GET /v1/ai/stores/global`
- `GET /v1/ai/stores/names`
- `GET /v1/ai/stores/providers`
- `DELETE /v1/ai/stores/{owner}/{name}`
- `GET /v1/ai/stores/{owner}/{name}`
- `PATCH /v1/ai/stores/{owner}/{name}`
- `PUT /v1/ai/stores/{owner}/{name}`
- `POST /v1/ai/stores/{owner}/{name}/vectors`
- `GET /v1/ai/system`
- `GET /v1/ai/tasks`
- `POST /v1/ai/tasks`
- `GET /v1/ai/tasks/global`
- `DELETE /v1/ai/tasks/{owner}/{name}`
- `GET /v1/ai/tasks/{owner}/{name}`
- `PATCH /v1/ai/tasks/{owner}/{name}`
- `PUT /v1/ai/tasks/{owner}/{name}`
- `POST /v1/ai/tasks/{owner}/{name}/analyze`
- `POST /v1/ai/tasks/{owner}/{name}/document`
- `GET /v1/ai/templates`
- `POST /v1/ai/templates`
- `DELETE /v1/ai/templates/{owner}/{name}`
- `GET /v1/ai/templates/{owner}/{name}`
- `PATCH /v1/ai/templates/{owner}/{name}`
- `PUT /v1/ai/templates/{owner}/{name}`
- `GET /v1/ai/training-contribution`
- `PATCH /v1/ai/training-contribution`
- `PUT /v1/ai/training-contribution`
- `POST /v1/ai/tree-files`
- `DELETE /v1/ai/tree-files/{owner}/{name}`
- `PATCH /v1/ai/tree-files/{owner}/{name}`
- `PUT /v1/ai/tree-files/{owner}/{name}`
- `GET /v1/ai/usages`
- `GET /v1/ai/usages/by-user`
- `GET /v1/ai/usages/cloud`
- `GET /v1/ai/usages/range`
- `GET /v1/ai/usages/user-names`
- `GET /v1/ai/vectors`
- `POST /v1/ai/vectors`
- `DELETE /v1/ai/vectors/all`
- `GET /v1/ai/vectors/global`
- `DELETE /v1/ai/vectors/{owner}/{name}`
- `GET /v1/ai/vectors/{owner}/{name}`
- `PATCH /v1/ai/vectors/{owner}/{name}`
- `PUT /v1/ai/vectors/{owner}/{name}`
- `GET /v1/ai/version`
- `GET /v1/ai/videos`
- `POST /v1/ai/videos`
- `GET /v1/ai/videos/global`
- `POST /v1/ai/videos/upload`
- `DELETE /v1/ai/videos/{owner}/{name}`
- `GET /v1/ai/videos/{owner}/{name}`
- `PATCH /v1/ai/videos/{owner}/{name}`
- `PUT /v1/ai/videos/{owner}/{name}`
- `GET /v1/ai/workflows`
- `POST /v1/ai/workflows`
- `GET /v1/ai/workflows/global`
- `DELETE /v1/ai/workflows/{owner}/{name}`
- `GET /v1/ai/workflows/{owner}/{name}`
- `PATCH /v1/ai/workflows/{owner}/{name}`
- `PUT /v1/ai/workflows/{owner}/{name}`

### `/v1/audio`

- `POST /v1/audio/foley`
- `POST /v1/audio/music`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/voice`

### `/v1/chat`

- `POST /v1/chat`
- `POST /v1/chat/completions`
- `POST /v1/chat/public`

### `/v1/completions`

- `POST /v1/completions`

### `/v1/dev-bridge`

- `GET /v1/dev-bridge`

### `/v1/docs`

- `POST /v1/docs/ingest`

### `/v1/documents`

- `DELETE /v1/documents`
- `GET /v1/documents/{file_id}/context`

### `/v1/embeddings`

- `POST /v1/embeddings`

### `/v1/feedback`

- `POST /v1/feedback`

### `/v1/finetune`

- `POST /v1/finetune/cancel`
- `POST /v1/finetune/deploy`
- `GET /v1/finetune/hf/datasets`
- `GET /v1/finetune/hf/models`
- `GET /v1/finetune/hf/repo`
- `GET /v1/finetune/job`
- `GET /v1/finetune/jobs`
- `POST /v1/finetune/jobs`
- `GET /v1/finetune/presets`

### `/v1/images`

- `POST /v1/images/generations`

### `/v1/install-patch`

- `POST /v1/install-patch`

### `/v1/memory`

- `POST /v1/memory/delete`
- `GET /v1/memory/facts`
- `GET /v1/memory/list`
- `GET /v1/memory/recall`
- `POST /v1/memory/remember`
- `GET /v1/memory/search`
- `POST /v1/memory/update`

### `/v1/messages`

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

### `/v1/models`

- `GET /v1/models`
- `GET /v1/models/providers`
- `GET /v1/models/{model}/access`
- `POST /v1/models/{model}/access`

### `/v1/org`

- `DELETE /v1/org/settings`
- `GET /v1/org/settings`
- `PATCH /v1/org/settings`
- `POST /v1/org/settings`
- `PUT /v1/org/settings`
- `DELETE /v1/org/settings/list`
- `GET /v1/org/settings/list`
- `PATCH /v1/org/settings/list`
- `POST /v1/org/settings/list`
- `PUT /v1/org/settings/list`

### `/v1/query`

- `POST /v1/query`

### `/v1/query_multiple`

- `POST /v1/query_multiple`

### `/v1/rag`

- `GET /v1/rag/context`
- `POST /v1/rag/delete`
- `POST /v1/rag/embed`
- `POST /v1/rag/query`
- `POST /v1/rag/query-multiple`

### `/v1/rerank`

- `POST /v1/rerank`

### `/v1/responses`

- `POST /v1/responses`

### `/v1/router`

- `DELETE /v1/router/artifact-meta`
- `GET /v1/router/artifact-meta`
- `PATCH /v1/router/artifact-meta`
- `POST /v1/router/artifact-meta`
- `PUT /v1/router/artifact-meta`
- `DELETE /v1/router/data`
- `GET /v1/router/data`
- `DELETE /v1/router/defaults`
- `GET /v1/router/defaults`
- `PATCH /v1/router/defaults`
- `POST /v1/router/defaults`
- `PUT /v1/router/defaults`
- `GET /v1/router/history`
- `GET /v1/router/judge-panel`
- `DELETE /v1/router/ledger`
- `GET /v1/router/ledger`
- `PATCH /v1/router/ledger`
- `POST /v1/router/ledger`
- `PUT /v1/router/ledger`
- `DELETE /v1/router/policy`
- `GET /v1/router/policy`
- `PATCH /v1/router/policy`
- `POST /v1/router/policy`
- `PUT /v1/router/policy`
- `DELETE /v1/router/rewards`
- `GET /v1/router/rewards`
- `PATCH /v1/router/rewards`
- `POST /v1/router/rewards`
- `PUT /v1/router/rewards`
- `GET /v1/router/stats`

### `/v1/traffic`

- `GET /v1/traffic/globe`

### `/v1/videos`

- `POST /v1/videos/generations`
- `GET /v1/videos/{id}`
- `GET /v1/videos/{id}/content`

### `/v1/wecom-bot`

- `GET /v1/wecom-bot/callback/{botId}`
- `POST /v1/wecom-bot/callback/{botId}`

## Example Calls

```bash
TOKEN=$(hanzo auth token)
```

**Returns the list of available models from the routing table.**

```bash
curl -sS "https://api.hanzo.ai/v1/models"
```

**Implements the OpenAI-compatible chat completions API**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/chat/completions"
```

**Implements the Anthropic Messages API.**

```bash
curl -sS -X POST "https://api.hanzo.ai/v1/messages"
```

## Answers

Documented status codes across this product: `200` (1).

18 of these operations are marked public in the document — they are reachable without a bearer token. The rest are not.

## Composition

- `pricing` prices a model before you call it; `usage` and `billing` report what the call cost
- `router` paths inside this product pick a model per request; `plan` decides which are allowed
- `sandbox` runs code the model writes; `tool` lists what the model may call
- `o11y` receives the traces — a model call is the span most worth keeping

## Related Skills

- `hanzo-api/eval.md` — Scoring a model on your own data, with a judge you choose.
- `hanzo-api/benchmark.md` — One honest score for any model, on the tests everyone quotes.
- `hanzo-api/research.md` — Every experiment you have ever run, kept and comparable.
- `hanzo-api/experiment.md` — A/B testing anything: a flag, an ad, a subject line, a model.
- `hanzo-cloud-architecture/SKILL.md` — how one binary serves all of this
- `hanzo-api/INDEX.md` — every Hanzo product, by domain

---

**Last Updated**: 2026-08-20
**Generated from**: https://api.hanzo.ai/v1/openapi.json
