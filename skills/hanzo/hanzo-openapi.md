# Hanzo OpenAPI - Unified API Specification for 26 Cloud Services

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-gateway.md`, `hanzo/hanzo-cloud.md`, `hanzo/python-sdk.md`, `hanzo/js-sdk.md`, `hanzo/go-sdk.md`

## Overview

Hanzo OpenAPI is the **canonical OpenAPI 3.1.0 specification** for all 26 Hanzo Cloud services. One master spec (`hanzo.yaml`) plus per-service specs, shared schemas, and everything needed to generate SDKs, validate APIs, and render documentation. Version 4.0.0.

### Why Hanzo OpenAPI?

- **Single source of truth**: Every Hanzo service endpoint, schema, and error is defined here
- **SDK generation**: Feed any spec to `openapi-generator` to produce Python, TypeScript, Go, Rust clients
- **Documentation**: Render with Redoc, Scalar, or Swagger UI
- **Validation**: Lint specs with `@redocly/cli` before deploy
- **Gateway routing**: Defines the contract that `api.hanzo.ai` enforces

### Spec Format

- **OpenAPI 3.1.0** YAML files
- Shared components in `shared/` (auth, errors, pagination schemas)
- Master unified spec: `hanzo.yaml` (33KB, all 26 services combined)
- Per-service specs: `<service>/openapi.yaml`

### OSS Base

Repo: `hanzoai/openapi` (1.8MB, 3 stars).

## When to use

- Adding or modifying any Hanzo API endpoint
- Generating a new SDK client for a language
- Validating API contracts before deployment
- Rendering API documentation (docs.hanzo.ai)
- Understanding the full surface area of Hanzo Cloud

## Hard requirements

1. **OpenAPI 3.1.0** compliance for all specs
2. **Shared schemas** must be referenced from `shared/`, not duplicated
3. **All services** must appear in the master `hanzo.yaml`
4. **Auth** uses `Authorization: Bearer hk-...` (API key) or OAuth2 JWT

## Quick reference

| Item | Value |
|------|-------|
| Master Spec | `hanzo.yaml` (33KB) |
| Spec Version | OpenAPI 3.1.0 |
| API Version | 4.0.0 |
| Gateway | `api.hanzo.ai` |
| Docs | `docs.hanzo.ai` |
| Repo | `github.com/hanzoai/openapi` |
| Branch | `main` |

## Services Covered (26)

### AI and Intelligence
| Service | Spec | Endpoint |
|---------|------|----------|
| Cloud | `cloud/openapi.yaml` (116KB) | `api.cloud.hanzo.ai` |
| Chat | `chat/openapi.yaml` | `hanzo.chat` |
| Search | `search/openapi.yaml` | `search.hanzo.ai` |
| Bot | `bot/openapi.yaml` | `app.hanzo.bot` |
| Nexus | `nexus/openapi.yaml` | `nexus.hanzo.ai` |
| Vector | `vector/openapi.yaml` | `vector.hanzo.ai` |

### Automation
| Service | Spec | Endpoint |
|---------|------|----------|
| Flow | `flow/openapi.yaml` | `flow.hanzo.ai` |
| Auto | `auto/openapi.yaml` | `auto.hanzo.ai` |
| Operative | `operative/openapi.yaml` | `operative.hanzo.ai` |

### Platform and Identity
| Service | Spec | Endpoint |
|---------|------|----------|
| IAM | `iam/openapi.yaml` | `hanzo.id` |
| Commerce | `commerce/openapi.yaml` | `commerce.hanzo.ai` |
| Gateway | `gateway/openapi.yaml` | `api.hanzo.ai` |
| Console | `console/openapi.yaml` | `console.hanzo.ai` |
| KMS | `kms/openapi.yaml` | `kms.hanzo.ai` |
| Analytics | `analytics/openapi.yaml` | `analytics.hanzo.ai` |

### Infrastructure
| Service | Spec | Endpoint |
|---------|------|----------|
| PaaS | `paas/openapi.yaml` | `paas.hanzo.ai` |
| Platform | `platform/openapi.yaml` | `platform.hanzo.ai` |
| DB | `db/openapi.yaml` | `db.hanzo.ai` |
| KV | `kv/openapi.yaml` | `kv.hanzo.ai` |
| MQ | `mq/openapi.yaml` | `mq.hanzo.ai` |
| Edge | `edge/openapi.yaml` | `edge.hanzo.ai` |
| Registry | `registry/openapi.yaml` | `registry.hanzo.ai` |
| Visor | `visor/openapi.yaml` | `vm.hanzo.ai` |

### Operations
| Service | Spec | Endpoint |
|---------|------|----------|
| Engine | `engine/openapi.yaml` | `engine.hanzo.ai` |
| O11y | `o11y/openapi.yaml` | `o11y.hanzo.ai` |
| DNS | `dns/openapi.yaml` | `dns.hanzo.ai` |
| ZT | `zt/openapi.yaml` | `zt.hanzo.ai` |

## Repository Structure

```
openapi/
  hanzo.yaml                # Master unified spec (all 26 services)
  LLM.md
  README.md
  shared/
    auth.yaml               # Auth security schemes (API key, OAuth2, JWT)
    errors.yaml             # Shared error response schemas
    pagination.yaml         # Cursor/offset pagination schemas
  analytics/openapi.yaml
  auto/openapi.yaml
  bot/openapi.yaml
  chat/openapi.yaml
  cloud/openapi.yaml        # Largest spec (116KB) -- LLM inference API
  commerce/openapi.yaml
  console/openapi.yaml
  db/openapi.yaml
  did/openapi.yaml
  dns/openapi.yaml
  edge/openapi.yaml
  engine/openapi.yaml
  flow/openapi.yaml
  gateway/openapi.yaml
  guard/openapi.yaml
  iam/openapi.yaml
  kms/openapi.yaml
  kv/openapi.yaml
  ml/openapi.yaml
  mq/openapi.yaml
  nexus/openapi.yaml
  o11y/openapi.yaml
  operative/openapi.yaml
  paas/openapi.yaml
  platform/openapi.yaml
  pricing/openapi.yaml
  pubsub/openapi.yaml
  registry/openapi.yaml
  s3/openapi.yaml
  search/openapi.yaml
  stream/openapi.yaml
  vector/openapi.yaml
  visor/openapi.yaml
  zt/openapi.yaml
```

## Gateway Routes

All services accessible through `api.hanzo.ai`:

```
api.hanzo.ai/v1/chat/completions   -> Cloud (LLM inference)
api.hanzo.ai/v1/models             -> Cloud (model listing)
api.hanzo.ai/v1/chat/*             -> Chat
api.hanzo.ai/v1/search/*           -> Search
api.hanzo.ai/v1/bot/*              -> Bot
api.hanzo.ai/v1/nexus/*            -> Nexus
api.hanzo.ai/v1/vector/*           -> Vector DB
api.hanzo.ai/v1/flow/*             -> Flow
api.hanzo.ai/v1/operative/*        -> Operative
api.hanzo.ai/v1/auth/*             -> IAM
api.hanzo.ai/v1/billing/*          -> Commerce
api.hanzo.ai/v1/kms/*              -> KMS
api.hanzo.ai/v1/console/*          -> Console
api.hanzo.ai/v1/analytics/*        -> Analytics
api.hanzo.ai/v1/paas/*             -> PaaS
api.hanzo.ai/v1/platform/*         -> Platform
api.hanzo.ai/v1/db/*               -> DB
api.hanzo.ai/v1/kv/*               -> KV
api.hanzo.ai/v1/mq/*               -> MQ
api.hanzo.ai/v1/edge/*             -> Edge
api.hanzo.ai/v1/registry/*         -> Registry
api.hanzo.ai/v1/vm/*               -> Visor
api.hanzo.ai/v1/engine/*           -> Engine
api.hanzo.ai/v1/o11y/*             -> O11y
api.hanzo.ai/v1/dns/*              -> DNS
api.hanzo.ai/v1/zt/*               -> ZT
```

## Authentication

All services use a single `HANZO_API_KEY`:

```bash
curl -H "Authorization: Bearer hk-your-api-key" https://api.hanzo.ai/v1/models
```

| Method | Use Case |
|--------|----------|
| API Key | Server-to-server (`Authorization: Bearer hk-...`) |
| OAuth2 + PKCE | User-facing apps (`hanzo.id/oauth/authorize`) |
| Client Credentials | Service-to-service (`POST hanzo.id/oauth/token`) |
| JWT | After OAuth login (`Authorization: Bearer eyJ...`) |

## Common Operations

### Validate All Specs

```bash
for spec in */openapi.yaml; do
  echo "Validating $spec..."
  openapi-generator validate -i "$spec"
done

# Master spec
npx @redocly/cli lint hanzo.yaml
```

### Generate SDK from Spec

```bash
# Python
openapi-generator generate -i hanzo.yaml -g python -o clients/python

# TypeScript
openapi-generator generate -i hanzo.yaml -g typescript-axios -o clients/typescript

# Go
openapi-generator generate -i hanzo.yaml -g go -o clients/go

# Rust
openapi-generator generate -i hanzo.yaml -g rust -o clients/rust
```

### Serve Docs Locally

```bash
npx @redocly/cli preview-docs hanzo.yaml           # Redoc
npx @scalar/cli serve hanzo.yaml                    # Scalar
docker run -p 8080:8080 -e SWAGGER_JSON=/specs/hanzo.yaml \
  -v $(pwd):/specs swaggerapi/swagger-ui             # Swagger UI
```

## Open Source Orgs

| GitHub Org | Focus |
|------------|-------|
| @hanzoai | Core platform (all 26 services) |
| @hanzozt | Zero-trust networking |
| @hanzo-ml | GPU/ML (kubeflow, kuberay) |
| @hanzodns | DNS (coredns) |
| @hanzofn | Edge functions (edge-runtime) |
| @hanzosql | Databases (neon) |
| @hanzokv | Key-value store (valkey) |
| @hanzocr | Container registry (harbor) |
| @hanzomsg | Messaging (nats-server, nats.go, nats.js, nats.py) |
| @hanzoo11y | Observability (loki, grafana, signoz) |

## Related Skills

- `hanzo/hanzo-gateway.md` - API gateway that enforces these specs
- `hanzo/hanzo-cloud.md` - LLM inference (largest spec)
- `hanzo/python-sdk.md` - Python SDK (generated from these specs)
- `hanzo/js-sdk.md` - JavaScript SDK
- `hanzo/go-sdk.md` - Go SDK
- `hanzo/rust-sdk.md` - Rust SDK
- `hanzo/hanzo-sdk.md` - Unified multi-language CLI SDK

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: openapi, api-specification, sdk-generation, documentation
**Prerequisites**: OpenAPI 3.1.0 familiarity, REST API basics
