# Hanzo Sentry - Error Tracking & Performance Monitoring

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-o11y.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-id.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Sentry is a **self-hosted Sentry 24.2.0 deployment** for error tracking and performance monitoring across all Hanzo, Lux, Zoo, and Pars services. It is a fork of [getsentry/sentry](https://github.com/getsentry/sentry) with a custom Dockerfile branded for Hanzo, configured for OIDC SSO via hanzo.id, and deployed to the hanzo-k8s cluster.

### Why Hanzo Sentry?

- **Self-hosted**: Full control over data, no SaaS dependency
- **Multi-tenant**: Single instance serving four domains (hanzo, lux, pars, zoo)
- **IAM integrated**: OIDC SSO login via hanzo.id (`app-sentry` client)
- **Secrets via KMS**: All credentials synced from kms.hanzo.ai
- **OTEL instrumented**: Exports traces to the shared OTEL collector
- **Shared infrastructure**: Uses hanzo-sql (PostgreSQL) and shared Redis

### Tech Stack

- **Backend**: Python 3.11 (Django 5.x, Celery, DRF)
- **Frontend**: React + TypeScript (Webpack, Emotion CSS)
- **Queue**: Celery via Redis (or RabbitMQ if configured)
- **Database**: PostgreSQL via `hanzo-sql.hanzo.svc`
- **Cache**: Redis via `redis.hanzo.svc`
- **Image**: `ghcr.io/hanzoai/sentry:latest`
- **Base version**: Sentry 24.2.0 (FSL-1.0-Apache-2.0 license)

### OSS Base

Fork of `getsentry/sentry`. Local clone at `~/work/hanzo/sentry/`. The GitHub repo `hanzoai/sentry` does not yet exist as a public fork -- currently origin points to `getsentry/sentry` and Hanzo customizations live in the `self-hosted/` directory and K8s manifests in `universe/infra/k8s/sentry/`.

## When to use

- Instrumenting Hanzo services with error tracking
- Monitoring performance of API endpoints, workers, and frontends
- Viewing crash reports, stack traces, and breadcrumbs
- Setting up alerting rules for error spikes
- Tracking release health across deployments

## Hard requirements

1. **PostgreSQL** via `hanzo-sql.hanzo.svc` (database `sentry`)
2. **Redis** via `redis.hanzo.svc` (cache, queues, rate limits, buffers)
3. **KMS** at kms.hanzo.ai for secret sync (secret-key, db-password, oidc-client-secret)
4. **IAM** at hanzo.id with `app-sentry` OIDC client for SSO

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://sentry.hanzo.ai` |
| Alt domains | `sentry.lux.network`, `sentry.pars.network`, `sentry.zoo.network` |
| Port | 9000 (web), internal K8s service |
| Image | `ghcr.io/hanzoai/sentry:latest` |
| Base version | Sentry 24.2.0 |
| Language | Python 3.11 + TypeScript |
| K8s namespace | `hanzo` |
| Deployments | `sentry-web` (2 replicas), `sentry-worker` (2 replicas), `sentry-cron` (1 replica) |
| Database | `hanzo-sql.hanzo.svc:5432/sentry` |
| Redis | `redis.hanzo.svc:6379` |
| Auth | OIDC SSO via hanzo.id (`app-sentry`) |
| KMS project | `hanzo-k8s-epiq`, path `/sentry` |
| Local clone | `~/work/hanzo/sentry/` |
| K8s manifests | `~/work/hanzo/universe/infra/k8s/sentry/` |
| Health check | `GET /_health/` |

## One-file quickstart

### Instrument a Python service

```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://<key>@sentry.hanzo.ai/<project-id>",
    traces_sample_rate=0.1,
    environment="production",
    release="my-service@1.0.0",
)
```

### Instrument a Node.js service

```javascript
const Sentry = require("@sentry/node");

Sentry.init({
  dsn: "https://<key>@sentry.hanzo.ai/<project-id>",
  tracesSampleRate: 0.1,
  environment: "production",
  release: "my-service@1.0.0",
});
```

### Instrument a Go service

```go
import "github.com/getsentry/sentry-go"

func main() {
    sentry.Init(sentry.ClientOptions{
        Dsn:              "https://<key>@sentry.hanzo.ai/<project-id>",
        TracesSampleRate: 0.1,
        Environment:      "production",
        Release:          "my-service@1.0.0",
    })
    defer sentry.Flush(2 * time.Second)
}
```

### Docker Compose (local development)

```yaml
# compose.yml
services:
  sentry-web:
    image: ghcr.io/hanzoai/sentry:latest
    command: ["sentry", "run", "web"]
    ports:
      - "9000:9000"
    environment:
      SENTRY_SECRET_KEY: "${SENTRY_SECRET_KEY}"
      SENTRY_POSTGRES_HOST: postgres
      SENTRY_DB_NAME: sentry
      SENTRY_DB_USER: sentry
      SENTRY_DB_PASSWORD: "${SENTRY_DB_PASSWORD}"
      SENTRY_REDIS_HOST: redis

  sentry-worker:
    image: ghcr.io/hanzoai/sentry:latest
    command: ["sentry", "run", "worker"]
    environment:
      SENTRY_SECRET_KEY: "${SENTRY_SECRET_KEY}"
      SENTRY_POSTGRES_HOST: postgres
      SENTRY_DB_NAME: sentry
      SENTRY_DB_USER: sentry
      SENTRY_DB_PASSWORD: "${SENTRY_DB_PASSWORD}"
      SENTRY_REDIS_HOST: redis

  sentry-cron:
    image: ghcr.io/hanzoai/sentry:latest
    command: ["sentry", "run", "cron"]
    environment:
      SENTRY_SECRET_KEY: "${SENTRY_SECRET_KEY}"
      SENTRY_POSTGRES_HOST: postgres
      SENTRY_DB_NAME: sentry
      SENTRY_DB_USER: sentry
      SENTRY_DB_PASSWORD: "${SENTRY_DB_PASSWORD}"
      SENTRY_REDIS_HOST: redis

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: sentry
      POSTGRES_USER: sentry
      POSTGRES_PASSWORD: "${SENTRY_DB_PASSWORD}"

  redis:
    image: redis:7-alpine
```

## Core Concepts

### Architecture

```
                  ┌──────────────────────────────────┐
                  │         K8s Ingress               │
                  │  sentry.hanzo.ai                  │
                  │  sentry.lux.network               │
                  │  sentry.pars.network              │
                  │  sentry.zoo.network               │
                  └───────────────┬──────────────────┘
                                  │
                  ┌───────────────▼──────────────────┐
                  │   sentry-web (2 replicas)        │
                  │   Django + uWSGI on port 9000    │
                  └──┬────────────┬─────────────┬────┘
                     │            │             │
              ┌──────▼──┐  ┌─────▼─────┐  ┌───▼────────┐
              │ Postgres │  │   Redis   │  │ OTEL       │
              │ hanzo-sql│  │  (shared) │  │ Collector  │
              └─────────┘  └───────────┘  └────────────┘
                     │            │
              ┌──────▼──┐  ┌─────▼──────────────┐
              │ sentry- │  │ sentry-worker       │
              │ cron    │  │ (2 replicas, Celery) │
              └─────────┘  └─────────────────────┘
```

### Three Deployments

| Component | Command | Replicas | Purpose |
|-----------|---------|----------|---------|
| `sentry-web` | `sentry run web` | 2 | Django web server (uWSGI) |
| `sentry-worker` | `sentry run worker` | 2 | Celery async task processing |
| `sentry-cron` | `sentry run cron` | 1 | Scheduled tasks (digests, cleanup) |

### Multi-Tenant Domains

A single Sentry instance serves four domains via K8s Ingress. Each domain maps to a separate Sentry organization, allowing per-org project isolation:

- `sentry.hanzo.ai` -- Hanzo AI services
- `sentry.lux.network` -- Lux blockchain nodes
- `sentry.pars.network` -- Pars network services
- `sentry.zoo.network` -- Zoo Labs experiments

### Auth Flow

SSO is via OIDC against hanzo.id:
1. User visits `sentry.hanzo.ai` and clicks "SSO Login"
2. Redirected to `hanzo.id` for authentication (client `app-sentry`)
3. OIDC callback returns to Sentry with identity token
4. Sentry creates/links user account and grants org membership

### Dockerfile Customizations

The `self-hosted/Dockerfile` in the local clone contains Hanzo-specific changes:
- OCI labels with `ops@hanzo.ai` maintainer and Hanzo branding
- References `ghcr.io/hanzoai/sentry` as image source
- Copies `self-hosted/sentry.conf.py` and `self-hosted/config.yml` into `/etc/sentry/`
- Entrypoint via `tini` + `gosu` for proper signal handling

### KMS Secret Sync

Secrets are managed via the `KMSSecret` CRD that syncs from `kms.hanzo.ai`:

```yaml
# kms-secrets.yaml
spec:
  hostAPI: https://kms.hanzo.ai/api
  managedSecretReference:
    secretName: sentry-secrets
    secretNamespace: hanzo
  authentication:
    universalAuth:
      secretsScope:
        projectSlug: hanzo-k8s-epiq
        envSlug: prod
        secretsPath: /sentry
```

Required secrets: `secret-key`, `db-password`, `oidc-client-secret`, `commerce-api-key` (optional).

### Environment Variables

```bash
# Database (PostgreSQL)
SENTRY_POSTGRES_HOST=hanzo-sql.hanzo.svc
SENTRY_DB_NAME=sentry
SENTRY_DB_USER=sentry
SENTRY_DB_PASSWORD=<from KMS>
SENTRY_DB_PORT=5432

# Redis
SENTRY_REDIS_HOST=redis.hanzo.svc
SENTRY_REDIS_PORT=6379

# Core
SENTRY_SECRET_KEY=<from KMS>

# OIDC SSO
SENTRY_OIDC_ENABLED=true
SENTRY_OIDC_ISSUER=https://hanzo.id
SENTRY_OIDC_CLIENT_ID=app-sentry
SENTRY_OIDC_CLIENT_SECRET=<from KMS>

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.hanzo.svc:4318
OTEL_SERVICE_NAME=sentry

# Optional: Commerce billing
HANZO_COMMERCE_URL=https://commerce.hanzo.ai
HANZO_COMMERCE_API_KEY=<from KMS>
```

### Local Development

```bash
cd ~/work/hanzo/sentry

# Python setup (uses direnv + pyenv, not uv — upstream pattern)
make develop

# Run dependent services (Postgres, Redis, Kafka, etc.)
make run-dependent-services

# Apply database migrations
make apply-migrations

# Start web server
sentry run web

# In another terminal: start worker
sentry run worker

# Run Python tests
make test-python-ci

# Run JavaScript tests
make test-js
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| OIDC login fails | `app-sentry` not configured in IAM | Create OIDC client at hanzo.id with redirect URI `https://sentry.hanzo.ai/auth/sso/` |
| Redis connection error | Missing `SENTRY_REDIS_HOST` | Ensure `redis.hanzo.svc` is reachable in cluster |
| `SENTRY_SECRET_KEY is undefined` | KMS sync not running | Verify `sentry-secrets` K8s secret exists in `hanzo` namespace |
| Migration errors | DB not created | Run `sentry upgrade` to create tables and apply migrations |
| Worker not processing | Celery not connected to broker | Check Redis connectivity or RabbitMQ config |
| Health check failing | Sentry not fully started | Wait for `initialDelaySeconds` (60s); check logs for startup errors |
| `npm ci` fails | Peer dep conflicts | Use `npm install` with `.npmrc` containing `legacy-peer-deps=true` |

## Related Skills

- `hanzo/hanzo-o11y.md` - Observability stack (OTEL, Prometheus, Grafana)
- `hanzo/hanzo-id.md` - IAM and OIDC SSO configuration
- `hanzo/hanzo-kms.md` - Secret management via KMSSecret CRDs
- `hanzo/hanzo-universe.md` - Production K8s infrastructure and manifests
- `hanzo/hanzo-database.md` - Shared PostgreSQL (hanzo-sql)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: sentry, error-tracking, monitoring, observability, self-hosted
**Prerequisites**: Kubernetes, PostgreSQL, Redis, OIDC basics
