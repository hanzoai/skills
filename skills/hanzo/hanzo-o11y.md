# Hanzo O11y - Full-Stack Observability Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo O11y is a **full-stack observability platform** for logs, metrics, and traces. Go 1.25 backend (Gin/gorilla-mux) with a React 18 + Vite frontend. Uses **ClickHouse** as the telemetry datastore and **OpenTelemetry** for ingestion. Fork of SigNoz. Live at `o11y.hanzo.ai`.

### Why Hanzo O11y?

- **Single pane of glass**: Logs, metrics, traces, and LLM observability in one UI
- **OpenTelemetry native**: No vendor lock-in, standard instrumentation
- **ClickHouse backend**: Columnar storage optimized for observability queries (50% less resources than Elastic)
- **Self-hostable**: Docker Compose or Helm, community or enterprise edition
- **LLM Observability**: Track LLM calls, token usage, costs, prompt/response analysis
- **Built-in alerting**: Alert on any telemetry signal (logs, metrics, traces)

### Tech Stack

- **Backend**: Go 1.25, Gin, gorilla/mux, go-redis, cobra CLI
- **Frontend**: React 18, TypeScript, Vite (rolldown-vite), Ant Design, `@hanzo/ui`, `@hanzo/insights`
- **Telemetry Store**: ClickHouse (`ghcr.io/hanzoai/datastore`)
- **Ingestion**: OpenTelemetry Collector fork (`ghcr.io/hanzoai/otel-collector`)
- **Metadata Store**: SQLite (community) or PostgreSQL (enterprise)
- **Auth**: JWT tokens (built-in), OIDC + SAML (enterprise), OpenFGA for authz
- **Tests**: Go (`go test -race ./...`), Frontend (Jest), Integration (Python/uv/pytest)

### OSS Base

Fork of `signoz/signoz`. Repo: `hanzoai/o11y`.

## When to use

- Monitoring applications and infrastructure (APM, errors, latency)
- Centralized log management with full-text search and aggregation
- Distributed tracing across microservices
- LLM observability (token usage, costs, prompt debugging)
- Custom metrics dashboards and alerting
- Replacing Datadog, New Relic, or Elastic with a self-hosted solution

## Hard requirements

1. **ClickHouse** (via `ghcr.io/hanzoai/datastore`) for telemetry storage
2. **ZooKeeper** for ClickHouse coordination (single-node or cluster)
3. **OpenTelemetry Collector** (`ghcr.io/hanzoai/otel-collector`) for data ingestion
4. **Docker** or **Kubernetes** for deployment

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://o11y.hanzo.ai` |
| Docs | `https://o11y.hanzo.ai/docs` |
| API | Port 8080 (backend serves frontend + API) |
| OTel gRPC | Port 4317 |
| OTel HTTP | Port 4318 |
| Prometheus metrics | Port 9090 (self-instrumentation) |
| Enterprise image | `ghcr.io/hanzoai/o11y` |
| Community image | `ghcr.io/hanzoai/o11y-community` |
| OTel Collector image | `ghcr.io/hanzoai/otel-collector` |
| Datastore image | `ghcr.io/hanzoai/datastore:25.5.6` |
| Go module | `github.com/hanzoai/o11y` |
| Frontend package | `@hanzo/o11y` |
| License | MIT (community), proprietary (`ee/` directory) |
| Repo | `github.com/hanzoai/o11y` |

## Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Your Apps   │────>│  OTel Collector   │────>│  ClickHouse  │
│  (OTel SDK)  │     │  (gRPC/HTTP)      │     │  (Datastore) │
└──────────────┘     │  :4317 / :4318    │     └──────┬───────┘
                     └───────────────────┘            │
                                                      │
┌──────────────┐     ┌───────────────────┐            │
│  Frontend    │────>│  O11y Backend     │────────────┘
│  (React/Vite)│     │  (Go, :8080)      │────> SQLite/PostgreSQL
└──────────────┘     └───────────────────┘      (metadata, users, alerts)
```

### Data Flow

1. Applications instrumented with OpenTelemetry SDKs send traces, metrics, and logs to the OTel Collector
2. OTel Collector processes and exports to ClickHouse databases: `observe_traces`, `observe_metrics`, `observe_logs`, `observe_meter`, `observe_metadata`
3. The Go backend queries ClickHouse for telemetry data and serves the React frontend
4. SQLite (or PostgreSQL in enterprise) stores metadata: users, orgs, dashboards, alerts, saved views

### Directory Structure

```
cmd/
  server/          # Community edition entrypoint
  enterprise/      # Enterprise edition entrypoint
pkg/
  alertmanager/    # Alert rule evaluation + notification
  apiserver/       # HTTP API server (Gin/gorilla-mux)
  authn/           # Authentication (JWT)
  authz/           # Authorization
  cache/           # In-memory or Redis cache
  prometheus/      # PromQL query engine
  querier/         # Query layer over ClickHouse
  query-service/   # Legacy query service
  querybuilder/    # Query builder for UI
  sqlstore/        # SQLite/PostgreSQL metadata store
  telemetrylogs/   # Log query handlers
  telemetrymetrics/# Metric query handlers
  telemetrytraces/ # Trace query handlers
  telemetrystore/  # ClickHouse connection management
  web/             # Embedded frontend serving
ee/
  anomaly/         # Anomaly detection (enterprise)
  authn/           # OIDC + SAML SSO (enterprise)
  authz/           # OpenFGA fine-grained authz (enterprise)
  gateway/         # Multi-tenant gateway (enterprise)
  licensing/       # License management
  zeus/            # Cloud/license API client
frontend/
  src/             # React 18 + TypeScript + Vite
deploy/
  docker/          # Docker Compose deployment
  docker-swarm/    # Docker Swarm deployment
  common/          # Shared ClickHouse configs, dashboards
conf/
  example.yaml     # Full configuration reference
tests/
  integration/     # Python integration tests (uv/pytest)
```

## Docker Compose quickstart

```bash
git clone https://github.com/hanzoai/o11y.git && cd o11y/deploy/docker
docker compose -f docker-compose.yaml up -d
```

This starts: ZooKeeper, ClickHouse (datastore), O11y backend, OTel Collector, and a schema migrator.

Access the UI at `http://localhost:8080`.

Send telemetry to `localhost:4317` (gRPC) or `localhost:4318` (HTTP).

## Development

### Backend (Go)

```bash
# Start dependencies (ClickHouse + OTel Collector)
make devenv-up

# Run enterprise server locally
make go-run-enterprise

# Run community server locally
make go-run-community

# Run Go tests
make go-test
```

### Frontend (React)

```bash
cd frontend
yarn install
yarn dev          # Vite dev server
yarn build        # Production build
yarn jest         # Unit tests
```

### Integration Tests (Python)

```bash
cd tests/integration
uv run pytest --basetemp=./tmp/ -vv --capture=no src/
```

### Build Docker Images

```bash
# Community
make docker-build-community

# Enterprise
make docker-build-enterprise
```

## Key Environment Variables

```bash
# Telemetry store (ClickHouse)
HANZO_TELEMETRYSTORE_DATASTORE_DSN=tcp://127.0.0.1:9000
HANZO_TELEMETRYSTORE_DATASTORE_CLUSTER=cluster

# Metadata store
HANZO_SQLSTORE_SQLITE_PATH=o11y.db

# Auth
HANZO_TOKENIZER_JWT_SECRET=secret

# Alertmanager
HANZO_ALERTMANAGER_PROVIDER=observe

# Web frontend
HANZO_WEB_ENABLED=true

# Logging
HANZO_INSTRUMENTATION_LOGS_LEVEL=info

# OTel Collector (separate process)
HANZO_OTEL_COLLECTOR_CLICKHOUSE_DSN=tcp://clickhouse:9000
HANZO_OTEL_COLLECTOR_CLICKHOUSE_CLUSTER=cluster
```

## OTel Collector Pipeline

The bundled OTel Collector config (`deploy/docker/otel-collector-config.yaml`) defines these pipelines:

| Pipeline | Receivers | Exporters |
|----------|-----------|-----------|
| traces | OTLP | clickhousetraces, metadataexporter, observemeter |
| metrics | OTLP | observeclickhousemetrics, metadataexporter, observemeter |
| metrics/prometheus | prometheus scraper | observeclickhousemetrics, metadataexporter, observemeter |
| logs | OTLP | clickhouselogsexporter, metadataexporter, observemeter |

ClickHouse databases: `observe_traces`, `observe_metrics`, `observe_logs`, `observe_meter`, `observe_metadata`.

## Instrumenting Your Application

Send telemetry to the OTel Collector using any OpenTelemetry SDK. Point the OTLP exporter at:

- **gRPC**: `otel-collector:4317`
- **HTTP**: `otel-collector:4318`

All languages supported by OpenTelemetry work: Java, Python, Node.js, Go, .NET, Ruby, Rust, PHP, Elixir, Swift.

## Community vs Enterprise

| Feature | Community | Enterprise |
|---------|-----------|------------|
| Logs, Metrics, Traces | Yes | Yes |
| LLM Observability | Yes | Yes |
| Alerts | Yes | Yes |
| Dashboards | Yes | Yes |
| Exception Monitoring | Yes | Yes |
| SSO (SAML/OIDC) | No | Yes |
| Fine-grained RBAC (OpenFGA) | No | Yes |
| Anomaly Detection | No | Yes |
| Multi-tenant Gateway | No | Yes |
| PostgreSQL metadata store | No | Yes |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No data in UI | OTel Collector not receiving data | Check `localhost:4317` is reachable, verify SDK config |
| ClickHouse OOM | High cardinality queries | Tune `max_execution_time` in conf, add resource limits |
| Slow log queries | Missing indexes | Use the query builder, avoid unbounded time ranges |
| OTel Collector crash loops | Schema migration incomplete | Run migrator: `/o11y-otel-collector migrate sync up` |
| Frontend 502 | Backend not ready | Check `localhost:8080/api/v1/health` |

## Related Skills

- `hanzo/hanzo-console.md` - LLM-specific observability (traces, costs, scores)
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-stack.md` - Local dev stack
- `hanzo/hanzo-platform.md` - PaaS deployment platform

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: observability, monitoring, tracing, logs, metrics, opentelemetry, clickhouse, apm
**Prerequisites**: Docker, OpenTelemetry concepts, ClickHouse basics
