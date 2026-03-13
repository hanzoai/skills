# Hanzo Logs - Scalable Log Aggregation and Querying

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-metrics.md`, `hanzo/hanzo-o11y.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Logs is a **horizontally-scalable, multi-tenant log aggregation system** for Hanzo infrastructure. Fork of Grafana Loki. Written in Go. Indexes labels (not full text) for cost-effective log storage and querying via LogQL (Prometheus-inspired query language). Integrates natively with Grafana.

### Why Hanzo Logs?

- **Label-based indexing**: Stores compressed unstructured logs, indexes only metadata labels -- cheaper to run than full-text indexing systems
- **Prometheus-native**: Uses the same labels as Prometheus metrics for seamless correlation between metrics and logs
- **Kubernetes-native**: Automatically scrapes and indexes Pod labels
- **Multi-tenant**: Isolated log streams per tenant
- **Grafana integration**: Native datasource in Grafana v6.0+

### Tech Stack

- **Language**: Go (1.25.5)
- **Go module**: `github.com/grafana/loki/v3` (upstream module path, not yet rebranded)
- **Query language**: LogQL
- **Storage backends**: S3, GCS, Azure Blob, filesystem, Bigtable
- **Index backends**: BoltDB, TSDB
- **Streaming**: Kafka integration
- **License**: AGPL-3.0-only (Apache-2.0 exceptions in LICENSING.md)

### OSS Base

Fork of `github.com/grafana/loki`. Repo: `hanzoai/logs`.

## When to use

- Aggregating logs from K8s workloads across Hanzo infrastructure
- Correlating log events with Prometheus/Hanzo Metrics time-series data
- Querying structured and unstructured logs via LogQL
- Running a cost-effective alternative to Elasticsearch/Splunk for log storage
- Operating a Loki-compatible stack for Grafana dashboards

## Hard requirements

1. **Go 1.25.5+** for building from source
2. **Object storage** (S3, GCS, Azure Blob, or local filesystem) for log chunk storage
3. **Grafana v6.0+** for native log querying UI
4. **Alloy or Promtail** agent for shipping logs to the server

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/logs` |
| Language | Go |
| Go Module | `github.com/grafana/loki/v3` |
| Default branch | `main` |
| License | AGPL-3.0-only |
| Default port | 3100 (HTTP), 9096 (gRPC) |

## Binaries

Built from `cmd/`:

| Binary | Purpose |
|--------|---------|
| `loki` | Log aggregation server |
| `logcli` | CLI for querying logs |
| `loki-canary` | End-to-end logging pipeline test tool |
| `lokitool` | Operational tooling (migration, inspection) |
| `querytee` | Query comparison proxy |
| `chunks-inspect` | Chunk data inspector |
| `dataobj-inspect` | Data object inspector |
| `logql-analyzer` | LogQL query analyzer |

## Build commands

```bash
# Build all binaries
make all

# Build individual binaries
make loki
make logcli

# Run tests
make test
go test ./...

# Single package test
go test -v ./pkg/logql/...

# Integration tests
make test-integration

# Linting
make lint

# Formatting
make format
```

## Architecture

```
Write path:   distributor -> ingester -> storage
Read path:    query-frontend -> querier -> storage
```

### Key directories

```
cmd/                # Binary entry points
pkg/                # All library code
  logql/            # LogQL query engine
  ingester/         # Log ingestion
  distributor/      # Write path distribution
  compactor/        # Index compaction
  querier/          # Read path
  storage/          # Storage backends
  chunkenc/         # Chunk encoding
  kafka/            # Kafka integration
  bloomgateway/     # Bloom filter gateway
  engine/           # Query engine
  ui/frontend/      # Vite-based UI (TypeScript)
clients/            # Log shipping clients (Promtail)
  cmd/              # Client binaries
  pkg/              # Client libraries
operator/           # Kubernetes operator (separate Go module)
production/         # Deployment configs
  docker/           # Docker configs
  helm/             # Helm charts
  ksonnet/          # Ksonnet/jsonnet manifests
  nomad/            # Nomad jobs
  terraform/        # Terraform configs
docs/               # Documentation source
integration/        # Integration tests
```

### Kubernetes Operator

The `operator/` directory contains a separate Go module with a full K8s operator for managing Loki deployments, including CRDs, controllers, and OLM bundle.

## Frontend UI

Vite-based TypeScript UI in `pkg/ui/frontend/`:

```bash
cd pkg/ui/frontend
make build   # Build
make dev     # Dev server
make lint    # Lint
make test    # Tests
```

## Deployment

Production configs in `production/`:
- Docker Compose (`production/docker-compose.yaml`)
- Helm charts (`production/helm/`)
- Ksonnet/jsonnet manifests
- Nomad jobs
- Terraform configs

### Quick local run

```bash
# Build and run with local config
go build ./cmd/loki
./loki -config.file=./cmd/loki/loki-local-config.yaml
```

### Multi-tenant local run

```bash
make loki
./loki -config.file=./cmd/loki/loki-local-multi-tenant-config.yaml \
       -runtime-config.file=./cmd/loki/loki-overrides.yaml
```

## Code style

- Standard Go formatting (gofmt/goimports)
- Import order: stdlib, external, then loki packages
- Structured logging via go-kit/log
- Table-driven tests preferred
- Conventional commits: `<type>: message`
- Frontend: TypeScript, functional components, dash-case directories

## Related Skills

- `hanzo/hanzo-metrics.md` - Time-series metrics (VictoriaMetrics fork)
- `hanzo/hanzo-o11y.md` - Observability overview
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: logging, observability, loki, logql, grafana
**Prerequisites**: Go, object storage, Grafana
