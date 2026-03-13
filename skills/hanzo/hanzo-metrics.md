# Hanzo Metrics - High-Performance Time-Series Database

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-logs.md`, `hanzo/hanzo-o11y.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Metrics is a **fast, cost-effective time-series database** for monitoring AI infrastructure. Fork of VictoriaMetrics. Written in Go. Supports PromQL and MetricsQL queries, multiple ingestion protocols, and both single-node and cluster deployment modes. Drop-in Prometheus replacement with 10x better compression and lower memory footprint.

### Why Hanzo Metrics?

- **PromQL compatible**: Use existing Prometheus queries and Grafana dashboards
- **MetricsQL**: Extended query language with additional functions for better performance
- **10x compression**: Stores far more data points per GB than alternatives
- **Low memory**: Handles millions of unique time series with 10x less RAM than InfluxDB, 7x less than Prometheus
- **Multi-protocol ingestion**: Prometheus, InfluxDB, Graphite, OpenTSDB, DataDog, NewRelic, OpenTelemetry, CSV, JSON
- **Multi-tenant**: Isolated tenant data with per-tenant rate limiting
- **Easy operation**: Single binary, no external dependencies, command-line configuration

### Tech Stack

- **Language**: Go (1.26)
- **Go module**: `github.com/VictoriaMetrics/VictoriaMetrics` (upstream module path, not yet rebranded)
- **Query languages**: PromQL, MetricsQL
- **Storage**: Custom TSDB with NFS support (EFS, Filestore)
- **License**: Apache-2.0

### OSS Base

Fork of `github.com/VictoriaMetrics/VictoriaMetrics`. Repo: `hanzoai/metrics`.

## When to use

- Storing and querying Prometheus-compatible metrics from Hanzo infrastructure
- Long-term metrics retention with cost-effective storage
- Drop-in replacement for Prometheus remote storage
- Monitoring AI workloads (GPU utilization, inference latency, model metrics)
- Running a Grafana-backed metrics dashboard
- Stream aggregation as a StatsD alternative

## Hard requirements

1. **Go 1.26+** for building from source
2. **Storage**: Local disk, S3, GCS, or Azure Blob for data persistence

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/metrics` |
| Language | Go |
| Go Module | `github.com/VictoriaMetrics/VictoriaMetrics` |
| Code branch | `master` (main branch has only README stub) |
| Default port | 8428 (single-node HTTP) |
| License | Apache-2.0 |

**IMPORTANT**: The actual code lives on the `master` branch. The `main` branch contains only a README placeholder.

## Components

Built from `app/`:

### Single-Node

| Binary | Purpose |
|--------|---------|
| `victoria-metrics` | All-in-one single-node server |

### Cluster Mode

| Binary | Purpose |
|--------|---------|
| `vminsert` | Write path -- ingestion proxy |
| `vmselect` | Read path -- query proxy |
| `vmstorage` | Storage node |

### Operational Tools

| Binary | Purpose |
|--------|---------|
| `vmagent` | Metrics scraping and forwarding agent |
| `vmalert` | Alerting and recording rules evaluation |
| `vmalert-tool` | Alert testing and validation CLI |
| `vmauth` | Authentication proxy and load balancer |
| `vmbackup` | Backup tool |
| `vmbackupmanager` | Automated backup management |
| `vmctl` | Data migration tool |
| `vmrestore` | Restore from backup |
| `vmgateway` | Rate limiting and access control gateway |
| `vmui` | Web UI for querying and exploration |

### VictoriaLogs (Bundled)

| Binary | Purpose |
|--------|---------|
| `victoria-logs` | Log aggregation engine |
| `vlagent` | Log shipping agent |
| `vlinsert` | Log ingestion |
| `vlselect` | Log querying |
| `vlstorage` | Log storage |
| `vlogscli` | Log query CLI |
| `vlogsgenerator` | Log data generator for testing |

## Build commands

```bash
# Build all
go build ./...

# Run tests
go test ./...

# Using Makefile
make victoria-metrics   # Build single-node
make vminsert vmselect vmstorage  # Build cluster components
```

## Quick start

```bash
# Docker (single-node)
docker run -p 8428:8428 hanzo/metrics

# From source
go build ./app/victoria-metrics
./victoria-metrics
```

## Key directories (master branch)

```
app/                    # Application binaries
  victoria-metrics/     # Single-node server
  vminsert/             # Cluster write proxy
  vmselect/             # Cluster read proxy
  vmstorage/            # Cluster storage
  vmagent/              # Scraping agent
  vmalert/              # Alerting engine
  vmauth/               # Auth proxy
  vmbackup/             # Backup
  vmrestore/            # Restore
  vmctl/                # Migration tool
  vmgateway/            # Rate limiting gateway
  vmui/                 # Web UI
  victoria-logs/        # Bundled log engine
apptest/                # Application-level tests
benchmarks/             # Performance benchmarks
dashboards/             # Grafana dashboards
deployment/             # Deployment configs
docs/                   # Documentation
lib/                    # Core libraries
package/                # Packaging scripts
ports/                  # Protocol-specific port handlers
vendor/                 # Go vendor directory
```

## Ingestion protocols

Hanzo Metrics accepts data via multiple protocols:

- Prometheus remote write and exposition format
- InfluxDB line protocol (HTTP, TCP, UDP)
- Graphite plaintext protocol with tags
- OpenTSDB put and HTTP API
- DataDog agent / DogStatsD
- NewRelic infrastructure agent
- OpenTelemetry metrics
- JSON line format
- CSV data
- Native binary format

## Deployment modes

### Single-node

One binary handles ingestion, storage, and querying. Suitable for up to millions of active time series.

### Cluster

Horizontally scalable with separate `vminsert`, `vmselect`, and `vmstorage` components. Add nodes to scale independently.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Code on main branch is empty | Actual code on `master` | Use `master` branch |
| High memory usage | Too many active time series | Check cardinality, use stream aggregation |
| Slow queries | Large time range or high cardinality | Use MetricsQL optimizations, add `vmselect` nodes |

## Related Skills

- `hanzo/hanzo-logs.md` - Log aggregation (Loki fork)
- `hanzo/hanzo-o11y.md` - Observability overview
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: metrics, monitoring, prometheus, victoriametrics, grafana, time-series
**Prerequisites**: Go, storage backend
