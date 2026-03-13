# Hanzo ZAP - Protocol Bridge Sidecar

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-orm.md`, `hanzo/hanzo-database.md`, `hanzo/hanzo-kv.md`

## Overview

Hanzo ZAP (`hanzoai/zap`) is a **Go sidecar that bridges the ZAP protocol to backend infrastructure services**. It runs as a sidecar container alongside databases and caches, translating ZAP protocol calls from the Hanzo Gateway into native protocol calls against the co-located backend. Supports 4 modes: SQL (PostgreSQL via pgx), KV (Valkey/Redis), Datastore (ClickHouse native TCP), and DocumentDB (MongoDB wire protocol). ZAP's schema maps 1:1 with MCP (Model Context Protocol), so any service implementing the ZAP interface gets MCP tools and resources for free.

### What it actually is

- Single Go binary (`bin/zap`) with mode flag selecting the backend
- 4 backend proxies, each in `internal/`:
  - `sql/` -- PostgreSQL proxy (pgx v5 driver)
  - `kv/` -- Valkey/Redis proxy (hanzoai/kv-go v9)
  - `datastore/` -- ClickHouse proxy (clickhouse-go v2, native TCP)
  - `documentdb/` -- MongoDB/FerretDB proxy (mongo-driver v2)
- MCP tool and resource definitions in `internal/mcp.go`
- ZAP protocol implementation via `luxfi/zap` v0.2.0 (mDNS service discovery)
- Docker image: `ghcr.io/hanzoai/zap:latest`
- Healthcheck on port 9651 (`/health`)
- Runs as a sidecar in the `sql` StatefulSet (not a standalone Deployment)

### ZAP-MCP mapping

ZAP's schema natively maps 1:1 with MCP:
- ZAP tools --> MCP tools (listTools, callTool)
- ZAP resources --> MCP resources (listResources, readResource)
- ZAP prompts --> MCP prompts (listPrompts, getPrompt)

Any service implementing the ZAP interface gets MCP for free via the ZAP Gateway (`zapd`).

## When to use

- Deploying database sidecars in K8s for ZAP/MCP access
- Adding MCP tool support to PostgreSQL, Redis, ClickHouse, or MongoDB
- Building AI agents that need direct database access via MCP
- Bridging Hanzo Gateway to backend infrastructure

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/zap` |
| Module | `github.com/hanzoai/zap-sidecar` |
| Go | 1.26 |
| Branch | `main` |
| Binary | `bin/zap` |
| Modes | `sql`, `kv`, `datastore`, `documentdb` |
| Default port | 9651 |
| Health | `GET /health` (port 9651) |
| Docker image | `ghcr.io/hanzoai/zap:latest` |
| Build | `make build` |
| Test | `make test` |
| Docker build | `make docker` |
| Push | `make push` |
| CI | GitHub Actions (Build and Deploy, Release) |
| License | MIT |

## Project structure

```
hanzoai/zap/
  go.mod                    # github.com/hanzoai/zap-sidecar, Go 1.26
  go.sum
  Makefile                  # build, test, docker, push, clean
  Dockerfile                # Multi-stage, multi-arch (BuildKit)
  .dockerignore
  cmd/
    zap-sidecar/
      main.go               # Entry point, mode switch, signal handling
  internal/
    mcp.go                  # MCP tool/resource definitions for all backends
    sql/
      proxy.go              # PostgreSQL proxy (pgx v5)
    kv/
      proxy.go              # Valkey/Redis proxy (kv-go v9)
    datastore/
      proxy.go              # ClickHouse proxy (clickhouse-go v2)
    documentdb/
      proxy.go              # MongoDB/FerretDB proxy (mongo-driver v2)
```

## Dependencies

From `go.mod`:
- `github.com/luxfi/zap` v0.2.0 -- ZAP protocol + mDNS discovery
- `github.com/jackc/pgx/v5` v5.7.2 -- PostgreSQL driver
- `github.com/hanzoai/kv-go/v9` v9.17.2-hanzo.1 -- Valkey/Redis client
- `github.com/ClickHouse/clickhouse-go/v2` v2.43.0 -- ClickHouse native driver
- `go.mongodb.org/mongo-driver/v2` v2.5.0 -- MongoDB driver

## MCP tools by mode

### SQL mode (`--mode sql`)

| Tool | Description |
|------|------------|
| `sql_query` | Execute read-only SQL SELECT, return JSON rows |
| `sql_exec` | Execute write SQL (INSERT/UPDATE/DELETE), return affected rows |
| `sql_health` | Check PostgreSQL connection health |

Resource: `hanzo://sql/schema` -- database schema (tables, columns, indexes)

### KV mode (`--mode kv`)

| Tool | Description |
|------|------------|
| `kv_get` | Get value by key |
| `kv_set` | Set key-value pair (optional TTL) |
| `kv_mget` | Get multiple values by keys |
| `kv_cmd` | Execute arbitrary Valkey/Redis command |

Resource: `hanzo://kv/info` -- server info and statistics

### Datastore mode (`--mode datastore`)

| Tool | Description |
|------|------------|
| `datastore_query` | Execute ClickHouse SQL query, return JSON rows |
| `datastore_exec` | Execute DDL/non-SELECT statement |
| `datastore_insert` | Bulk insert rows via native batch protocol |
| `datastore_tables` | List tables and metadata |
| `datastore_health` | Check connection health and server version |

Resource: `hanzo://datastore/tables` -- table definitions and schemas

### DocumentDB mode (`--mode documentdb`)

| Tool | Description |
|------|------------|
| `documentdb_find` | Find documents matching a filter |
| `documentdb_insert` | Insert documents into a collection |
| `documentdb_update` | Update documents matching a filter |
| `documentdb_delete` | Delete documents matching a filter |
| `documentdb_health` | Check connection health |

Resource: `hanzo://documentdb/collections` -- collection list and indexes

## Quickstart

### Build and run locally

```bash
git clone https://github.com/hanzoai/zap.git
cd zap
make build

# SQL mode (PostgreSQL sidecar)
./bin/zap --mode sql --backend "postgres://user:pass@localhost:5432/db"

# KV mode (Valkey/Redis sidecar)
./bin/zap --mode kv --backend localhost:6379 --password secret

# Datastore mode (ClickHouse sidecar)
ZAP_USER=default ZAP_DATABASE=default \
  ./bin/zap --mode datastore --backend localhost:9000

# DocumentDB mode (MongoDB/FerretDB sidecar)
ZAP_DATABASE=hanzo \
  ./bin/zap --mode documentdb --backend localhost:27017
```

### Docker

```bash
make docker
docker run -e ZAP_MODE=sql -e ZAP_BACKEND="postgres://..." ghcr.io/hanzoai/zap:latest
```

### K8s sidecar deployment

ZAP runs as a sidecar container in a StatefulSet, not as a standalone Deployment. Example pod spec:

```yaml
containers:
  - name: postgres
    image: postgres:16
  - name: zap
    image: ghcr.io/hanzoai/zap:latest
    args: ["--mode", "sql", "--backend", "localhost:5432"]
    ports:
      - containerPort: 9651
    livenessProbe:
      httpGet:
        path: /health
        port: 9651
```

## Environment variables

| Variable | Description | Default |
|----------|------------|---------|
| `ZAP_MODE` | Backend mode (sql/kv/datastore/documentdb) | required |
| `ZAP_BACKEND` | Backend address (host:port or DSN) | required |
| `ZAP_PASSWORD` | Backend password | empty |
| `ZAP_USER` | Backend username (datastore) | empty |
| `ZAP_DATABASE` | Database name (datastore/documentdb) | empty |

## CLI flags

| Flag | Description | Default |
|------|------------|---------|
| `--mode` | Sidecar mode | `$ZAP_MODE` |
| `--node-id` | ZAP node ID | mode name |
| `--port` | ZAP listen port | 9651 |
| `--service-type` | mDNS service type | `_hanzo._tcp` |
| `--backend` | Backend address | `$ZAP_BACKEND` |
| `--password` | Backend password | `$ZAP_PASSWORD` |

## Troubleshooting

- **"unknown mode" error**: Set `--mode` or `ZAP_MODE` to one of: sql, kv, datastore, documentdb
- **Connection refused**: Verify backend is reachable from sidecar (same pod in K8s = localhost)
- **Health check fails**: Sidecar listens on port 9651, check `wget -qO- http://localhost:9651/health`
- **Auto-deploy disabled**: ZAP is a sidecar in the sql StatefulSet; auto-restarting would bounce the database. Build/push image, deploy manually
- **Docker build fails**: Use `golang:1.26-alpine` (not pinned alpine version)

## Related Skills

- `hanzo/hanzo-orm.md` -- Go ORM that uses ZAP as a backend
- `hanzo/hanzo-database.md` -- Database configuration patterns
- `hanzo/hanzo-sql.md` -- PostgreSQL (hanzoai/sql)
- `hanzo/hanzo-kv.md` -- Valkey/Redis (hanzoai/kv)
- `hanzo/hanzo-datastore.md` -- ClickHouse analytics
- `hanzo/hanzo-mcp.md` -- MCP tools (ZAP provides MCP for free)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: zap, sidecar, protocol-bridge, mcp, postgresql, redis, clickhouse, mongodb
**Prerequisites**: Go 1.26, Docker (for container builds)
