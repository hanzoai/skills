# Hanzo ZAP - Protocol Bridge Sidecar

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-sql.md`, `hanzo/hanzo-kv.md`

## Overview

Hanzo ZAP (`hanzoai/zap`) is a **Go sidecar that bridges the ZAP protocol to backend infrastructure services**. Runs as a sidecar container alongside databases and caches, translating ZAP protocol calls into native protocol calls. Supports 4 modes: SQL (PostgreSQL), KV (Valkey/Redis), Datastore (ClickHouse), and DocumentDB (MongoDB wire protocol). ZAP's schema maps 1:1 with MCP, so any ZAP service gets MCP tools for free.

## When to use

- Adding MCP tool support to PostgreSQL, Redis, ClickHouse, or MongoDB
- Building AI agents that need direct database access via MCP
- Deploying database sidecars in K8s for ZAP/MCP access
- Bridging Hanzo Gateway to backend infrastructure

## Hard requirements

1. **Sidecar pattern**: ZAP runs in the same pod as the database, communicating via localhost
2. **Port 9651**: Default ZAP listen port (same as Lux staking port by convention)
3. **Single mode per instance**: Each sidecar runs one mode (`--mode sql|kv|datastore|documentdb`)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/zap` |
| Module | `github.com/hanzoai/zap-sidecar` |
| Go version | 1.26 |
| Binary | `bin/zap` |
| Modes | `sql`, `kv`, `datastore`, `documentdb` |
| Default port | 9651 |
| Health | `GET /health` (port 9651) |
| Image | `ghcr.io/hanzoai/zap:latest` |
| Build | `make build` |
| Test | `make test` |
| Service type | mDNS: `_hanzo._tcp` |
| K8s manifests | `universe/infra/k8s/sql/` (sidecar in sql StatefulSet) |

## ZAP-MCP mapping

ZAP's schema natively maps 1:1 with MCP:

| ZAP Concept | MCP Equivalent |
|-------------|----------------|
| ZAP tools | `listTools`, `callTool` |
| ZAP resources | `listResources`, `readResource` |
| ZAP prompts | `listPrompts`, `getPrompt` |

Any service implementing the ZAP interface gets MCP for free via the ZAP Gateway (`zapd`).

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
| `datastore_health` | Check connection health |

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

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `github.com/luxfi/zap` | v0.2.0 | ZAP protocol + mDNS discovery |
| `github.com/jackc/pgx/v5` | v5.7.2 | PostgreSQL driver |
| `github.com/hanzoai/kv-go/v9` | v9.17.2-hanzo.1 | Valkey/Redis client |
| `github.com/ClickHouse/clickhouse-go/v2` | v2.43.0 | ClickHouse native driver |
| `go.mongodb.org/mongo-driver/v2` | v2.5.0 | MongoDB driver |

## Quickstart

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

## K8s sidecar deployment

ZAP runs as a sidecar in a StatefulSet, not a standalone Deployment:

```yaml
# In the sql StatefulSet
containers:
  - name: postgres
    image: ghcr.io/hanzoai/sql:latest
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

## Wire format

ZAP is a native TCP binary protocol on port 9651. The wire format is designed for zero-copy operation:

- **Header**: 8 bytes (4-byte length + 4-byte type)
- **Body**: Protobuf-encoded request/response
- **Streaming**: Bidirectional streaming for large result sets
- **Service mesh**: mDNS discovery for automatic sidecar registration

## Environment variables

| Variable | Description | Default |
|----------|------------|---------|
| `ZAP_MODE` | Backend mode | required |
| `ZAP_BACKEND` | Backend address (host:port or DSN) | required |
| `ZAP_PASSWORD` | Backend password | empty |
| `ZAP_USER` | Backend username | empty |
| `ZAP_DATABASE` | Database name | empty |

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

| Issue | Cause | Solution |
|-------|-------|----------|
| "unknown mode" error | Invalid mode | Set `--mode` to: sql, kv, datastore, documentdb |
| Connection refused | Backend unreachable | Same pod = localhost. Check backend container |
| Health check fails | Wrong port | ZAP listens on 9651: `wget -qO- http://localhost:9651/health` |
| Auto-deploy disabled | Sidecar in StatefulSet | Build/push image, deploy manually (bouncing would restart DB) |

## Related Skills

- `hanzo/hanzo-mcp.md` -- MCP tools (ZAP provides MCP for free)
- `hanzo/hanzo-sql.md` -- PostgreSQL (hanzoai/sql)
- `hanzo/hanzo-kv.md` -- Valkey/Redis (hanzoai/kv)
- `hanzo/hanzo-datastore.md` -- ClickHouse analytics
- `hanzo/hanzo-documentdb.md` -- MongoDB wire protocol

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: zap, sidecar, protocol-bridge, mcp, postgresql, redis, clickhouse, mongodb
**Prerequisites**: Go 1.26, Docker
