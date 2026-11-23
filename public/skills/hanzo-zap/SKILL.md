# Hanzo ZAP - The Transport (and the DB Protocol Bridge)

**Category**: Hanzo Ecosystem
**Canonical spec**: HIP-0114 (ZAP transport), HIP-0116 (plugin VMs over ZAP)
**Related Skills**: `hanzo-cloud-architecture/SKILL.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-sql.md`, `hanzo/hanzo-kv.md`

## Overview

**ZAP is THE Hanzo transport** (HIP-0114): the zero-copy protocol every Hanzo
service speaks — subsystem↔subsystem inside the `cloud` binary (ZAP-typed Go
interfaces in-process), host↔plugin-VM over the wire (ZAP RPC, HIP-0116), and
service↔database via the bridge below. There is **ZERO gRPC and ZERO protobuf
in Hanzo code**; the only protobuf anywhere is the tiny `zap2pb` tool
(`~/work/zap/zap2pb`, ~30 LOC) at the OTel (OTLP/OpAMP) interop edge —
`pb2zap`/`zap2pb`/`zapc` handle the pb↔zap boundary while services hold only
ZAP types. The `zip` server (`github.com/hanzoai/zip`) serves ZAP as its
primary listener with HTTP as an extra.

This skill also covers `hanzoai/zap`, the **Go bridge that translates ZAP
calls into native database protocols**. It runs alongside databases and
caches in 4 modes: SQL (PostgreSQL), KV (Valkey/Redis), Datastore
(ClickHouse), and DocumentDB (MongoDB wire protocol). ZAP's schema maps 1:1
with MCP, so any ZAP service gets MCP tools for free.

## When to use

- Choosing the transport for ANY Hanzo service-to-service call (answer: ZAP)
- Wiring a plugin VM to the cloud host (ZAP RPC, HIP-0116)
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

ZAP is a native TCP binary protocol (bridge default port 9651; zip's primary
ZAP listener uses 9653). The wire format is designed for zero-copy operation:

- **Header**: 8 bytes (4-byte length + 4-byte type)
- **Body**: ZAP zero-copy encoding — NOT protobuf; messages are read in place
- **Streaming**: Bidirectional streaming for large result sets
- **Service mesh**: mDNS discovery for automatic sidecar registration

### The one pb boundary

Protobuf exists at exactly one edge: OTel interop (OTLP/OpAMP). `zap2pb`
converts outbound ZAP telemetry to OTLP pb; `pb2zap` converts inbound;
`zapc` compiles the schemas. No Hanzo service imports
`google.golang.org/grpc` or `google.golang.org/protobuf` — if you find one,
that is a bug.

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
