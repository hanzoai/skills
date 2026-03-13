# Hanzo Ledger - Programmable Double-Entry Financial Ledger

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-vault.md`, `hanzo/hanzo-commerce-api.md`, `hanzo/hanzo-commerce.md`

## Overview

Hanzo Ledger is a **programmable double-entry financial ledger** written in Go. It provides atomic multi-posting transactions, account-based modeling, and a scriptable DSL called [Numscript](https://github.com/hanzoai/numscript) for expressing complex money movements. Fork of Formance Ledger (Go module still `github.com/formancehq/ledger`).

### Why Hanzo Ledger?

- **Double-Entry** -- Every transaction balances. No money created or destroyed.
- **Multi-Posting** -- Atomic transactions across unlimited accounts in a single operation
- **Numscript DSL** -- Model complex splits, fees, and routing with a purpose-built language
- **Immutable Audit Trail** -- Append-only ledger with full transaction history
- **Multi-Currency** -- Native support for any asset type with arbitrary precision (`big.Int`)
- **Multi-Ledger** -- Run multiple isolated ledgers from a single instance
- **Chart of Accounts** -- Schema enforcement with variable segments and regex patterns
- **Replication** -- Built-in data export pipelines with gRPC worker support

### Architecture

```
hanzo/commerce    Storefront, catalog, orders
       |
hanzo/payments    Payment routing (50+ processors)
       |
hanzo/treasury    Ledger, reconciliation, wallets   <-- this
       |
lux/treasury      On-chain treasury, MPC/KMS wallets
```

## When to use

- Recording financial transactions with strict double-entry guarantees
- Multi-party fee splits and revenue sharing
- E-commerce order payments, refunds, and wallet balances
- Internal transfer accounting between services
- Multi-currency or multi-asset tracking (fiat, crypto, credits, points)
- Audit-compliant financial record keeping

## Hard requirements

1. **PostgreSQL** (16+ recommended) for transaction storage
2. **Go 1.24+** for building from source
3. **Docker** for local development (compose includes Postgres, Prometheus, Jaeger, OTEL collector)

## Quick reference

| Item | Value |
|------|-------|
| Language | Go |
| Go Module | `github.com/formancehq/ledger` |
| API | HTTP REST (chi router) |
| Port | 3068 |
| Database | PostgreSQL (via uptrace/bun ORM) |
| API Versions | v1 (legacy), v2 (current), v3 (OpenAPI spec exists) |
| CLI Commands | `serve`, `worker`, `buckets upgrade`, `version`, `docs` |
| Event Bus | Watermill (NATS, Kafka, HTTP) |
| Observability | OpenTelemetry (traces + metrics), Prometheus, Jaeger |
| Repo | `github.com/hanzoai/ledger` |
| License | MIT |

## API Reference

### Create a Transaction

```http
POST /v2/ledger/{name}/transactions
Content-Type: application/json

{
  "postings": [{
    "source": "world",
    "destination": "users:001",
    "amount": 10000,
    "asset": "USD/2"
  }]
}
```

### Create Transaction with Numscript

```http
POST /v2/ledger/{name}/transactions
Content-Type: application/json

{
  "script": {
    "plain": "send [USD/2 10000] (\n  source = @users:001\n  destination = {\n    90% to @merchants:042\n    10% to @platform:fees\n  }\n)"
  }
}
```

### List Transactions

```http
GET /v2/ledger/{name}/transactions
```

### Get Account Balance

```http
GET /v2/ledger/{name}/accounts/{address}
```

### Set Account Metadata

```http
POST /v2/ledger/{name}/accounts/{address}/metadata
Content-Type: application/json

{
  "role": "merchant",
  "tier": "gold"
}
```

### Revert a Transaction

```http
POST /v2/ledger/{name}/transactions/{id}/revert
```

### Bulk Operations

```http
POST /v2/ledger/{name}/bulk
Content-Type: application/json

[
  {"action": "CREATE_TRANSACTION", "data": {"postings": [...]}},
  {"action": "ADD_METADATA", "data": {"targetType": "ACCOUNT", "targetID": "users:001", "metadata": {...}}}
]
```

### Health and Info

```http
GET /_healthcheck
GET /_/info
GET /_/metrics
```

## Numscript DSL

Numscript is a purpose-built language for expressing money movements.

```numscript
// Simple transfer
send [USD/2 10000] (
  source = @users:001
  destination = @merchants:042
)

// Multi-party fee split
send [USD/2 10000] (
  source = @users:001
  destination = {
    90% to @merchants:042
    10% to {
      50% to @platform:fees
      50% to @platform:reserve
    }
  }
)
```

- `@world` is a special unlimited source (money creation)
- Assets use format `CURRENCY/precision` (e.g., `USD/2` = cents)
- Parsed via ANTLR4 grammar, executed by internal machine

## Core Concepts

### Accounts

Hierarchical addresses separated by colons: `users:001`, `merchants:042:fees`. The special account `world` is an unlimited source.

### Postings

Each posting moves an amount of an asset from source to destination:

```go
type Posting struct {
    Source      string   // source account address
    Destination string   // destination account address
    Amount      *big.Int // amount (arbitrary precision)
    Asset       string   // asset identifier (e.g., "USD/2")
}
```

### Volumes

Every account tracks input/output volumes per asset. Balance = Input - Output.

### Ledgers and Buckets

Multiple ledgers can share a single database (bucket). Each ledger has independent accounts, transactions, and metadata. Bucket names and ledger names must match `^[0-9a-zA-Z_-]{1,63}$`.

### Chart of Accounts

Optional schema enforcement. Define allowed account structures with fixed segments, variable segments (`$variable`), and regex patterns.

### Features

Per-ledger configurable features:

| Feature | Options | Default |
|---------|---------|---------|
| `MOVES_HISTORY` | ON, OFF | ON |
| `MOVES_HISTORY_POST_COMMIT_EFFECTIVE_VOLUMES` | SYNC, DISABLED | SYNC |
| `HASH_LOGS` | SYNC, ASYNC, DISABLED | SYNC |
| `ACCOUNT_METADATA_HISTORY` | SYNC, DISABLED | SYNC |
| `TRANSACTION_METADATA_HISTORY` | SYNC, DISABLED | SYNC |

## Development

### Build and Test

```bash
go build ./...
go test ./...

# With integration tests
go test -tags it ./...

# Full test with coverage
just tests
```

### Local Development with Docker

```bash
docker compose up -d
# Ledger available at http://localhost:3068
```

Docker compose includes: PostgreSQL 16, Prometheus, Jaeger, OTEL collector, ledger server, and worker.

### Task Runner (Justfile)

```bash
just              # list available commands
just lint         # golangci-lint
just tidy         # go mod tidy
just generate     # go generate
just tests        # full test suite with coverage
just openapi      # regenerate OpenAPI spec
just generate-client  # regenerate Go client SDK (requires SPEAKEASY_API_KEY)
just pre-commit   # tidy + generate + lint + openapi (run before commits)
```

### Project Structure

```
ledger/
  cmd/              CLI commands (serve, worker, buckets, docs, version)
  internal/
    api/            HTTP API (v1, v2 routers, bulking)
    bus/            Event publishing (Watermill)
    controller/     Business logic (ledger + system controllers)
    machine/        Numscript execution engine
    queries/        Query building
    replication/    Data export pipelines (gRPC, drivers)
    storage/        PostgreSQL storage layer (bun ORM)
    tracing/        OpenTelemetry instrumentation
  pkg/
    accounts/       Account address validation
    assets/         Asset format validation
    client/         Generated Go client SDK (Speakeasy)
    events/         Event type definitions
    features/       Feature flag definitions
  openapi/          OpenAPI specs (v1, v2, v3)
  deployments/      Docker Compose + Pulumi configs
  examples/         Example setups (standalone, publisher-http, publisher-kafka)
  docs/             Generated documentation
  tools/            Build tooling
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_URI` | PostgreSQL connection string | required |
| `POSTGRES_MAX_OPEN_CONNS` | Max open connections | -- |
| `POSTGRES_MAX_IDLE_CONNS` | Max idle connections | -- |
| `POSTGRES_CONN_MAX_IDLE_TIME` | Connection max idle time | -- |
| `AUTO_UPGRADE` | Auto-upgrade schemas on start | false |
| `EXPERIMENTAL_FEATURES` | Enable feature configurability | false |
| `BULK_PARALLEL` | Bulk operation parallelism | 10 |
| `DEBUG` | Enable debug mode | false |

### CLI Flags (serve command)

| Flag | Description | Default |
|------|-------------|---------|
| `--bind` | API bind address | `0.0.0.0:3068` |
| `--auto-upgrade` | Auto-upgrade schemas | false |
| `--numscript-cache-max-count` | Numscript cache size | 1024 |
| `--bulk-max-size` | Max bulk operation size | 100 |
| `--bulk-parallel` | Bulk parallelism | 10 |
| `--default-page-size` | Default pagination size | 15 |
| `--max-page-size` | Max pagination size | 100 |
| `--worker` | Enable embedded worker | false |
| `--schema-enforcement-mode` | Schema enforcement (`audit`, `strict`) | audit |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `github.com/formancehq/numscript` | Numscript DSL parser/interpreter |
| `github.com/formancehq/go-libs/v4` | Shared library (auth, ORM, health, OTLP, publishing) |
| `github.com/go-chi/chi/v5` | HTTP router |
| `github.com/uptrace/bun` | PostgreSQL ORM |
| `github.com/jackc/pgx/v5` | PostgreSQL driver |
| `github.com/ThreeDotsLabs/watermill` | Event bus (NATS, Kafka, HTTP) |
| `github.com/spf13/cobra` | CLI framework |
| `go.uber.org/fx` | Dependency injection |
| `go.opentelemetry.io/otel` | Observability |
| `github.com/antlr/antlr4` | ANTLR parser runtime (Numscript grammar) |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused on 3068 | Server not running | `docker compose up -d` or `go run main.go serve` |
| Schema not found | Ledger not created or not upgraded | Set `AUTO_UPGRADE=true` or run `ledger buckets upgrade` |
| Insufficient funds | Source account balance too low | Use `@world` as source or fund the account first |
| Invalid posting | Negative amount or invalid account address | Amounts must be >= 0, addresses match `^[a-zA-Z0-9_:]+$` |
| Duplicate reference | Transaction reference already used | Use a unique `reference` field or omit it |
| Numscript parse error | Invalid Numscript syntax | Check DSL syntax, ensure assets use `ASSET/precision` format |

## Related Skills

- `hanzo/hanzo-vault.md` - Card tokenization (PCI-compliant, used alongside ledger for payment flows)
- `hanzo/hanzo-commerce-api.md` - Commerce API (uses ledger for order accounting)
- `hanzo/hanzo-commerce.md` - Commerce storefront (drives transactions into ledger)
- `hanzo/hanzo-database.md` - PostgreSQL infrastructure (ledger's storage backend)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ledger, accounting, double-entry, numscript, financial, transactions, go
**Prerequisites**: Double-entry accounting concepts, Go, PostgreSQL, Docker
