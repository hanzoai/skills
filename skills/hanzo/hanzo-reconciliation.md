# Hanzo Reconciliation - Transaction Auto-Matching Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-numscript.md`, `hanzo/hanzo-billing.md`

## Overview

Hanzo Reconciliation is an **automated transaction matching engine** that compares ledger balances against payment provider data to verify financial consistency and identify discrepancies. It runs as a standalone HTTP service backed by PostgreSQL, with policies that define how and when to reconcile ledger and payment data. Part of the Hanzo treasury stack.

### Why Hanzo Reconciliation?

- **Policy-based matching** -- Define reconciliation rules per payment provider and ledger
- **Drift detection** -- Computes `driftBalances` showing exact differences between ledger and payments
- **Multi-provider** -- Reconcile across Stripe, Adyen, bank feeds, and 50+ payment connectors
- **OpenAPI spec** -- Full OpenAPI 3.0.3 definition for all endpoints
- **OpenTelemetry** -- Built-in tracing (OTLP) and metrics export
- **Auto-migration** -- Database migrations with `--auto-migrate` flag or dedicated `migrate` command

### Tech Stack

- **Language**: Go 1.24
- **HTTP**: go-chi/chi v5
- **ORM**: uptrace/bun (PostgreSQL via pgdialect)
- **DI**: uber/fx
- **CLI**: spf13/cobra
- **Auth**: OAuth2 client credentials (Formance SDK)
- **Observability**: OpenTelemetry (traces + metrics)
- **Testing**: testify + gomock + dockertest (PostgreSQL integration tests)
- **Build**: goreleaser + Earthly

### OSS Base

Repo: `hanzoai/reconciliation` (Formance reconciliation fork). Go module path is `github.com/formancehq/reconciliation` (upstream module name retained).

### Architecture Context

```
hanzo/commerce    Storefront, catalog, orders
       |
hanzo/payments    Payment routing (50+ processors)
       |
hanzo/treasury    Ledger, reconciliation, wallets   <-- this service
       |
lux/treasury      On-chain treasury, MPC/KMS wallets
```

## When to use

- Verifying ledger balances match payment provider records
- Detecting discrepancies between internal ledger and external payment data
- Automating periodic reconciliation across multiple payment providers
- Generating compliance-ready reconciliation audit reports
- Matching transactions at specific points in time for both ledger and payments

## Hard requirements

1. **PostgreSQL** (v12+) for policy and reconciliation result storage
2. **Formance Stack** (ledger + payments services) for data sources
3. **Go 1.24+** for building from source

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/formancehq/reconciliation` |
| Default port | `:8080` |
| API version | `v1` (prefixed in OpenAPI) |
| Default branch | `main` |
| Binary | `reconciliation` |
| Image | `ghcr.io/formancehq/base:22.04` (base) |
| Repo | `github.com/hanzoai/reconciliation` |

## One-file quickstart

### Build and run

```bash
# Build
go build -o ./bin/reconciliation .

# Run with auto-migrate
./bin/reconciliation serve \
  --postgres-uri "postgresql://user:pass@localhost:5432/reconciliation" \
  --stack-url "http://localhost:8080" \
  --stack-client-id "client-id" \
  --stack-client-secret "client-secret" \
  --auto-migrate \
  --listen ":8080"
```

### Create a reconciliation policy

```bash
curl -X POST http://localhost:8080/policies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "stripe-daily",
    "ledgerName": "default",
    "ledgerQuery": {"$match": {"metadata[provider]": "stripe"}},
    "paymentsPoolID": "stripe-pool"
  }'
```

### Trigger reconciliation

```bash
curl -X POST http://localhost:8080/policies/{policyID}/reconciliation \
  -H "Content-Type: application/json" \
  -d '{
    "reconciledAtLedger": "2026-03-13T00:00:00.000Z",
    "reconciledAtPayments": "2026-03-13T00:00:00.000Z"
  }'
```

### Check results

```bash
# List all reconciliation results
curl http://localhost:8080/reconciliations

# Get specific result
curl http://localhost:8080/reconciliations/{reconciliationID}
```

## Core Concepts

### Data Model

```
Policy
  |-- id, name, createdAt
  |-- ledgerName          (which ledger to query)
  |-- ledgerQuery         (filter for ledger transactions)
  |-- paymentsPoolID      (which payment pool to compare against)

Reconciliation
  |-- id, policyID, createdAt
  |-- reconciledAtLedger      (point-in-time for ledger snapshot)
  |-- reconciledAtPayments    (point-in-time for payments snapshot)
  |-- status                  (COMPLETED, FAILED, etc.)
  |-- ledgerBalances          (map[asset]bigint from ledger)
  |-- paymentsBalances        (map[asset]bigint from payments)
  |-- driftBalances           (map[asset]bigint -- the difference)
  |-- error                   (if status is FAILED)
```

### API Endpoints (OpenAPI 3.0.3)

| Method | Path | Purpose | Auth Scope |
|--------|------|---------|------------|
| `GET` | `/_info` | Server version info | `reconciliation:read` |
| `POST` | `/policies` | Create a reconciliation policy | `reconciliation:write` |
| `GET` | `/policies` | List policies (paginated) | `reconciliation:read` |
| `GET` | `/policies/{policyID}` | Get a single policy | `reconciliation:read` |
| `DELETE` | `/policies/{policyID}` | Delete a policy | `reconciliation:write` |
| `POST` | `/policies/{policyID}/reconciliation` | Run reconciliation | `reconciliation:write` |
| `GET` | `/reconciliations` | List reconciliation results | `reconciliation:read` |
| `GET` | `/reconciliations/{reconciliationID}` | Get a single result | `reconciliation:read` |

### Pagination

Cursor-based pagination on list endpoints:

```
GET /policies?pageSize=15&cursor=base64EncodedCursor
```

Response includes `cursor.hasMore`, `cursor.next`, `cursor.previous`.

### CLI Commands

```bash
reconciliation serve      # Start HTTP server
reconciliation migrate    # Run database migrations
reconciliation version    # Print version info
```

### Serve Flags

```
--listen              HTTP listen address (default ":8080")
--stack-url           Formance stack URL (for ledger + payments API)
--stack-client-id     OAuth2 client ID for stack API
--stack-client-secret OAuth2 client secret for stack API
--auto-migrate        Auto-run database migrations on startup
--postgres-uri        PostgreSQL connection string
```

## Directory structure

```
github.com/hanzoai/reconciliation/
    main.go                          # Entrypoint (calls cmd.Execute())
    openapi.yaml                     # OpenAPI 3.0.3 spec (full API definition)
    Justfile                         # Build commands (test, lint, build, release)
    build.Dockerfile                 # Production container image
    scratch.Dockerfile               # Minimal scratch-based image
    Earthfile                        # Earthly build definition
    .goreleaser.yml                  # Cross-platform release config
    cmd/
        root.go                      # CLI root command, service name, flags
        serve.go                     # HTTP server with fx DI, OAuth2 stack client
        migrate.go                   # Database migration command
        version.go                   # Version info command
    internal/
        api/
            router.go                # Chi router setup
            module.go                # fx HTTP module
            policy.go                # Policy CRUD handlers
            reconciliation.go        # Reconciliation run + query handlers
            query.go                 # Query builder helpers
            utils.go                 # HTTP response utilities
            backend/                 # Backend service interface + mock
            service/                 # Business logic layer
            policy_test.go           # Policy handler tests
            reconciliation_test.go   # Reconciliation handler tests
        models/
            policy.go                # Policy domain model
            reconciliation.go        # Reconciliation domain model
        storage/
            store.go                 # Store interface
            module.go                # fx storage module
            policy.go                # Policy DB queries (bun)
            reconciliations.go       # Reconciliation DB queries (bun)
            error.go                 # Storage error types
            ping.go                  # Database health check
            utils.go                 # Query utilities
            migrations/              # SQL migration files
            main_test.go             # Test harness (dockertest PostgreSQL)
            policy_test.go           # Policy storage tests
            reconciliations_test.go  # Reconciliation storage tests
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused on startup | PostgreSQL not reachable | Check `--postgres-uri` and network |
| Migration fails | Schema conflicts | Run `reconciliation migrate` standalone first |
| OAuth2 token error | Wrong stack credentials | Verify `--stack-client-id` and `--stack-client-secret` |
| Empty drift balances | No matching transactions | Check `ledgerQuery` filter matches actual ledger data |
| Module path confusion | Upstream module name | Import as `github.com/formancehq/reconciliation` |

## Related Skills

- `hanzo/hanzo-numscript.md` - DSL for modeling the transactions being reconciled
- `hanzo/hanzo-billing.md` - Billing and subscription management

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: fintech, reconciliation, accounting, payments, ledger, go
**Prerequisites**: PostgreSQL 12+, Go 1.24+, Formance stack (ledger + payments)
