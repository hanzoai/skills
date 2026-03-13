# Hanzo Treasury - Programmable Financial Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-payments.md`, `hanzo/hanzo-commerce.md`, `hanzo/hanzo-ledger.md`

## Overview

Hanzo Treasury is a **programmable financial infrastructure** stack providing a double-entry ledger, payment connectors, virtual wallets, reconciliation, and workflow orchestration. Go monorepo (forked from Formance Stack). Uses Earthly for builds. License: MIT.

### Why Hanzo Treasury?

- **Double-entry ledger**: Immutable, programmable source of truth for all financial transactions
- **Numscript DSL**: Domain-specific language for modeling complex monetary flows
- **50+ payment connectors**: Unified API for payment processing across providers
- **Virtual wallets**: Multi-currency with hold/release mechanics
- **Reconciliation**: Auto-match ledger entries against payment provider data
- **Workflow orchestration**: Composable payment and treasury operation flows
- **Event-driven**: JSON Schema-validated events across all services

### Tech Stack

- **Language**: Go (module: `github.com/formancehq/stack`, Go 1.22)
- **Build**: Earthly (v0.8)
- **Events**: Go + Node.js (JSON Schema validation with `gojsonschema`)
- **API**: OpenAPI 3.0.3 (merged spec from all sub-services)

### OSS Base

Repo: `hanzoai/treasury` (Formance Stack fork). Default branch: `main`.

## When to use

- Recording financial transactions with double-entry accounting
- Payment routing across multiple providers
- Multi-currency virtual wallet management
- Reconciling ledger data against external payment sources
- Orchestrating multi-step payment workflows
- Modeling complex fee splits and fund distributions with Numscript

## Hard requirements

1. **Go 1.22+** for building
2. **Earthly** for the build system
3. **PostgreSQL** for ledger and payment data storage
4. **Docker** for running the full stack locally

## Quick reference

| Item | Value |
|------|-------|
| Ledger API | `POST /v2/ledger/{name}/transactions` |
| Accounts API | `GET /v2/ledger/{name}/accounts` |
| Payments API | `POST /v2/payments` |
| Wallets API | `POST /v2/wallets` |
| Reconciliation API | `POST /v2/reconciliation/policies` |
| Local Port | 3068 (ledger) |
| License | MIT |
| Repo | `github.com/hanzoai/treasury` |

## One-file quickstart

### Start local stack

```bash
docker compose up -d
```

### Create a ledger transaction

```bash
curl -X POST http://localhost:3068/v2/ledger/demo/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "postings": [{
      "source": "world",
      "destination": "users:001",
      "amount": 10000,
      "asset": "USD/2"
    }]
  }'
```

### Numscript example

```numscript
// Transfer with multi-party fee split
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

## Core Concepts

### Architecture

```
hanzo/commerce    Storefront, catalog, orders
       |
hanzo/payments    Payment routing (50+ processors)
       |
hanzo/treasury    Ledger, reconciliation, wallets, flows   <-- this repo
       |
lux/treasury      On-chain treasury, MPC/KMS wallets
```

### Components

| Component | Description |
|-----------|-------------|
| **Ledger** | Programmable double-entry, immutable source of truth |
| **Payments** | Unified API and data layer for payment processing |
| **Numscript** | DSL for modeling monetary computations and flows |
| **Wallets** | Multi-currency virtual wallets with hold/release |
| **Reconciliation** | Auto-match ledger vs. payment provider data |
| **Flows** | Workflow orchestration (via `orchestration` service) |
| **Webhooks** | Event delivery for financial state changes |
| **Auth** | Authentication and authorization for treasury APIs |
| **Search** | Full-text search across treasury data |
| **Gateway** | API gateway for all treasury services |

### Service Versions (from Earthfile)

| Service | Version |
|---------|---------|
| Ledger | v2.3.1 |
| Payments | v3.0.18 |
| Wallets | v2.1.5 |
| Webhooks | v2.2.0 |
| Auth | v2.4.0 |
| Search | v2.1.0 |
| Orchestration | v2.4.0 |
| Reconciliation | v2.2.1 |
| Gateway | v2.1.0 |

### Event System

Events are JSON Schema-validated. The `events/` directory contains:

- `base.yaml` - Common event envelope (type, app, payload)
- `services/ledger/` - Ledger events (e.g., `SAVED_TRANSACTION`)
- `services/payments/` - Payment events (e.g., `SAVED_PAYMENT`)
- `services/orchestration/` - Workflow events

Event validation in Go:
```go
import "github.com/formancehq/stack/events"

err := events.Check(data, "payments", "SAVED_PAYMENT")
```

### OpenAPI Specification

The `releases/` directory merges OpenAPI specs from all sub-services into a unified API spec:

- Auth: `/api/auth/*`
- Ledger: `/api/ledger/*`
- Payments: `/api/payments/*`
- Wallets: `/api/wallets/*`
- Webhooks: `/api/webhooks/*`
- Search: `/api/search/*`
- Orchestration: `/api/orchestration/*`
- Reconciliation: `/api/reconciliation/*`
- Gateway: top-level

Authentication: OAuth2 client credentials (`/api/auth/oauth/token`).

### Directory Structure

```
treasury/
  README.md              # Project overview
  LLM.md                 # AI agent documentation
  LICENSE                # MIT
  go.mod                 # Root Go module (github.com/formancehq/stack)
  Earthfile              # Earthly build (downloads + merges OpenAPI specs)
  base.Dockerfile        # Base Docker image
  events/
    base.yaml            # Common event envelope schema
    events.go            # Go event validation library
    services/
      ledger/            # Ledger event schemas
      payments/          # Payment event schemas
      orchestration/     # Orchestration event schemas
    generated/           # Auto-generated event code
  libs/
    events/
      generated/         # Generated event library
  releases/
    base.yaml            # OpenAPI 3.0.3 base spec
    openapi-merge.json   # Config for merging all service specs
    openapi-overlay.json # Spec overlays
```

### Integration with Hanzo Stack

- **hanzo/payments** - Payment routing layer (upstream source of payment connectors)
- **hanzo/commerce** - Order settlement and storefront
- **lux/treasury** - On-chain treasury operations, MPC wallet management
- **hanzo/kms** - Secrets and key management

## Building

```bash
# Build all (OpenAPI spec + events)
earthly +build

# Build just the merged OpenAPI spec
earthly +build-final-spec

# Go tests
go test ./...
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Earthly build fails | Missing Earthly CLI | Install with `brew install earthly` |
| Event validation fails | Schema version mismatch | Check `events/services/` for latest version dir |
| API 401 | Missing OAuth token | Get token from `/api/auth/oauth/token` |
| Ledger balance error | Insufficient funds in source | Check account balances via `GET /v2/ledger/{name}/accounts` |

## Related Skills

- `hanzo/hanzo-payments.md` - Payment routing
- `hanzo/hanzo-commerce.md` - E-commerce platform
- `hanzo/hanzo-ledger.md` - Ledger details
- `hanzo/hanzo-kms.md` - Secret management
- `hanzo/hanzo-sql.md` - PostgreSQL backend

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: treasury, ledger, payments, wallets, reconciliation, numscript, fintech
**Prerequisites**: Go 1.22+, Earthly, Docker, PostgreSQL
