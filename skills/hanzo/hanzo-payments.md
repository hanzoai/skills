# Hanzo Payments - Payment Orchestration Switch

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-vault.md`, `hanzo/hanzo-kms.md`

## Overview

Hanzo Payments is an **open-source payment orchestration switch** written in Rust. It routes payments across 50+ processors with smart retries, fallback cascading, and unified analytics. Based on [Hyperswitch](https://github.com/juspay/hyperswitch). Exposes a REST API on port 8080. Licensed Apache 2.0.

### Why Hanzo Payments?

- **Smart routing**: Route to optimal processor based on cost, success rate, and latency
- **50+ connectors**: Stripe, Adyen, Braintree, Checkout.com, PayPal, Klarna, Coinbase, and more
- **Automatic retries**: Cascade to backup processors on failure
- **Unified API**: Single integration for cards, bank transfers, wallets, BNPL, crypto
- **PCI DSS**: Vault-based card tokenization via `hanzo/vault`
- **3DS Authentication**: Native 3D Secure support
- **Multi-currency**: 135+ currencies with automatic FX
- **Decision engine**: TOML-based routing rules (Euclid engine)

### Tech Stack

- **Language**: Rust (edition 2021, MSRV 1.85.0)
- **Database**: PostgreSQL (via Diesel ORM)
- **Cache**: Redis
- **Build**: Cargo workspace with 35+ crates
- **Monitoring**: Prometheus, Grafana, Loki, Tempo, OpenTelemetry
- **API spec**: OpenAPI / Smithy
- **Testing**: Cargo test, Newman (Postman), Cypress E2E

### OSS Base

Repo: `hanzoai/payments` (fork of `juspay/hyperswitch`).

## When to use

- Processing payments across multiple payment providers
- Building smart payment routing with fallback logic
- Accepting cards, bank payments, wallets, BNPL, or crypto
- PCI-compliant card tokenization and vaulting
- Multi-currency payment processing with automatic FX
- Real-time payment analytics and reporting

## Hard requirements

1. **Rust toolchain** (1.85.0+) with protobuf compiler
2. **PostgreSQL** database
3. **Redis** for caching and process tracking
4. **libpq-dev, libssl-dev** system libraries

## Quick reference

| Item | Value |
|------|-------|
| API Port | `8080` |
| Language | Rust 2021 edition |
| DB | PostgreSQL (Diesel ORM) |
| Cache | Redis |
| Config | TOML (`config/development.toml`) |
| Binaries | `router`, `scheduler` (consumer/producer) |
| License | Apache 2.0 |
| Repo | `github.com/hanzoai/payments` |

## One-file quickstart

### Docker Compose

```bash
# Start all services
docker compose up -d

# Create a payment
curl -X POST http://localhost:8080/payments/create \
  -H "Content-Type: application/json" \
  -H "api-key: dev_key" \
  -d '{
    "amount": 1000,
    "currency": "USD",
    "payment_method": "card",
    "payment_method_data": {
      "card": {
        "card_number": "4242424242424242",
        "card_exp_month": "12",
        "card_exp_year": "2027",
        "card_cvc": "123"
      }
    },
    "connector": "stripe"
  }'
```

### Local Development

```bash
# Build
cargo build

# Run tests
cargo test --all-features

# Format and lint
cargo +nightly fmt --all
cargo clippy --all-features --all-targets -- -D warnings

# Build release binary (stripped, LTO)
cargo build --release --no-default-features --features release --features v1
```

## Core Concepts

### Architecture

```
hanzo/commerce    Storefront, catalog, orders
       |
hanzo/payments    Payment routing (50+ processors)   <-- this service
       |
hanzo/treasury    Ledger, reconciliation, wallets
       |
lux/treasury      On-chain treasury, MPC/KMS wallets
```

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `router` | Main application server (port 8080) |
| `scheduler` | Process tracker (consumer + producer) |
| `analytics` | Payment analytics and reporting |
| `api_models` | API request/response types |
| `cards` | Card number handling and validation |
| `drainer` | Event draining to storage |
| `euclid` | Smart routing decision engine |
| `euclid_wasm` | WebAssembly build of routing engine |
| `hyperswitch_connectors` | 50+ payment processor integrations |
| `hyperswitch_domain_models` | Core domain types |
| `hyperswitch_interfaces` | Connector trait interfaces |
| `diesel_models` | Database models (Diesel ORM) |
| `storage_impl` | Storage layer implementation |
| `masking` | PII data masking |
| `redis_interface` | Redis client wrapper |
| `payment_methods` | Payment method management |
| `payment_link` | Payment link generation |
| `subscriptions` | Recurring payment support |
| `kgraph_utils` | Knowledge graph for routing |
| `smithy` / `smithy-core` / `smithy-generator` | API spec generation |
| `currency_conversion` | FX conversion |

### Supported Processors

| Category | Processors |
|----------|-----------|
| **Cards** | Stripe, Adyen, Braintree, Checkout.com, Cybersource, Worldpay, NMI, Authorise.net, Square |
| **Bank** | Plaid, GoCardless, ACH (Column, Modern Treasury), SEPA, BACS |
| **Wallets** | Apple Pay, Google Pay, PayPal, Venmo, Cash App |
| **BNPL** | Klarna, Affirm, Afterpay, Sezzle |
| **Crypto** | Coinbase Commerce, BitPay, NOWPayments |
| **Regional** | Mercado Pago, Razorpay, Paytm, Mollie, iDEAL, Bancontact |
| **Wire** | Wise, CurrencyCloud, SWIFT, Fedwire |

### Decision Engine (Euclid)

Smart routing rules defined in TOML:

```toml
[[rules]]
name = "route_high_value"
condition = "amount > 10000 AND currency == 'USD'"
action = "route"
connector = "adyen"
fallback = ["stripe", "checkout"]

[[rules]]
name = "route_eu"
condition = "country IN ['DE', 'FR', 'NL', 'BE']"
action = "route"
connector = "mollie"
fallback = ["adyen"]
```

The Euclid engine also compiles to WebAssembly for client-side routing preview.

### Binaries

The Dockerfile builds three binary targets:

1. **`router`** -- Main application server (default)
2. **`scheduler` (consumer)** -- Process tracker consumer
3. **`scheduler` (producer)** -- Process tracker producer

### Observability

Full monitoring stack via config directory:
- **Prometheus** -- Metrics collection
- **Grafana** -- Dashboards and visualization
- **Loki** + **Promtail** -- Log aggregation
- **Tempo** -- Distributed tracing
- **OpenTelemetry Collector** -- Telemetry pipeline
- **Vector** -- Log routing

### Integration with Hanzo Stack

Payments connects to:
- **hanzo/commerce** for order checkout and settlement
- **hanzo/treasury** for ledger recording and reconciliation
- **hanzo/vault** for PCI-compliant card tokenization
- **hanzo/kms** for API key and secret management

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails | Missing system deps | `apt-get install libpq-dev libssl-dev pkg-config protobuf-compiler` |
| WASM build fails | Missing wasm-pack | `cargo install wasm-pack` then `make euclid-wasm` |
| Clippy warnings | Strict lint config | Run `cargo clippy --all-features --all-targets -- -D warnings` |
| Stack overflow at runtime | Default stack too small | Set `RUST_MIN_STACK=6291456` |

## Related Skills

- `hanzo/hanzo-commerce.md` - E-commerce platform
- `hanzo/hanzo-vault.md` - PCI-compliant card tokenization
- `hanzo/hanzo-kms.md` - Secret management
- `hanzo/hanzo-commerce-api.md` - Commerce API

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: payments, commerce, stripe, processing, routing
**Prerequisites**: Rust, PostgreSQL, Redis
