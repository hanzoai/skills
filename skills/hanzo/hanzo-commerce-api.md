# Hanzo Commerce - E-Commerce & Payments Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-vault.md`, `hanzo/hanzo-web3.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo Commerce is a **Go/Gin e-commerce API** with SQLite storage, providing products, orders, subscriptions, invoices, usage metering, and multi-currency support (fiat + crypto). Powers billing for Hanzo Cloud, Console, and third-party integrations.

**NOTE**: The repo is `hanzoai/commerce` (not `commerce-api`). Built with **Go + Gin + SQLite** with 35+ API handler directories. For payment processing, Hanzo also has `hanzoai/payments` — a **Rust Hyperswitch fork** supporting 50+ payment processors.

### Components

| Component | Repo | Stack | Purpose |
|-----------|------|-------|---------|
| **Commerce API** | `hanzoai/commerce` | Go, Gin, SQLite | Products, orders, subscriptions, billing |
| **Payments** | `hanzoai/payments` | Rust (Hyperswitch fork) | 50+ payment processors, routing |

### Why Hanzo Commerce?

- **Go/Gin backend**: Fast, lightweight, easy to deploy
- **SQLite storage**: Zero-config embedded database
- **35+ endpoint groups**: Full e-commerce API surface
- **Multi-currency**: USD, EUR, USDC, ETH, LUX, BTC
- **Usage-based**: Metered billing for API calls, compute, storage
- **50+ processors**: Via Payments service (Stripe, Adyen, PayPal, crypto, etc.)
- **PCI compliant**: Card handling via Hanzo Vault

## When to use

- Creating products, plans, or pricing tiers
- Processing payments or managing subscriptions
- Usage-based billing for API or compute services
- Multi-currency payments (fiat + crypto)
- Subscription lifecycle management
- Invoice generation and tracking

## Quick reference

| Item | Value |
|------|-------|
| Commerce API | `https://api.hanzo.ai/v1/commerce` |
| Commerce Repo | `github.com/hanzoai/commerce` |
| Payments Repo | `github.com/hanzoai/payments` |
| Commerce Stack | Go, Gin, SQLite |
| Payments Stack | Rust (Hyperswitch fork) |
| Auth | Bearer token (Hanzo API key) |

## Commerce API (Go/Gin)

### Handler Groups (35+)

The Commerce API organizes handlers by resource type:

| Group | Endpoints | Description |
|-------|-----------|-------------|
| products | CRUD | Product catalog management |
| prices | CRUD | Pricing configurations |
| customers | CRUD | Customer management |
| orders | CRUD + status | Order lifecycle |
| subscriptions | CRUD + cancel | Recurring billing |
| invoices | CRUD + pay | Billing statements |
| payments | create + capture | Payment transactions |
| usage | report + query | Metered usage tracking |
| webhooks | CRUD | Event notifications |
| coupons | CRUD | Discount codes |
| tax_rates | CRUD | Tax configuration |
| shipping | CRUD | Shipping methods |
| refunds | create + list | Refund processing |
| disputes | list + respond | Chargeback handling |
| payment_methods | CRUD | Stored payment methods |
| plans | CRUD | Subscription plan templates |

### Running Locally

```bash
git clone https://github.com/hanzoai/commerce.git
cd commerce
go build ./...
go test ./...

# Start server
go run main.go --port 4242 --db commerce.db
```

### API Examples

```bash
# Create product
curl -X POST http://localhost:4242/v1/products \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "name": "AI API Access",
    "type": "service",
    "description": "Access to Hanzo AI APIs"
  }'

# Create price
curl -X POST http://localhost:4242/v1/prices \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "product_id": "prod_123",
    "amount": 2999,
    "currency": "usd",
    "interval": "month",
    "type": "recurring"
  }'

# Create subscription
curl -X POST http://localhost:4242/v1/subscriptions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "customer_id": "cus_123",
    "price_id": "price_456",
    "payment_method": "tok_card_789"
  }'

# Report usage
curl -X POST http://localhost:4242/v1/usage \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "subscription_id": "sub_123",
    "metric": "api_calls",
    "quantity": 1000
  }'
```

## Payments Service (Rust/Hyperswitch)

The `hanzoai/payments` repo is a **Rust fork of Hyperswitch** providing payment processor routing:

### Supported Processors (50+)

| Category | Processors |
|----------|-----------|
| **Cards** | Stripe, Adyen, Braintree, Checkout.com, Worldpay, Cybersource |
| **Digital Wallets** | Apple Pay, Google Pay, PayPal, Venmo |
| **BNPL** | Klarna, Affirm, Afterpay |
| **Bank** | ACH, SEPA, iDEAL, Bancontact |
| **Crypto** | Coinbase Commerce, BitPay, various on-chain |
| **Regional** | Boleto, PIX, UPI, GrabPay, GCash |

### Smart Routing

```yaml
# Payment routing configuration
routing:
  rules:
    - condition: amount > 10000
      processor: stripe     # High-value → primary processor
      fallback: adyen
    - condition: currency == "BTC"
      processor: coinbase   # Crypto → crypto processor
    - default:
      processor: stripe
      fallback: [adyen, checkout]
```

## SDK Usage

### Python

```python
from hanzoai import Hanzo

client = Hanzo()

# Create a product
product = client.commerce.products.create(
    name="AI API Access",
    type="service",
)

# Create subscription
sub = client.commerce.subscriptions.create(
    customer_id="cus_123",
    price_id="price_456",
)

# Report usage
client.commerce.usage.report(
    subscription_id=sub.id,
    metric="api_calls",
    quantity=1000,
)
```

### TypeScript

```typescript
import Hanzo from "hanzoai"

const client = new Hanzo()

const product = await client.commerce.products.create({
  name: "AI API Access",
  type: "service",
})

const sub = await client.commerce.subscriptions.create({
  customerId: "cus_123",
  priceId: "price_456",
})
```

## Webhooks

| Event | Description |
|-------|-------------|
| `subscription.created` | New subscription activated |
| `subscription.updated` | Plan or quantity changed |
| `subscription.cancelled` | Subscription cancelled |
| `invoice.created` | New invoice generated |
| `invoice.paid` | Invoice successfully paid |
| `invoice.payment_failed` | Payment attempt failed |
| `payment.succeeded` | One-time payment succeeded |
| `payment.failed` | Payment attempt failed |
| `usage.threshold` | Usage crossed configured threshold |

## Related Skills

- `hanzo/hanzo-vault.md` - PCI card tokenization
- `hanzo/hanzo-web3.md` - Crypto payments
- `hanzo/hanzo-cloud.md` - Billing dashboard

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: billing, payments, subscriptions, commerce, go, hyperswitch
**Prerequisites**: Go or API concepts, payment processing basics
