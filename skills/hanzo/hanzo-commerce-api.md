# Hanzo Commerce API - Billing, Payments & E-Commerce

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-vault.md`, `hanzo/hanzo-web3.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo Commerce provides a **full e-commerce and billing API** — products, orders, subscriptions, invoices, usage-based metering, and multi-currency support (fiat + crypto).

## When to use

- Creating products, plans, or pricing
- Processing payments or managing subscriptions
- Usage-based billing for API or compute services
- Multi-currency payments (USD, EUR, USDC, ETH, LUX)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/commerce` |
| Local | ``github.com/hanzoai/commerce`` |

## Endpoints

| Resource | Endpoint | Method |
|----------|----------|--------|
| Products | `/v1/products` | CRUD |
| Orders | `/v1/orders` | CRUD |
| Subscriptions | `/v1/subscriptions` | CRUD |
| Invoices | `/v1/invoices` | Read |
| Payments | `/v1/payments` | Create |
| Usage | `/v1/usage` | Report/Query |

## One-file quickstart

```python
from hanzo import Hanzo

client = Hanzo()

# Create a product
product = client.commerce.products.create(
    name="AI API Access",
    type="service",
    prices=[{
        "amount": 2999,  # $29.99
        "currency": "usd",
        "interval": "month",
    }]
)

# Create subscription
sub = client.commerce.subscriptions.create(
    customer_id="cus_123",
    price_id=product.prices[0].id,
)

# Report usage
client.commerce.usage.report(
    subscription_id=sub.id,
    metric="api_calls",
    quantity=1000,
)
```

## Related Skills

- `hanzo/hanzo-vault.md` - PCI card tokenization
- `hanzo/hanzo-web3.md` - Crypto payments
- `hanzo/hanzo-cloud.md` - Billing dashboard

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
