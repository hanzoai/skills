# Hanzo Commerce API - Billing, Payments & E-Commerce

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-vault.md`, `hanzo/hanzo-web3.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo Commerce provides a **full e-commerce and billing API** — products, orders, subscriptions, invoices, usage-based metering, and multi-currency support (fiat + crypto). Powers billing for Hanzo Cloud, Console, and third-party integrations.

### Why Hanzo Commerce?

- **Unified billing**: Products, subscriptions, invoices, metering in one API
- **Multi-currency**: USD, EUR, USDC, ETH, LUX, BTC
- **Usage-based**: Metered billing for API calls, compute, storage
- **Webhook-driven**: Real-time events for payment lifecycle
- **PCI compliant**: Card handling via Hanzo Vault (zero PCI scope for your app)
- **Crypto-native**: Native support for on-chain payments

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
| API Base | `https://api.hanzo.ai/v1/commerce` |
| Repo | `github.com/hanzoai/commerce` |
| Auth | Bearer token (Hanzo API key) |
| Webhooks | `POST` to your endpoint |

## API Reference

### Products

```bash
# Create product
curl -X POST https://api.hanzo.ai/v1/commerce/products \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "name": "AI API Access",
    "type": "service",
    "description": "Access to Hanzo AI APIs",
    "metadata": {"tier": "pro"}
  }'

# List products
curl https://api.hanzo.ai/v1/commerce/products \
  -H "Authorization: Bearer $HANZO_API_KEY"

# Get product
curl https://api.hanzo.ai/v1/commerce/products/prod_123 \
  -H "Authorization: Bearer $HANZO_API_KEY"

# Update product
curl -X PATCH https://api.hanzo.ai/v1/commerce/products/prod_123 \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{"name": "AI API Pro"}'
```

### Prices

```bash
# Create price
curl -X POST https://api.hanzo.ai/v1/commerce/prices \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "product_id": "prod_123",
    "amount": 2999,
    "currency": "usd",
    "interval": "month",
    "type": "recurring"
  }'

# Usage-based price
curl -X POST https://api.hanzo.ai/v1/commerce/prices \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "product_id": "prod_123",
    "currency": "usd",
    "type": "metered",
    "tiers": [
      {"up_to": 1000, "unit_amount": 1},
      {"up_to": 10000, "unit_amount": 0.5},
      {"up_to": null, "unit_amount": 0.1}
    ]
  }'
```

### Subscriptions

```bash
# Create subscription
curl -X POST https://api.hanzo.ai/v1/commerce/subscriptions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "customer_id": "cus_123",
    "price_id": "price_456",
    "payment_method": "tok_card_789"
  }'

# Cancel subscription
curl -X POST https://api.hanzo.ai/v1/commerce/subscriptions/sub_123/cancel \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{"at_period_end": true}'

# Update subscription
curl -X PATCH https://api.hanzo.ai/v1/commerce/subscriptions/sub_123 \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{"price_id": "price_789"}'
```

### Usage Reporting

```bash
# Report usage
curl -X POST https://api.hanzo.ai/v1/commerce/usage \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "subscription_id": "sub_123",
    "metric": "api_calls",
    "quantity": 1000,
    "timestamp": "2026-03-13T00:00:00Z"
  }'

# Query usage
curl "https://api.hanzo.ai/v1/commerce/usage?subscription_id=sub_123&metric=api_calls&period=current" \
  -H "Authorization: Bearer $HANZO_API_KEY"
```

### Invoices

```bash
# List invoices
curl "https://api.hanzo.ai/v1/commerce/invoices?customer_id=cus_123" \
  -H "Authorization: Bearer $HANZO_API_KEY"

# Get invoice
curl https://api.hanzo.ai/v1/commerce/invoices/inv_123 \
  -H "Authorization: Bearer $HANZO_API_KEY"

# Pay invoice
curl -X POST https://api.hanzo.ai/v1/commerce/invoices/inv_123/pay \
  -H "Authorization: Bearer $HANZO_API_KEY"
```

### Orders (One-time)

```bash
# Create order
curl -X POST https://api.hanzo.ai/v1/commerce/orders \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "customer_id": "cus_123",
    "items": [
      {"product_id": "prod_123", "quantity": 1, "amount": 9999}
    ],
    "currency": "usd",
    "payment_method": "tok_card_789"
  }'
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
    prices=[{
        "amount": 2999,
        "currency": "usd",
        "interval": "month",
    }],
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

# List invoices
invoices = client.commerce.invoices.list(customer_id="cus_123")
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

```bash
# Register webhook
curl -X POST https://api.hanzo.ai/v1/commerce/webhooks \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -d '{
    "url": "https://your-app.com/webhooks/commerce",
    "events": [
      "subscription.created",
      "subscription.cancelled",
      "invoice.paid",
      "invoice.payment_failed",
      "payment.succeeded",
      "payment.failed"
    ]
  }'
```

### Webhook Events

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

## Resource Types

| Resource | ID Prefix | Description |
|----------|-----------|-------------|
| Product | `prod_` | Goods or services |
| Price | `price_` | Pricing configuration |
| Customer | `cus_` | Billing entity |
| Subscription | `sub_` | Recurring billing |
| Invoice | `inv_` | Billing statement |
| Order | `ord_` | One-time purchase |
| Payment | `pay_` | Payment transaction |
| Token | `tok_` | Payment method (via Vault) |

## Related Skills

- `hanzo/hanzo-vault.md` - PCI card tokenization
- `hanzo/hanzo-web3.md` - Crypto payments
- `hanzo/hanzo-cloud.md` - Billing dashboard
- `hanzo/hanzo-commerce.md` - Legacy commerce platform

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: billing, payments, subscriptions, commerce, metering
**Prerequisites**: API concepts, payment processing basics
