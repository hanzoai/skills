# Hanzo Commerce - Billing, Payments, and E-Commerce API

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-web3.md`, `hanzo/hanzo-chat.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Commerce is the **billing and payments infrastructure** for the Hanzo ecosystem and any application that needs e-commerce capabilities. It handles products, orders, subscriptions, invoices, payments, and usage-based billing through a single REST API. Built for both traditional e-commerce and AI/API usage metering.

### Why Hanzo Commerce?

- **Unified billing**: Products, subscriptions, invoices, and payments in one API
- **Usage-based metering**: Bill per API call, token, or compute unit
- **Multi-currency**: Fiat (USD, EUR) and crypto (USDC, ETH, LUX)
- **Webhook events**: Real-time notifications for payment lifecycle events
- **ORM-backed**: Strongly typed models with automatic serialization
- **Part of Hanzo ecosystem**: Powers billing for Hanzo Chat, Web3, Console

## When to use

Use this skill when:
- The user needs to create products, plans, or pricing
- The user wants to process payments or manage subscriptions
- The user needs usage-based billing for API or compute services
- The user wants to generate invoices
- The user is building e-commerce features into an application

## Hard requirements

1. **API Key required.** `HANZO_API_KEY` must be set. Get one at https://hanzo.ai/dashboard.
2. **Never expose the API key** in user-visible output, logs, or screenshots.
3. **PCI compliance**: Never log or store raw card numbers. Use tokenized references only.

## Preflight checks

Before making any request, silently verify:
- `HANZO_API_KEY` environment variable is set and non-empty
- If unset, suggest: `export HANZO_API_KEY=<your-key>`

## Quick reference

| Item | Value |
|------|-------|
| Base URL | `https://commerce.hanzo.ai/v1` |
| Auth header | `Authorization: Bearer ${HANZO_API_KEY}` |
| Content type | `application/json` |
| Docs | https://hanzo.ai/docs/commerce |
| Dashboard | https://hanzo.ai/dashboard/billing |

## One-file quickstart

### Create a product

```bash
curl -X POST https://commerce.hanzo.ai/v1/products \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{
    "name": "Pro Plan",
    "slug": "pro-plan",
    "description": "Full access to all AI models",
    "price": 4900,
    "currency": "usd"
  }'
```

### Create a subscription

```bash
curl -X POST https://commerce.hanzo.ai/v1/subscriptions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{
    "customer_id": "cus_abc123",
    "product_id": "prod_xyz789",
    "interval": "month"
  }'
```

### Record usage

```bash
curl -X POST https://commerce.hanzo.ai/v1/usage \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{
    "customer_id": "cus_abc123",
    "metric": "api_calls",
    "quantity": 1500,
    "timestamp": "2026-02-26T00:00:00Z"
  }'
```

### Python

```python
from hanzo import Hanzo

client = Hanzo()  # uses HANZO_API_KEY from env

# Create product
product = client.commerce.products.create(
    name="Pro Plan",
    price=4900,
    currency="usd",
)

# Create subscription
sub = client.commerce.subscriptions.create(
    customer_id="cus_abc123",
    product_id=product.id,
    interval="month",
)

# Record usage
client.commerce.usage.record(
    customer_id="cus_abc123",
    metric="api_calls",
    quantity=1500,
)

# Generate invoice
invoice = client.commerce.invoices.create(
    customer_id="cus_abc123",
    auto_collect=True,
)
```

## Endpoint selector

### Products

| Task | Endpoint | Method |
|------|----------|--------|
| Create product | `POST /v1/products` | POST |
| List products | `GET /v1/products` | GET |
| Get product | `GET /v1/products/{id}` | GET |
| Update product | `PATCH /v1/products/{id}` | PATCH |
| Delete product | `DELETE /v1/products/{id}` | DELETE |

### Orders

| Task | Endpoint | Method |
|------|----------|--------|
| Create order | `POST /v1/orders` | POST |
| List orders | `GET /v1/orders` | GET |
| Get order | `GET /v1/orders/{id}` | GET |
| Update order | `PATCH /v1/orders/{id}` | PATCH |
| Cancel order | `POST /v1/orders/{id}/cancel` | POST |

### Subscriptions

| Task | Endpoint | Method |
|------|----------|--------|
| Create subscription | `POST /v1/subscriptions` | POST |
| List subscriptions | `GET /v1/subscriptions` | GET |
| Get subscription | `GET /v1/subscriptions/{id}` | GET |
| Update subscription | `PATCH /v1/subscriptions/{id}` | PATCH |
| Cancel subscription | `POST /v1/subscriptions/{id}/cancel` | POST |
| Resume subscription | `POST /v1/subscriptions/{id}/resume` | POST |

### Invoices

| Task | Endpoint | Method |
|------|----------|--------|
| Create invoice | `POST /v1/invoices` | POST |
| List invoices | `GET /v1/invoices` | GET |
| Get invoice | `GET /v1/invoices/{id}` | GET |
| Pay invoice | `POST /v1/invoices/{id}/pay` | POST |
| Void invoice | `POST /v1/invoices/{id}/void` | POST |

### Payments

| Task | Endpoint | Method |
|------|----------|--------|
| Create payment | `POST /v1/payments` | POST |
| List payments | `GET /v1/payments` | GET |
| Get payment | `GET /v1/payments/{id}` | GET |
| Refund payment | `POST /v1/payments/{id}/refund` | POST |

### Usage & Metering

| Task | Endpoint | Method |
|------|----------|--------|
| Record usage | `POST /v1/usage` | POST |
| Get usage summary | `GET /v1/usage/summary` | GET |
| List usage records | `GET /v1/usage` | GET |

### Customers

| Task | Endpoint | Method |
|------|----------|--------|
| Create customer | `POST /v1/customers` | POST |
| List customers | `GET /v1/customers` | GET |
| Get customer | `GET /v1/customers/{id}` | GET |
| Update customer | `PATCH /v1/customers/{id}` | PATCH |

### Webhooks

| Task | Endpoint | Method |
|------|----------|--------|
| Create webhook | `POST /v1/webhooks` | POST |
| List webhooks | `GET /v1/webhooks` | GET |
| Delete webhook | `DELETE /v1/webhooks/{id}` | DELETE |

Supported event types: `payment.completed`, `payment.failed`, `subscription.created`, `subscription.cancelled`, `invoice.paid`, `invoice.overdue`, `order.completed`, `usage.threshold`.

## Usage-based billing pattern

For AI/API platforms that bill by usage:

```python
# 1. Create a metered product
product = client.commerce.products.create(
    name="API Access",
    pricing_type="metered",
    unit_label="API calls",
    tiers=[
        {"up_to": 1000, "unit_price": 0},        # Free tier
        {"up_to": 100000, "unit_price": 0.001},   # $0.001/call
        {"up_to": None, "unit_price": 0.0005},    # Volume discount
    ],
)

# 2. Subscribe customer
sub = client.commerce.subscriptions.create(
    customer_id="cus_abc123",
    product_id=product.id,
    interval="month",
)

# 3. Record usage as it happens (idempotent)
client.commerce.usage.record(
    customer_id="cus_abc123",
    metric="api_calls",
    quantity=1,
    idempotency_key="req_abc123",
)

# 4. Invoice auto-generates at period end
```

## MCP Integration

Expose commerce capabilities as MCP tools:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'

const commerceTools: Tool[] = [
  {
    name: 'commerce_create_order',
    description: 'Create a new order for a customer',
    parameters: {
      customer_id: { type: 'string', required: true },
      items: { type: 'array', required: true },
      currency: { type: 'string', default: 'usd' }
    },
    async execute({ customer_id, items, currency }) {
      const res = await fetch('https://commerce.hanzo.ai/v1/orders', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.HANZO_API_KEY}`
        },
        body: JSON.stringify({ customer_id, items, currency })
      })
      return await res.json()
    }
  },
  {
    name: 'commerce_record_usage',
    description: 'Record metered usage for billing',
    parameters: {
      customer_id: { type: 'string', required: true },
      metric: { type: 'string', required: true },
      quantity: { type: 'number', required: true }
    },
    async execute({ customer_id, metric, quantity }) {
      const res = await fetch('https://commerce.hanzo.ai/v1/usage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.HANZO_API_KEY}`
        },
        body: JSON.stringify({ customer_id, metric, quantity })
      })
      return await res.json()
    }
  }
]
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check parameters, required fields |
| 401 | Unauthorized | Check API key |
| 404 | Not found | Check resource ID |
| 409 | Conflict | Duplicate idempotency key or state conflict |
| 422 | Validation error | Check field types and constraints |
| 429 | Rate limited | Exponential backoff |
| 500 | Server error | Retry up to 3 times |

## Official links

- Documentation: https://hanzo.ai/docs/commerce
- Dashboard: https://hanzo.ai/dashboard/billing
- API Reference: https://hanzo.ai/docs/commerce/api
- Hanzo AI: https://hanzo.ai

---

**Last Updated**: 2026-02-26
**Category**: Hanzo Ecosystem
**Related**: billing, payments, e-commerce, subscriptions
**Prerequisites**: HTTP/curl, REST API basics
