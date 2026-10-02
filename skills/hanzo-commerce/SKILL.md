---
name: hanzo-commerce
description: "E-commerce platform with product catalog, orders, payments (Stripe), inventory, fulfillment workflows, and analytics."
metadata: { "bot": { "requires": { "bins": ["curl"] } } }
---

# Hanzo Commerce — E-Commerce Platform

Full-featured e-commerce API for product catalog, orders, payments, inventory, and fulfillment. Integrates with Stripe, Meilisearch, and analytics.

## Quick Start

```bash
COMMERCE_URL=http://localhost:8001

# List products
curl -s -H "Authorization: Bearer $TOKEN" \
  "$COMMERCE_URL/api/products" | jq

# Create order
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"product_id": "...", "quantity": 1}]}' \
  "$COMMERCE_URL/api/orders" | jq
```

## Products

```bash
# Create product
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Agent Credits",
    "price": 99.00,
    "currency": "USD",
    "description": "1000 agent execution credits"
  }' "$COMMERCE_URL/api/products" | jq

# Search products (via Meilisearch)
curl -s "$COMMERCE_URL/api/products/search?q=agent" | jq
```

## Orders

```bash
# Create order
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [{"product_id": "prod_123", "quantity": 2}],
    "payment_method": "pm_..."
  }' "$COMMERCE_URL/api/orders" | jq

# Get order status
curl -s -H "Authorization: Bearer $TOKEN" \
  "$COMMERCE_URL/api/orders/ord_123" | jq
```

## Payments (Stripe)

```bash
# Create payment intent
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"amount": 9900, "currency": "usd"}' \
  "$COMMERCE_URL/api/payments/intent" | jq

# Webhook handling (automatic)
# Stripe webhooks → $COMMERCE_URL/api/webhooks/stripe
```

## Subscriptions

```bash
# Create subscription
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan_enterprise",
    "payment_method": "pm_..."
  }' "$COMMERCE_URL/api/subscriptions" | jq
```

## Port

- API: `8001`

## Environment Variables

```bash
COMMERCE_PORT=8001
COMMERCE_SECRET=...
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
```
