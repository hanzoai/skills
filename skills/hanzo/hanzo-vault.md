# Hanzo Vault - PCI-Compliant Card Tokenization

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kms.md`, `hanzo/hanzo-commerce-api.md`

## Overview

Hanzo Vault is a **PCI DSS-compliant card tokenization service** (CDE - Cardholder Data Environment). Securely tokenize, store, and detokenize payment card data without handling raw card numbers in your application. Written in Go with hardware-backed encryption.

### Why Hanzo Vault?

- **PCI DSS Level 1**: Full cardholder data environment isolation
- **Zero plaintext**: Card numbers never exist in plaintext outside Vault
- **Hardware-backed**: Encryption keys in HSM/KMS, never in memory
- **Go performance**: Sub-millisecond tokenization
- **Format-preserving**: Tokens maintain card number format (Luhn-valid)
- **Audit trail**: Every access logged for compliance

### Architecture

Vault is an **internal service** — never exposed publicly. Applications call Vault's internal API to tokenize/detokenize. Only Vault touches raw card data, keeping PCI scope minimal.

## When to use

- Processing credit/debit card payments
- PCI DSS compliance requirements
- Tokenizing sensitive payment data
- Building e-commerce payment flows
- Storing card-on-file for subscriptions

## Hard requirements

1. **Internal network only** (never expose to public internet)
2. **KMS integration** for encryption key management
3. **PostgreSQL** for token storage
4. **TLS** for all connections (even internal)

## Quick reference

| Item | Value |
|------|-------|
| Language | Go |
| API | Internal gRPC + HTTP |
| Port | 8080 (HTTP), 8443 (gRPC) |
| Tokenize | `POST /v1/tokens` |
| Detokenize | `POST /v1/detokenize` |
| Repo | `github.com/hanzoai/vault` |
| License | Proprietary |

## API Reference

### Tokenize Card

```http
POST /v1/tokens
Content-Type: application/json
Authorization: Bearer <service-token>

{
  "card": {
    "number": "4111111111111111",
    "exp_month": 12,
    "exp_year": 2027,
    "cvc": "123",
    "name": "Card Holder"
  }
}
```

Response:
```json
{
  "id": "tok_abc123def456",
  "last4": "1111",
  "brand": "visa",
  "exp_month": 12,
  "exp_year": 2027,
  "fingerprint": "fp_a1b2c3d4e5"
}
```

### Detokenize (Restricted)

```http
POST /v1/detokenize
Content-Type: application/json
Authorization: Bearer <restricted-service-token>

{
  "token_id": "tok_abc123def456",
  "fields": ["number", "cvc"]
}
```

Response:
```json
{
  "number": "4111111111111111",
  "cvc": "123"
}
```

### List Tokens

```http
GET /v1/tokens?customer_id=cus_123
Authorization: Bearer <service-token>
```

### Delete Token

```http
DELETE /v1/tokens/tok_abc123def456
Authorization: Bearer <service-token>
```

## Go Client SDK

```go
import "github.com/hanzoai/vault/client"

vc := client.New(client.Config{
    Endpoint: "http://vault.hanzo.svc:8080",
    Token:    os.Getenv("VAULT_SERVICE_TOKEN"),
})

// Tokenize a card
token, err := vc.Tokenize(ctx, client.Card{
    Number:   "4111111111111111",
    ExpMonth: 12,
    ExpYear:  2027,
    CVC:      "123",
})
// token.ID = "tok_abc123" (safe to store anywhere)
// token.Last4 = "1111"
// token.Brand = "visa"

// Detokenize (restricted — requires elevated permissions)
card, err := vc.Detokenize(ctx, "tok_abc123", client.DetokenizeOpts{
    Fields: []string{"number", "cvc"},
    Reason: "payment_processing",  // Audit trail
})

// List customer's tokens
tokens, err := vc.ListTokens(ctx, client.ListOpts{
    CustomerID: "cus_123",
})

// Delete token
err = vc.DeleteToken(ctx, "tok_abc123")
```

## Token Properties

| Property | Description |
|----------|-------------|
| `id` | Unique token ID (`tok_*`) |
| `last4` | Last 4 digits of card |
| `brand` | Card brand (visa, mastercard, amex) |
| `exp_month` | Expiration month |
| `exp_year` | Expiration year |
| `fingerprint` | Unique card fingerprint for dedup |
| `created_at` | Token creation timestamp |
| `customer_id` | Associated customer (optional) |

## Security

- **CRITICAL**: Never store raw card numbers. Always use Vault tokens.
- **CRITICAL**: Always hash passwords — NEVER store in plaintext.
- **CVC**: Stored temporarily (max 2 hours) then auto-deleted per PCI rules
- **Encryption**: AES-256-GCM with KMS-managed keys
- **Access control**: Service tokens scoped to tokenize-only or detokenize
- **Audit log**: Every tokenize/detokenize call logged with timestamp, caller, reason

## Deployment

```yaml
# K8s deployment (internal only)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vault
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: vault
          image: ghcr.io/hanzoai/vault:latest
          ports:
            - containerPort: 8080
            - containerPort: 8443
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: vault-secrets
                  key: database-url
            - name: KMS_ENDPOINT
              value: "http://kms.hanzo.svc:8080"
---
# No Ingress — internal only
apiVersion: v1
kind: Service
metadata:
  name: vault
  namespace: hanzo
spec:
  type: ClusterIP  # Internal only
  ports:
    - port: 8080
      name: http
    - port: 8443
      name: grpc
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Tokenize fails | KMS unreachable | Check KMS connection |
| Detokenize denied | Insufficient permissions | Use restricted token |
| Token not found | Wrong ID or deleted | Check token ID format |
| Slow tokenize | HSM latency | Check KMS health |

## Related Skills

- `hanzo/hanzo-kms.md` - Key management (Vault uses KMS for encryption keys)
- `hanzo/hanzo-commerce-api.md` - Payment processing (uses Vault for card tokenization)
- `hanzo/hanzo-contracts.md` - Smart contracts (crypto payments, not card)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: pci, tokenization, payments, security, go
**Prerequisites**: PCI DSS concepts, Go, internal networking
