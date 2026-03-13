# Hanzo Vault - PCI-Compliant Card Tokenization

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kms.md`, `hanzo/hanzo-commerce-api.md`

## Overview

Hanzo Vault is a **PCI DSS-compliant card tokenization service** (CDE - Cardholder Data Environment). Securely tokenize, store, and detokenize payment card data without handling raw card numbers in your application.

## When to use

- Processing credit/debit card payments
- PCI DSS compliance requirements
- Tokenizing sensitive payment data
- Building e-commerce payment flows

## Quick reference

| Item | Value |
|------|-------|
| API | Internal service (not public) |
| Tokenize | `POST /v1/tokens` |
| Detokenize | `POST /v1/detokenize` |
| Tech | Go |
| Repo | `github.com/hanzoai/vault` |

## One-file quickstart

```go
import "github.com/hanzoai/vault/client"

vc := client.New("http://vault.hanzo.svc:8080")

// Tokenize a card
token, _ := vc.Tokenize(ctx, client.Card{
    Number: "4111111111111111",
    ExpMonth: 12,
    ExpYear: 2027,
    CVC: "123",
})
// token.ID = "tok_abc123" (safe to store)
// token.Last4 = "1111"

// Detokenize (restricted access)
card, _ := vc.Detokenize(ctx, "tok_abc123")
```

**CRITICAL**: Never store raw card numbers. Always hash passwords. Vault handles PCI scope isolation.

## Related Skills

- `hanzo/hanzo-kms.md` - Secret management
- `hanzo/hanzo-commerce-api.md` - Payment processing

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Prerequisites**: PCI DSS concepts, Go
