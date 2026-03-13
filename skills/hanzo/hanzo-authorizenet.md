# Hanzo AuthorizeNet-Go - Authorize.net Payment Gateway Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-billing.md`

## Overview

AuthorizeNet-Go is a **Go client library for the Authorize.net XML/JSON API**. It covers three major API surfaces: **AIM** (Advanced Integration Method) for payment transactions, **CIM** (Customer Information Manager) for stored payment profiles, and **ARB** (Automated Recurring Billing) for subscriptions. Also includes transaction reporting, fraud management, and batch settlement queries. Zero external dependencies -- standard library only (`net/http`, `encoding/json`).

### Why AuthorizeNet-Go?

- **Full AIM/CIM/ARB coverage**: Charge, auth-only, capture, void, refund, customer profiles, subscriptions
- **Zero dependencies**: Pure Go standard library
- **Client struct pattern**: Instantiate with `New(name, key, testMode)`, all methods hang off `*Client`
- **Sandbox support**: Toggle between test and production endpoints
- **Custom HTTP client**: Inject your own `*http.Client` (useful for App Engine, proxies)

### Tech Stack

- **Language**: Go (pre-modules, no go.mod -- use `go get`)
- **Package**: `github.com/hanzoai/authorizenet-go`
- **Dependencies**: None (standard library only)
- **CI**: Travis CI (Go 1.8)
- **License**: MIT
- **Upstream**: Fork of `hunterlong/AuthorizeCIM`

### OSS Base

Repo: `hanzoai/authorizenet-go` (3 stars). Default branch: `master`.

## When to use

- Processing credit card payments (charge, auth-only, capture, void, refund)
- Storing customer payment profiles for repeat billing (CIM)
- Creating recurring subscriptions (ARB)
- Querying settled/unsettled batches and transaction details
- Managing fraud holds (approve/decline held transactions)
- Any Go service that needs Authorize.net integration

## Hard requirements

1. **Authorize.net API credentials** (API Login ID + Transaction Key)
2. **Sandbox account** for testing: https://developer.authorize.net/hello_world/sandbox/
3. **Go 1.8+** (no modules required, uses `go get`)

## Quick reference

| Item | Value |
|------|-------|
| Package | `github.com/hanzoai/authorizenet-go` |
| Go version | 1.8+ (no go.mod) |
| API protocol | JSON over HTTPS (Authorize.net XML/JSON API v1) |
| Test endpoint | `https://apitest.authorize.net/xml/v1/request.api` |
| Live endpoint | `https://api.authorize.net/xml/v1/request.api` |
| Default branch | `master` |
| License | MIT |
| Repo | `github.com/hanzoai/authorizenet-go` |

## One-file quickstart

```go
package main

import (
    "fmt"
    "os"

    authorizenet "github.com/hanzoai/authorizenet-go"
)

func main() {
    client := authorizenet.New(
        os.Getenv("AUTHORIZENET_API_NAME"),
        os.Getenv("AUTHORIZENET_API_KEY"),
        true, // testMode
    )

    // Verify connection
    connected, err := client.IsConnected()
    if err != nil || !connected {
        fmt.Println("Failed to connect:", err)
        os.Exit(1)
    }

    // Charge a card
    txn := authorizenet.NewTransaction{
        Amount: "15.90",
        CreditCard: authorizenet.CreditCard{
            CardNumber:     "4007000000027",
            ExpirationDate: "10/27",
            CardCode:       "123",
        },
        BillTo: &authorizenet.BillTo{
            FirstName: "Jane",
            LastName:  "Doe",
            Address:   "123 Main St",
            City:      "Los Angeles",
            State:     "CA",
            Zip:       "90001",
            Country:   "USA",
        },
    }

    res, err := txn.Charge(*client)
    if err != nil {
        fmt.Println("Error:", err)
        os.Exit(1)
    }

    if res.Approved() {
        fmt.Println("Approved! Transaction ID:", res.TransactionID())
    } else {
        fmt.Println("Declined:", res.Response.Errors)
    }
}
```

## Core Concepts

### Architecture

```
authorizenet.New(name, key, testMode) -> *Client
    |
    |-- Client.IsConnected()
    |-- Client.GetMerchantDetails()
    |-- Client.GetProfileIds()
    |-- Client.GetPaymentProfileIds()
    |-- Client.SubscriptionList(search)
    |-- Client.UnsettledBatchList()
    |-- Client.UnSettledBatch()
    |
    NewTransaction (payment operations)
    |-- txn.Charge(client)           # authCaptureTransaction
    |-- txn.AuthOnly(client)         # authOnlyTransaction
    |-- txn.Refund(client)           # refundTransaction
    |-- txn.ChargeProfile(customer, client)
    |
    PreviousTransaction (post-auth operations)
    |-- prev.Capture(client)         # priorAuthCaptureTransaction
    |-- prev.Void(client)            # voidTransaction
    |-- prev.Approve(client)         # approve held transaction
    |-- prev.Decline(client)         # decline held transaction
    |-- prev.Info(client)            # get transaction details
    |
    Profile (CIM customer profiles)
    |-- profile.CreateProfile(client)
    |-- profile.UpdateProfile(client)
    |-- profile.UpdatePaymentProfile(client)
    |-- profile.CreateShipping(client)
    |-- profile.UpdateShippingProfile(client)
    |
    Customer (CIM queries)
    |-- customer.Info(client)
    |-- customer.Validate(client)
    |-- customer.DeleteProfile(client)
    |-- customer.DeletePaymentProfile(client)
    |-- customer.DeleteShippingProfile(client)
    |
    Subscription (ARB)
    |-- sub.Charge(client)
    |-- sub.Update(client)
    |
    SetSubscription (ARB queries)
    |-- setSub.Info(client)
    |-- setSub.Status(client)
    |-- setSub.Cancel(client)
    |
    Range (reporting)
    |-- range.SettledBatch(client)
    |-- range.Transactions(client)
    |-- range.Statistics(client)
```

### Client Initialization

```go
// Test mode (sandbox)
client := authorizenet.New("apiLoginId", "transactionKey", true)

// Production mode
client := authorizenet.New("apiLoginId", "transactionKey", false)

// Custom HTTP client (e.g., for App Engine)
client.SetHTTPClient(customHTTPClient)

// Verbose logging
client.Verbose = true
```

### Response Checking

All responses embed `MessagesResponse` which provides:

```go
res.Ok()           // bool - ResultCode == "Ok"
res.Approved()     // bool - ResponseCode == "1" or "4" (held)
res.Held()         // bool - ResponseCode == "4"
res.TransactionID() // string
res.AVS()          // AVS struct with avsResultCode, cvvResultCode, cavvResultCode
res.ErrorMessage() // string - first error message
res.Message()      // string - first message text
```

### Interval Helpers for Subscriptions

```go
authorizenet.IntervalWeekly()      // every 7 days
authorizenet.IntervalMonthly()     // every 1 month
authorizenet.IntervalQuarterly()   // every 3 months
authorizenet.IntervalYearly()      // every 365 days
authorizenet.IntervalDays("15")    // every N days
authorizenet.IntervalMonths("6")   // every N months
```

### Time Helpers for Reporting

```go
authorizenet.Now()          // current UTC time
authorizenet.LastWeek()     // 1 day ago (note: misnamed, actually yesterday)
authorizenet.LastMonth()    // 1 month ago
authorizenet.LastYear()     // 1 year ago
authorizenet.CurrentDate()  // "2006-01-02" formatted string
```

## Directory structure

```
github.com/hanzoai/authorizenet-go/
    authorizenet.go               # Client struct, New(), SendRequest(), SetHTTPClient(), AVS
    payment_transactions.go       # NewTransaction (Charge, AuthOnly, Refund), PreviousTransaction
                                  #   (Void, Capture), all payment types and request/response structs
    transaction_responses.go      # Response helpers: Approved(), Ok(), Held(), TransactionID(), AVS()
    customer_profile.go           # CIM: Profile CRUD, Customer CRUD, PaymentProfile, ShippingProfile
    recurring_billing.go          # ARB: Subscription create/update/cancel/status/list
    transaction_reporting.go      # Reporting: settled batches, unsettled, transaction details,
                                  #   batch statistics, merchant details
    fraud_management.go           # Unsettled batch list, approve/decline held transactions
    time_references.go            # Time and interval helper functions
    customer_profile_test.go      # CIM integration tests
    payment_transactions_test.go  # Payment integration tests
    recurring_billing_test.go     # ARB integration tests
    transaction_reporting_test.go # Reporting integration tests
    fraud_management_test.go      # Fraud management tests
    xcleanup_test.go              # Test cleanup (runs last due to x prefix)
    examples/
        new_transaction/
            main.go               # Example: charge + void
    .travis.yml                   # CI config (Go 1.8)
    .codeclimate.yml              # Code climate config
    LICENSE                       # MIT
    README.md                     # Full API documentation with examples
```

## API Coverage

| Area | Operations | Status |
|------|-----------|--------|
| **AIM: Payments** | Charge, AuthOnly, Capture, Void, Refund, ChargeProfile | Complete |
| **CIM: Profiles** | Create/Get/Update/Delete Customer Profile | Complete |
| **CIM: Payment Profiles** | Create/Get/Update/Delete/Validate Payment Profile | Complete |
| **CIM: Shipping** | Create/Get/Update/Delete Shipping Address | Complete |
| **ARB: Subscriptions** | Create/Update/Cancel/Status/Info/List | Complete |
| **Reporting** | Settled Batches, Unsettled, Transaction List, Transaction Details, Batch Statistics, Merchant Details | Complete |
| **Fraud** | Unsettled Batch List, Approve/Decline Held Transactions | Complete |
| **Not Implemented** | Bank Account (debit/credit), Split Tender, Token Card, Accept Payment, Hosted Payment Page | Stubs only |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "E00007: User authentication failed" | Wrong API credentials | Verify API Login ID and Transaction Key |
| "E00003: Root element is missing" | Malformed JSON | Check struct field tags, ensure `json.Marshal` succeeds |
| Tests fail with duplicate transaction | Sandbox dedup | MailChimp sandbox rejects identical transactions within 2 min |
| `nil pointer` on response check | Error not checked before accessing response fields | Always check `err != nil` before using response |
| BOM in response | Authorize.net returns UTF-8 BOM | Library strips `\xef\xbb\xbf` prefix automatically |
| Custom HTTP client needed | App Engine / proxy environments | Use `client.SetHTTPClient(httpClient)` |

## Related Skills

- `hanzo/hanzo-commerce.md` - E-commerce platform (payment processing)
- `hanzo/hanzo-billing.md` - Billing and subscription management
- `hanzo/hanzo-commerce-api.md` - Commerce API

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: payments, authorize-net, credit-card, subscriptions, go, api-client
**Prerequisites**: Go 1.8+, Authorize.net sandbox or production credentials
