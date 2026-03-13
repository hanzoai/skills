# Hanzo GoChimp3 - MailChimp API v3 Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-commerce-api.md`

## Overview

GoChimp3 is a **Go client library for the MailChimp API v3.0**. It provides typed Go structs and methods for managing audiences (lists), members, campaigns, automations, e-commerce data, templates, segments, webhooks, and batch operations. Single package, zero external dependencies beyond `net/http` and `encoding/json` from the standard library. Test dependency on `stretchr/testify`.

### Why GoChimp3?

- **Pure Go**: Standard library HTTP client with configurable timeout and transport
- **Full API v3 coverage**: Lists, members, campaigns, automations, e-commerce, templates, segments, webhooks, batch operations
- **Fluent API**: Chain methods on response objects (e.g., `list.CreateMember(req)`)
- **Typed errors**: `APIError` struct with status, type, title, detail, and field-level errors
- **Configurable**: Custom timeout, HTTP transport, debug mode with request/response dumping

### Tech Stack

- **Language**: Go (module requires Go 1.13+)
- **Module**: `github.com/hanzoai/gochimp3`
- **Dependencies**: `github.com/stretchr/testify v1.5.1` (test only)
- **CI**: Travis CI
- **License**: MIT

### OSS Base

Repo: `hanzoai/gochimp3` (68 stars). Default branch: `master`.

## When to use

- Adding/removing/updating MailChimp list subscribers from Go services
- Creating and sending email campaigns programmatically
- Syncing e-commerce order/product/customer data to MailChimp
- Managing automation workflows
- Batch subscribing members with update-or-create semantics

## Hard requirements

1. **MailChimp API key** (format: `<key>-<datacenter>`, e.g., `abc123-us6`)
2. **Go 1.13+** (uses Go modules)

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/gochimp3` |
| Go version | 1.13+ |
| API version | MailChimp API v3.0 |
| Auth | HTTP Basic Auth (API key) |
| Default branch | `master` |
| License | MIT |
| Repo | `github.com/hanzoai/gochimp3` |

## One-file quickstart

```go
package main

import (
    "fmt"
    "os"
    "time"

    "github.com/hanzoai/gochimp3"
)

func main() {
    client := gochimp3.New(os.Getenv("MAILCHIMP_API_KEY"))
    client.Timeout = 5 * time.Second

    // Get a list by ID
    list, err := client.GetList("YOUR_LIST_ID", nil)
    if err != nil {
        fmt.Println("Error:", err)
        os.Exit(1)
    }

    // Subscribe a member
    req := &gochimp3.MemberRequest{
        EmailAddress: "user@example.com",
        Status:       "subscribed",
        MergeFields: map[string]interface{}{
            "FNAME": "Jane",
            "LNAME": "Doe",
        },
    }

    member, err := list.CreateMember(req)
    if err != nil {
        fmt.Println("Subscribe failed:", err)
        os.Exit(1)
    }
    fmt.Println("Subscribed:", member.EmailAddress)
}
```

## Core Concepts

### Architecture

```
gochimp3.New(apiKey) -> *API
    |
    |-- API.GetLists()     -> *ListOfLists     (paginated)
    |-- API.GetList(id)    -> *ListResponse     (has sub-methods)
    |-- API.GetCampaigns() -> *ListOfCampaigns
    |-- API.CreateCampaign()
    |-- API.SendCampaign()
    |
    ListResponse (fluent sub-API)
    |-- list.GetMembers()
    |-- list.CreateMember()
    |-- list.UpdateMember()
    |-- list.AddOrUpdateMember()  (PUT - upsert)
    |-- list.DeleteMember()
    |-- list.DeleteMemberPermanent()
    |-- list.BatchSubscribeMembers()
    |-- list.GetInterestCategories()
    |-- list.GetMergeFields()
    |-- list.CreateMergeField()
    |
    Member (fluent sub-API)
    |-- member.GetActivity()
    |-- member.GetGoals()
    |-- member.GetNotes() / CreateNote() / UpdateNote()
    |-- member.GetTags() / UpdateTags()
```

### API Client

The `API` struct is the entry point. It extracts the datacenter from the API key using a regex and constructs the endpoint URL (`https://<dc>.api.mailchimp.com/3.0`). All requests use HTTP Basic Auth with the key.

```go
client := gochimp3.New("your-api-key-us6")
client.Timeout = 10 * time.Second    // optional
client.Debug = true                   // dumps request/response
client.Transport = customTransport    // optional custom transport
```

### Query Parameters

All list/get methods accept typed query parameter structs:

- `BasicQueryParams` -- Status, SortField, SortDirection, Fields, ExcludeFields
- `ExtendedQueryParams` -- adds Count and Offset for pagination
- `ListQueryParams` -- adds date filters and email filter
- `CampaignQueryParams` -- adds type, status, date filters, list/folder IDs

### Error Handling

API errors are returned as `*APIError` implementing the `error` interface:

```go
member, err := list.CreateMember(req)
if err != nil {
    if apiErr, ok := err.(*gochimp3.APIError); ok {
        fmt.Println("Status:", apiErr.Status)
        fmt.Println("Detail:", apiErr.Detail)
        for _, e := range apiErr.Errors {
            fmt.Println("Field:", e.Field, "Message:", e.Message)
        }
    }
}
```

### Member ID by Email

MailChimp uses MD5 hash of lowercase email as member ID. The library provides a helper:

```go
member := api.MemberForApiCalls("list-id", "user@example.com")
// member.ID is now the MD5 hash, ready for API calls
```

## Directory structure

```
github.com/hanzoai/gochimp3/
    api.go                  # API client, HTTP transport, request/response handling
    api_test.go             # Client tests
    common_types.go         # APIError, QueryParams, Link, Address, Contact, etc.
    lists.go                # Lists CRUD, abuse reports, activity, clients, growth history,
                            #   interest categories, interests, batch subscribe, merge fields
    members.go              # Members CRUD, activity, goals, notes, tags
    campaigns.go            # Campaigns CRUD, send test, send, content updates
    automations.go          # Automation workflows, emails, queues
    automation_workflows.go # Workflow-level operations
    ecommerce.go            # Stores, products, variants, orders, carts, customers
    segments.go             # List segments CRUD, batch modify
    templates.go            # Templates CRUD
    template_folders.go     # Template folder CRUD
    campaign_folders.go     # Campaign folder CRUD
    batches.go              # Batch operations
    webhooks.go             # Webhook CRUD on lists
    search.go               # Search members
    events.go               # Member events
    events_test.go          # Events tests
    root.go                 # API root info
    json.go                 # JSON helper (custom time parsing)
    go.mod                  # Module definition
    go.sum                  # Dependency checksums
```

## API Coverage

| Resource | Methods |
|----------|---------|
| Lists | GetLists, GetList, CreateList, UpdateList, DeleteList |
| Members | GetMembers, GetMember, CreateMember, UpdateMember, AddOrUpdateMember, DeleteMember, DeleteMemberPermanent, BatchSubscribeMembers |
| Campaigns | GetCampaigns, GetCampaign, CreateCampaign, UpdateCampaign, DeleteCampaign, SendTestEmail, SendCampaign, GetCampaignContent, UpdateCampaignContent |
| Automations | GetAutomations, GetAutomation, PauseAutomation, StartAutomation |
| E-commerce | Stores, Products, Variants, Orders, Carts, Customers (full CRUD) |
| Segments | GetSegments, GetSegment, CreateSegment, UpdateSegment, DeleteSegment, BatchModifySegment |
| Templates | GetTemplates, GetTemplate, CreateTemplate, UpdateTemplate, DeleteTemplate |
| Webhooks | GetWebhooks, GetWebhook, CreateWebhook, DeleteWebhook |
| Merge Fields | GetMergeFields, GetMergeField, CreateMergeField |
| Interest Categories | GetInterestCategories, GetInterestCategory, CreateInterestCategory, UpdateInterestCategory, DeleteInterestCategory |
| Tags | GetTags, UpdateTags (on members) |
| Search | SearchMembers |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No ID provided on list" | Using ListResponse without ID | Use `api.GetList(id)` or `api.NewListResponse(id)` |
| Wrong datacenter | API key format incorrect | Key must be `<key>-<dc>` (e.g., `abc-us6`) |
| 401 Unauthorized | Invalid API key | Verify key in MailChimp account settings |
| Timeout errors | Default no timeout | Set `client.Timeout = 10 * time.Second` |
| Rate limits (429) | Too many requests | MailChimp allows 10 concurrent connections; add backoff |

## Related Skills

- `hanzo/hanzo-commerce.md` - E-commerce platform (uses MailChimp for email marketing)
- `hanzo/hanzo-commerce-api.md` - Commerce API (subscriber sync)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: mailchimp, email, marketing, go, api-client
**Prerequisites**: Go 1.13+, MailChimp API key
