# Hanzo Search Go SDK - Meilisearch Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/go-sdk.md`, `hanzo/hanzo-kv-go.md`

## Overview

Go client library for Hanzo Search (Meilisearch compatible). Fork of `meilisearch/meilisearch-go` with module path rewritten to `github.com/hanzoai/search-go`. The Go package name is `meilisearch`. Provides index management, document CRUD, search with filters/facets/highlighting, settings management, task monitoring, chat, and webhook support.

### OSS Base

Fork of **meilisearch-go** (meilisearch/meilisearch-go). Repo: `hanzoai/search-go`, branch: `main`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/search-go` |
| Package | `meilisearch` |
| Version | 0.36.1 |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/search-go` |
| Branch | `main` |
| License | MIT |
| Protocol | HTTP/REST |

## Installation

```bash
go get github.com/hanzoai/search-go
```

## Quick start

```go
package main

import (
    "fmt"

    search "github.com/hanzoai/search-go"
)

func main() {
    client := search.New("http://localhost:7700", search.WithAPIKey("your-api-key"))

    // Create index
    task, _ := client.CreateIndex(&search.IndexConfig{
        Uid:        "movies",
        PrimaryKey: "id",
    })
    fmt.Println("Task:", task.TaskUID)

    // Add documents
    index := client.Index("movies")
    docs := []map[string]interface{}{
        {"id": 1, "title": "Carol", "genres": []string{"Romance", "Drama"}},
        {"id": 2, "title": "Wonder Woman", "genres": []string{"Action"}},
    }
    task, _ = index.AddDocuments(docs, nil)

    // Search
    res, _ := index.Search("wonder", &search.SearchRequest{Limit: 10})
    fmt.Println(res.Hits)
}
```

## Key features

- Index creation, listing, updating, deletion
- Document add, update, delete (batch and single)
- Full-text search with filters, facets, highlighting, sorting
- Settings management (filterable, sortable, searchable attributes)
- Multi-search across multiple indexes
- Facet search
- Task management (async operations)
- Chat interface (streaming)
- Webhook support
- Content encoding (gzip, deflate, brotli)
- Configurable retries with backoff
- JWT-based tenant tokens

## Client options

```go
client := search.New("http://localhost:7700",
    search.WithAPIKey("your-api-key"),
    search.WithCustomClient(http.DefaultClient),
    search.WithContentEncoding(search.GzipEncoding, search.BestCompression),
    search.WithCustomRetries([]int{502, 503, 504}, 3),
)
```

| Option | Description |
|--------|-------------|
| `WithAPIKey` | API key for authentication |
| `WithCustomClient` | Custom `http.Client` |
| `WithCustomClientWithTLS` | Enable TLS |
| `WithContentEncoding` | Request/response encoding |
| `WithCustomRetries` | Retry by status code and max retries |
| `DisableRetries` | Disable automatic retries |

## Build and test

```bash
go build ./...
go test ./...
# Integration tests require a running Meilisearch instance
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: search, meilisearch, full-text, go, client
**Prerequisites**: Go 1.26+
