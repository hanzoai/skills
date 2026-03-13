# Hanzo Vector Go SDK - Qdrant Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/go-sdk.md`, `hanzo/hanzo-search-go.md`

## Overview

Go client library for Hanzo Vector (Qdrant compatible). Fork of `qdrant/go-client` with module path rewritten to `github.com/hanzoai/vector-go`. Uses gRPC for communication. The main package is `qdrant` -- all types, client, and helpers live there.

### OSS Base

Fork of **go-client** (qdrant/go-client). Repo: `hanzoai/vector-go`, branch: `master`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/vector-go` |
| Package | `qdrant` |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/vector-go` |
| Branch | `master` |
| License | Apache-2.0 |
| Protocol | gRPC |
| Default port | 6334 |

## Installation

```bash
go get github.com/hanzoai/vector-go
```

## Quick start

```go
package main

import (
    "context"
    "fmt"

    "github.com/hanzoai/vector-go/qdrant"
)

func main() {
    client, err := qdrant.NewClient(&qdrant.Config{
        Host: "localhost",
        Port: 6334,
    })
    if err != nil {
        panic(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Create collection
    client.CreateCollection(ctx, &qdrant.CreateCollection{
        CollectionName: "docs",
        VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
            Size:     384,
            Distance: qdrant.Distance_Cosine,
        }),
    })

    // Upsert points
    client.Upsert(ctx, &qdrant.UpsertPoints{
        CollectionName: "docs",
        Points: []*qdrant.PointStruct{
            {
                Id:      qdrant.NewIDNum(1),
                Vectors: qdrant.NewVectors(0.1, 0.2, 0.3),
                Payload: qdrant.NewValueMap(map[string]any{"title": "hello"}),
            },
        },
    })

    // Search
    results, _ := client.Query(ctx, &qdrant.QueryPoints{
        CollectionName: "docs",
        Query:          qdrant.NewQuery(0.1, 0.2, 0.3),
        WithPayload:    qdrant.NewWithPayload(true),
    })
    fmt.Println(results)
}
```

## Key features

- Collection management (create, update, delete, list)
- Point upsert, delete, get, scroll, count
- Vector search (query) with filtering
- Payload filtering with match, range, geo conditions
- Snapshot management (create, list, delete)
- Connection pooling (configurable `PoolSize`, default 3, round-robin)
- TLS and API key authentication
- gRPC-based (protobuf generated types)

## Client configuration

```go
client, _ := qdrant.NewClient(&qdrant.Config{
    Host:   "xyz.cloud.qdrant.io",
    Port:   6334,
    APIKey: "<your-api-key>",
    UseTLS: true,
    // PoolSize: 3,          // gRPC connection pool size
    // TLSConfig: &tls.Config{},
    // GrpcOptions: []grpc.DialOption{},
})
```

## Low-level gRPC access

```go
// Direct gRPC service clients
collectionsClient := client.GetCollectionsClient()
pointsClient := client.GetPointsClient()
snapshotsClient := client.GetSnapshotsClient()
qdrantClient := client.GetQdrantClient()
conn := client.GetConnection() // raw *grpc.ClientConn
```

## Helper functions

```go
qdrant.NewIDNum(42)                    // numeric point ID
qdrant.NewIDUUID("uuid-string")       // UUID point ID
qdrant.NewVectors(0.1, 0.2, 0.3)     // dense vector
qdrant.NewVectorsConfig(params)       // collection vector config
qdrant.NewValueMap(map[string]any{})  // payload from map
qdrant.NewQuery(0.1, 0.2, 0.3)       // search query from floats
qdrant.NewWithPayload(true)           // include payload in results
qdrant.NewMatch("field", "value")     // filter condition
qdrant.PtrOf(value)                   // generic pointer helper
```

## Build and test

```bash
go build ./...
go test ./...
# Integration tests use testcontainers (Docker required)
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vector, qdrant, embeddings, similarity-search, go, client
**Prerequisites**: Go 1.26+
