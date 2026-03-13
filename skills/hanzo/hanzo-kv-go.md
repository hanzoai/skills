# Hanzo KV Go SDK - Redis/Valkey Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/go-sdk.md`, `hanzo/hanzo-cloud.md`

## Overview

Go client library for Hanzo KV (Redis/Valkey compatible). Fork of `redis/go-redis` v9 with module path rewritten to `github.com/hanzoai/kv-go/v9`. The Go package name is `redis` -- import the module but use `redis.NewClient()`.

### OSS Base

Fork of **go-redis** (redis/go-redis). Repo: `hanzoai/kv-go`, branch: `master`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/kv-go/v9` |
| Package | `redis` |
| Version | 9.18.0-beta.2 |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/kv-go` |
| Branch | `master` |
| License | BSD-2-Clause |
| Protocol | RESP2/RESP3 |

## Installation

```bash
go get github.com/hanzoai/kv-go/v9
```

## Quick start

```go
package main

import (
    "context"
    "fmt"

    "github.com/hanzoai/kv-go/v9"
)

func main() {
    rdb := redis.NewClient(&redis.Options{
        Addr:     "localhost:6379",
        Password: "",
        DB:       0,
    })
    defer rdb.Close()

    ctx := context.Background()

    err := rdb.Set(ctx, "key", "value", 0).Err()
    if err != nil {
        panic(err)
    }

    val, err := rdb.Get(ctx, "key").Result()
    if err != nil {
        panic(err)
    }
    fmt.Println("key", val)
}
```

## Key features

- Full Redis command coverage (strings, hashes, lists, sets, sorted sets, streams, HyperLogLog, geo, bitmaps, JSON, search, time series, probabilistic)
- Automatic connection pooling
- Pub/Sub
- Pipelines and transactions
- Lua scripting
- Redis Sentinel and Cluster support
- Streaming credentials provider (OAuth, Entra ID)
- OpenTelemetry instrumentation (`extra/redisotel`)
- Prometheus metrics (`extra/redisprometheus`)
- Vector set commands

## Client types

```go
// Standalone
rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})

// Sentinel (HA)
rdb := redis.NewFailoverClient(&redis.FailoverOptions{
    MasterName:    "mymaster",
    SentinelAddrs: []string{"localhost:26379"},
})

// Cluster
rdb := redis.NewClusterClient(&redis.ClusterOptions{
    Addrs: []string{"localhost:7000", "localhost:7001", "localhost:7002"},
})

// Universal (auto-detects standalone/sentinel/cluster)
rdb := redis.NewUniversalClient(&redis.UniversalOptions{
    Addrs: []string{"localhost:6379"},
})
```

## Extra modules

| Module | Path | Purpose |
|--------|------|---------|
| redisotel | `github.com/hanzoai/kv-go/extra/redisotel/v9` | OpenTelemetry tracing + metrics |
| redisprometheus | `github.com/hanzoai/kv-go/extra/redisprometheus/v9` | Prometheus collector |
| rediscmd | `github.com/hanzoai/kv-go/extra/rediscmd/v9` | Command string formatting |

## Build and test

```bash
go build ./...
go test ./...
make test  # runs via Docker
```

## Related Skills

- `hanzo/go-sdk.md` - Hanzo AI API client
- `hanzo/hanzo-cloud.md` - Cloud infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kv, redis, valkey, cache, go, client
**Prerequisites**: Go 1.26+
