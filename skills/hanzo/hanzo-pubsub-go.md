# Hanzo PubSub Go SDK - NATS Client for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kv-go.md`, `hanzo/go-sdk.md`

## Overview

Go client library for Hanzo PubSub (NATS compatible). Fork of `nats-io/nats.go` with module path rewritten to `github.com/hanzoai/pubsub-go`. Provides publish/subscribe messaging, request/reply, JetStream persistence, key-value store, object store, and microservice framework.

### OSS Base

Fork of **nats.go** (nats-io/nats.go). Repo: `hanzoai/pubsub-go`, branch: `main`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/pubsub-go` |
| Package | `nats` |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/pubsub-go` |
| Branch | `main` |
| License | Apache-2.0 |
| Protocol | NATS protocol |

## Installation

```bash
go get github.com/hanzoai/pubsub-go@latest
```

## Quick start

```go
package main

import (
    "fmt"
    "time"

    "github.com/hanzoai/pubsub-go"
)

func main() {
    nc, err := nats.Connect(nats.DefaultURL)
    if err != nil {
        panic(err)
    }
    defer nc.Close()

    // Subscribe
    nc.Subscribe("greet", func(m *nats.Msg) {
        fmt.Printf("Received: %s\n", string(m.Data))
    })

    // Publish
    nc.Publish("greet", []byte("Hello World"))

    // Request/Reply
    nc.Subscribe("help", func(m *nats.Msg) {
        m.Respond([]byte("I can help"))
    })
    msg, _ := nc.Request("help", []byte("need help"), time.Second)
    fmt.Println(string(msg.Data))

    nc.Drain()
}
```

## JetStream

```go
import (
    "context"

    "github.com/hanzoai/pubsub-go"
    "github.com/hanzoai/pubsub-go/jetstream"
)

nc, _ := nats.Connect(nats.DefaultURL)
js, _ := jetstream.New(nc)

ctx := context.Background()

// Create stream
stream, _ := js.CreateStream(ctx, jetstream.StreamConfig{
    Name:     "orders",
    Subjects: []string{"orders.>"},
})

// Publish
js.Publish(ctx, "orders.new", []byte(`{"id": 1}`))

// Consume
cons, _ := stream.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable: "processor",
})
cc, _ := cons.Consume(func(msg jetstream.Msg) {
    fmt.Println(string(msg.Data()))
    msg.Ack()
})
defer cc.Stop()
```

## Key features

- Publish/Subscribe with wildcard subjects (`*`, `>`)
- Request/Reply pattern
- Queue groups (load balancing)
- JetStream persistence (streams, consumers, KV, object store)
- Multiple encoders (JSON, GOB, Protobuf)
- TLS and NKey/JWT authentication
- Auto-reconnect with backoff
- Drain mode for graceful shutdown
- Microservice framework (`micro` package)

## Sub-packages

| Package | Import | Purpose |
|---------|--------|---------|
| jetstream | `github.com/hanzoai/pubsub-go/jetstream` | JetStream API (streams, consumers, KV, object store) |
| micro | `github.com/hanzoai/pubsub-go/micro` | NATS microservice framework |
| encoders/protobuf | `github.com/hanzoai/pubsub-go/encoders/protobuf` | Protobuf encoder |

## Build and test

```bash
go build ./...
go test ./...
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: pubsub, nats, messaging, jetstream, go, client
**Prerequisites**: Go 1.26+
