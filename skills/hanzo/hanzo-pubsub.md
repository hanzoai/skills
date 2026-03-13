# Hanzo PubSub - Event Streaming and Message Queue

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-storage.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo PubSub is a **high-performance messaging system** for pub/sub, persistent streams, and exactly-once delivery. It is a fork of NATS Server with Hanzo branding and a bundled Kafka-compatible Redpanda sidecar. Written in Go. Provides the messaging backbone for Hanzo infrastructure.

### Why Hanzo PubSub?

- **Pub/Sub Messaging**: Subject-based routing with wildcard subscriptions
- **JetStream**: Persistent streams with configurable retention and replay
- **Consumer Groups**: Scalable consumption with automatic load balancing
- **Exactly-Once Delivery**: Deduplication and acknowledgment semantics
- **Built-in Key-Value Store**: Distributed KV on top of JetStream
- **Built-in Object Store**: Large object storage via JetStream
- **Kafka Compatible**: Bundled Redpanda sidecar speaks Kafka wire protocol
- **Clustering**: Horizontal scaling with automatic failover via Raft

### Tech Stack

- **Language**: Go (module: `github.com/nats-io/nats-server/v2`)
- **Go Version**: 1.26
- **License**: Apache 2.0
- **Kafka Sidecar**: Redpanda v25.1.9

### OSS Base

Fork of [NATS Server](https://github.com/nats-io/nats-server). Repo: `hanzoai/pubsub`.

## When to use

- Inter-service messaging in Hanzo infrastructure
- Durable event streaming with replay capability
- Work queues with load-balanced consumers
- Real-time notifications and event-driven architectures
- Kafka-compatible streaming (via Redpanda sidecar)
- Lightweight distributed key-value or object storage

## Hard requirements

1. **Go 1.26+** to build from source
2. **Docker** for container deployment
3. **Persistent volume** for JetStream data durability

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/pubsub` |
| Module | `github.com/nats-io/nats-server/v2` (upstream path retained) |
| Go Version | 1.26 |
| License | Apache 2.0 |
| Client Port | 4222 |
| Monitoring Port | 8222 |
| Cluster Port | 6222 |
| WebSocket Port | 5222 |
| Kafka Port | 9092 (Redpanda sidecar) |
| Docker Image | `hanzoai/pubsub:latest` |
| Binary | `pubsub` (built from `main.go`) |
| Config | `/pubsub/conf/server.conf` |

## One-file quickstart

### Docker

```bash
docker run -d --name hanzo-pubsub \
  -p 4222:4222 \
  -p 8222:8222 \
  hanzoai/pubsub:latest
```

### Docker Compose

```yaml
# compose.yml
services:
  pubsub:
    image: hanzoai/pubsub:latest
    ports:
      - "4222:4222"   # Client connections
      - "8222:8222"   # HTTP monitoring
      - "6222:6222"   # Cluster routing
    volumes:
      - pubsub-data:/data
    command: ["--jetstream", "--store_dir=/data"]

  # Optional: Kafka-compatible sidecar
  kafka:
    build: kafka/
    ports:
      - "9092:9092"   # Kafka protocol
      - "8081:8081"   # Schema registry
      - "8082:8082"   # HTTP proxy
      - "9644:9644"   # Admin API

volumes:
  pubsub-data:
```

### Build from source

```bash
git clone https://github.com/hanzoai/pubsub.git
cd pubsub
go build -o pubsub .
./pubsub --jetstream --store_dir /tmp/pubsub-data
```

## Core Concepts

### Architecture

```
┌──────────────────────────────────────────────────┐
│                    Cluster                        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Node 1  │<-->│ Node 2  │<-->│ Node 3  │      │
│  │ :4222   │    │ :4222   │    │ :4222   │      │
│  └─────────┘    └─────────┘    └─────────┘      │
│       │              │              │            │
│       └──────────────┼──────────────┘            │
│                      │                           │
│              ┌───────┴───────┐                   │
│              │  JetStream    │                   │
│              │  (Streams,    │                   │
│              │   KV, ObjStore)│                  │
│              └───────────────┘                   │
└──────────────────────────────────────────────────┘

       ┌───────────────────────┐
       │  Redpanda Sidecar     │
       │  (Kafka :9092)        │
       │  (Schema Reg :8081)   │
       └───────────────────────┘
```

### Server Configuration

Default config at `docker/nats-server.conf`:

```conf
# Client port
port: 4222

# HTTP monitoring
monitor_port: 8222

# Clustering
cluster {
  port: 6222
  authorization {
    user: ruser
    password: T0pS3cr3t
    timeout: 2
  }
  routes = []
}
```

### Port Allocation

| Port | Protocol | Purpose |
|------|----------|---------|
| 4222 | TCP | Client connections (NATS protocol) |
| 8222 | HTTP | Monitoring and management |
| 6222 | TCP | Cluster routing between nodes |
| 5222 | WebSocket | WebSocket client connections |
| 9092 | TCP | Kafka protocol (Redpanda sidecar) |
| 8081 | HTTP | Schema registry (Redpanda) |
| 8082 | HTTP | HTTP proxy (Redpanda) |
| 9644 | HTTP | Redpanda admin API |

### Directory Structure

```
pubsub/
  main.go              # Entry point (configures and runs server)
  server/              # Core server implementation
  conf/                # Configuration parser (lexer, parser)
  internal/
    antithesis/        # Deterministic testing
    fastrand/          # Fast random number generation
    ldap/              # LDAP authentication
    ocsp/              # OCSP stapling
    testhelper/        # Test utilities
  kafka/
    Dockerfile         # Redpanda (Kafka-compatible) sidecar
  logger/              # Logging
  docker/
    nats-server.conf   # Default server configuration
    Dockerfile.nightly # Nightly build
  test/                # Integration tests
  scripts/             # Build and CI scripts
```

### Client SDKs

| Language | Package |
|----------|---------|
| Go | `github.com/nats-io/nats.go` |
| Python | `nats-py` (`pip install nats-py`) |
| TypeScript | `nats` (`npm install nats`) |
| Rust | `nats` (`cargo add nats`) |

All NATS clients work directly with Hanzo PubSub (wire-compatible).

### Usage Examples

**Basic Pub/Sub (Go):**

```go
nc, _ := nats.Connect("nats://localhost:4222")
defer nc.Close()

// Subscribe
nc.Subscribe("events.>", func(msg *nats.Msg) {
    fmt.Printf("Received: %s\n", string(msg.Data))
})

// Publish
nc.Publish("events.user.created", []byte(`{"user_id":"123"}`))
```

**JetStream Persistent Streams (Go):**

```go
nc, _ := nats.Connect("nats://localhost:4222")
js, _ := nc.JetStream()

// Create stream
js.AddStream(&nats.StreamConfig{
    Name:     "ORDERS",
    Subjects: []string{"orders.*"},
})

// Publish
js.Publish("orders.new", []byte(`{"order_id":"abc123"}`))

// Durable consumer
sub, _ := js.PullSubscribe("orders.*", "processor")
msgs, _ := sub.Fetch(10)
for _, msg := range msgs {
    msg.Ack()
}
```

## Performance

- **Throughput**: 10M+ messages/second
- **Latency**: Sub-millisecond publish latency
- **Connections**: 100K+ concurrent connections per node

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| JetStream not enabled | Missing flag | Add `--jetstream` to command |
| Cluster routes not connecting | Auth mismatch | Verify `ruser:T0pS3cr3t` matches all nodes |
| Kafka clients can't connect | Redpanda not running | Start Kafka sidecar container |
| Data lost on restart | No persistent volume | Mount volume at `--store_dir` path |

## Related Skills

- `hanzo/hanzo-storage.md` - S3-compatible object storage (uses PubSub for event notifications)
- `hanzo/hanzo-platform.md` - PaaS deployment
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-database.md` - PostgreSQL database

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: pubsub, messaging, nats, kafka, streaming, events
**Prerequisites**: Go 1.26+, Docker
