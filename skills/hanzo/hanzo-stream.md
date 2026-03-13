# Hanzo Stream - Kafka Wire Protocol Gateway

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-pubsub.md`, `hanzo/hanzo-kv.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Stream is a **stateless Kafka wire protocol gateway** that translates standard Kafka client requests into NATS JetStream operations against Hanzo PubSub. Go codebase (module `github.com/hanzoai/stream`), ships as a single `hanzo-stream` binary. Any Kafka producer, consumer, or CLI tool connects on port 9092 and works without code changes. All storage and replication is delegated to Hanzo PubSub (NATS JetStream). License: MIT.

### Why Hanzo Stream?

- **Zero code changes**: Standard Kafka clients work out of the box
- **Stateless**: No local storage, no Raft, no Serf -- pure protocol translation
- **Hanzo PubSub backed**: All durability and replication via NATS JetStream
- **Compression**: GZIP, Snappy, LZ4, ZSTD codec support
- **Lightweight**: Single Go binary, ~140KB source, minimal dependencies

### Tech Stack

- **Language**: Go 1.26
- **CLI**: spf13/cobra
- **Backend**: nats-io/nats.go (NATS JetStream client)
- **Compression**: klauspost/compress (zstd), pierrec/lz4, eapache/go-xerial-snappy
- **Image**: `ghcr.io/hanzoai/stream` (alpine base)
- **CI**: GitHub Actions (push to GHCR on main)

### OSS Base

Repo: `hanzoai/stream`. Default branch: `main`. Based on [MonKafka](https://github.com/cefboud/monkafka).

## When to use

- Kafka clients need to produce/consume against Hanzo PubSub without migration
- Bridging existing Kafka-based pipelines (analytics, logs) to NATS JetStream
- Insights pipeline ingestion via Kafka protocol
- Any scenario where Kafka wire compatibility is needed but NATS is the backend

## Hard requirements

1. **Hanzo PubSub** (NATS with JetStream enabled) running and accessible
2. **Port 9092** available for Kafka listener
3. **Port 9093** available for admin HTTP endpoint (optional, disable with `--admin-port 0`)

## Quick reference

| Item | Value |
|------|-------|
| Kafka Port | 9092 |
| Admin Port | 9093 |
| Go Module | `github.com/hanzoai/stream` |
| Go Version | 1.26 |
| Binary | `hanzo-stream` |
| Image | `ghcr.io/hanzoai/stream` |
| PubSub Default | `nats://localhost:4222` |
| License | MIT |
| Repo | `github.com/hanzoai/stream` |
| Default Branch | `main` |

## One-file quickstart

### Run locally

```bash
# Start NATS with JetStream
nats-server --jetstream

# Start Hanzo Stream
go run main.go --pubsub-url nats://localhost:4222 --port 9092

# Use standard Kafka CLI tools
kafka-topics.sh --create --topic test --bootstrap-server localhost:9092
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic test
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning
```

### Docker

```bash
docker run -d --name hanzo-stream \
  -p 9092:9092 -p 9093:9093 \
  ghcr.io/hanzoai/stream:latest \
  --pubsub-url nats://pubsub:4222 --host 0.0.0.0
```

## Core Concepts

### Architecture

```
Kafka Client ──TCP :9092──> Hanzo Stream (protocol translation) ──> Hanzo PubSub (NATS JetStream)
```

**Stateless gateway**: All state lives in Hanzo PubSub. Multiple Stream instances share the same PubSub cluster.

### Kafka-to-PubSub Mapping

| Kafka Concept | PubSub Equivalent |
|---|---|
| Topic `foo`, Partition N | Stream `kafka-foo-N`, Subject `kafka.foo.N` |
| Produce | `Publish("kafka.foo.0", recordBatchBytes)` -- seq = offset+1 |
| Fetch at offset | `GetMsg(streamName, offset+1)` (PubSub 1-based, Kafka 0-based) |
| Consumer group offsets | KV bucket `kafka-consumer-offsets`, key `{group}.{topic}.{partition}` |
| Create topic (N parts) | N calls to `AddStream()` |
| Metadata | `StreamInfo()` per partition stream |

### Offset Translation (Critical)

```
Kafka offset 0  <->  PubSub sequence 1
Kafka offset N  <->  PubSub sequence N+1
Produce: seq = Publish(); return seq - 1
Fetch:   msg = GetMsg(offset + 1)
```

### Supported Kafka API Keys

| API Key | Name | Implementation |
|---------|------|----------------|
| 0 | Produce | Hand-written decoder (performance) |
| 1 | Fetch | Hand-written decoder (performance) |
| 3 | Metadata | Reflection-based serde |
| 8 | OffsetCommit | KV-backed |
| 9 | OffsetFetch | KV-backed |
| 10 | FindCoordinator | Static response |
| 11 | JoinGroup | Static response |
| 18 | ApiVersions | Supported versions list |
| 19 | CreateTopics | Creates JetStream streams |
| 32 | DescribeConfigs | Static response |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pubsub-url` | `nats://localhost:4222` | Hanzo PubSub server URL |
| `--pubsub-creds` | - | PubSub credentials file |
| `--port` | 9092 | Kafka listener port |
| `--admin-port` | 9093 | Admin HTTP port (0 to disable) |
| `--host` | localhost | Advertised hostname |
| `--node-id` | 1 | Broker node ID |
| `--replicas` | 1 | JetStream stream replica count |
| `--storage` | file | JetStream storage type: file or memory |

### Directory Structure

```
stream/
  main.go              # CLI entry point (cobra)
  Dockerfile           # Multi-stage build (golang:1.26-alpine -> alpine:3.20)
  go.mod               # github.com/hanzoai/stream
  pubsub/              # NATS JetStream client wrapper
    client.go          # Connection + stream context
    streams.go         # Stream CRUD, publish, get message, list topics
    consumer.go        # KV-based consumer offset management
  protocol/            # Kafka wire protocol handlers
    broker.go          # TCP server, connection handling
    dispatcher.go      # API key -> handler routing
    produce.go         # Produce (API key 0)
    fetch.go           # Fetch (API key 1)
    metadata.go        # Metadata (API key 3)
    create_topic.go    # CreateTopics (API key 19)
    responses.go       # ListOffsets, OffsetCommit/Fetch, JoinGroup, etc.
    find_coordinator.go
    describe_configs.go
    api_versions.go
    admin.go           # Admin HTTP endpoints
    types.go           # Request/response struct definitions
    recordbatch.go     # Kafka record batch parsing
    error.go           # Kafka error codes
  serde/               # Kafka protocol serialization (reflection-based)
  compress/            # GZIP, Snappy, LZ4, ZSTD codecs
  logging/             # Simple log levels
  utils/               # Time utilities
  types/               # Shared types (Config, Request, Record, RecordBatch)
  test/                # E2E and cluster tests
  .github/workflows/   # CI (GHCR push)
```

### Deployment (hanzo-k8s)

| Deployment | Service | Purpose |
|-----------|---------|---------|
| `insights-kafka` | `insights-kafka:9092` | Dedicated to Insights pipeline |
| `stream` | `stream:9092` | General purpose |

Both connect to `pubsub.hanzo.svc:4222` (NATS with JetStream, 20Gi PVC).

### Testing

```bash
# Requires Kafka CLI tools
export KAFKA_BIN_DIR=/path/to/kafka_2.13-3.9.0/bin
go test -v ./...
```

- `test/e2e/` -- E2E tests using Kafka CLI binaries
- `test/cluster/` -- Multi-instance tests (two gateways sharing same PubSub)

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused on 9092 | Stream not running or wrong host | Check `hanzo-stream` process, verify `--host` matches advertised address |
| Produce errors | PubSub not reachable | Verify `--pubsub-url` connectivity, check NATS is running with `--jetstream` |
| Offset mismatch | Off-by-one in custom client | Remember: Kafka offset N = PubSub sequence N+1 |
| Consumer group errors | Limited group support | Only basic offset commit/fetch; no full rebalancing protocol |
| Topic not found | JetStream stream not created | Create topic first via `kafka-topics.sh --create` |

## Related Skills

- `hanzo/hanzo-pubsub.md` - NATS JetStream (underlying message broker)
- `hanzo/hanzo-kv.md` - Valkey/Redis (sibling infrastructure service)
- `hanzo/hanzo-universe.md` - K8s infrastructure where Stream runs
- `hanzo/hanzo-insights.md` - Analytics pipeline (primary consumer)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kafka, nats, jetstream, streaming, event-driven, pubsub, message-broker
**Prerequisites**: NATS with JetStream, Go 1.26+ or Docker
