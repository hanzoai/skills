# Hanzo Vector - High-Performance Vector Search Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/hanzo-storage.md`, `hanzo/hanzo-llm-gateway.md`

## Overview

Hanzo Vector is a **high-performance vector search engine** for the Hanzo AI platform. It is a full fork of [Qdrant](https://github.com/qdrant/qdrant) v1.17.0, written in Rust, providing dense and sparse vector similarity search with filtering, payloads, quantization, distributed clustering, and GPU acceleration.

The crate is named `hanzo-vector` (package name in Cargo.toml), but the binary is still `qdrant` for wire-protocol compatibility with Qdrant clients.

### Why Hanzo Vector?

- **Drop-in Qdrant replacement**: Compatible with all Qdrant client SDKs (Python, JS, Rust, Go)
- **Dense + sparse vectors**: Hybrid search combining semantic embeddings and keyword matching
- **HNSW indexing**: Approximate nearest neighbor with SIMD acceleration
- **Vector quantization**: Scalar, product, and binary quantization for memory efficiency
- **GPU acceleration**: NVIDIA (Vulkan) and AMD (ROCm) support via `--features gpu`
- **Distributed mode**: Raft consensus, sharding, and replication
- **REST + gRPC APIs**: OpenAPI 3.0 schema and Protocol Buffer definitions
- **Write-ahead logging**: Crash-safe persistence with snapshot-based backup
- **Edge runtime**: Embedded mode via `lib/edge` with Python bindings

### Tech Stack

- **Language**: Rust (edition 2024, rust-version 1.92)
- **HTTP server**: actix-web with rustls TLS
- **gRPC server**: tonic (custom fork for qdrant compatibility)
- **Consensus**: Raft (via `raft` crate, prost codec)
- **Allocator**: jemalloc (tikv-jemallocator) on x86_64/aarch64
- **Profiling**: Pyroscope, tracing-tracy, console-subscriber
- **Build**: cargo-chef for Docker layer caching, mold linker

### OSS Base

Fork of [qdrant/qdrant](https://github.com/qdrant/qdrant). Repo: `hanzoai/vector`, branch `master` (not `main`).

## When to use

- Storing and searching AI embeddings (LLM, vision, audio)
- Semantic similarity search with metadata filtering
- Recommendation engines using vector similarity
- RAG (Retrieval-Augmented Generation) pipelines
- Hybrid search combining dense vectors with sparse (BM25-style) vectors
- High-throughput vector operations requiring GPU acceleration

## Hard requirements

1. **Rust 1.92+** for building from source
2. **Protobuf compiler** (`protoc`) for gRPC code generation
3. **clang + lld** for linking (or mold)
4. **Docker** for containerized deployment
5. **Vulkan drivers** if using GPU features (NVIDIA or AMD)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/vector` |
| Branch | `master` (code), `main` (stub placeholder) |
| Version | 1.17.0 |
| Language | Rust (edition 2024) |
| Binary | `qdrant` |
| REST port | 6333 |
| gRPC port | 6334 |
| P2P port | 6335 (cluster mode) |
| Docker image | `ghcr.io/hanzoai/vector:latest` |
| License | Apache-2.0 |
| Config | `config/config.yaml` |
| OpenAPI | `openapi/` |
| gRPC protos | `lib/api/src/grpc/proto/` |

## One-file quickstart

### Docker (single node)

```bash
docker run -p 6333:6333 -p 6334:6334 ghcr.io/hanzoai/vector:latest
```

### Docker with persistent storage

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/data:/qdrant/storage \
  -v $(pwd)/snapshots:/qdrant/snapshots \
  ghcr.io/hanzoai/vector:latest
```

### Docker with custom config

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/config.yaml:/qdrant/config/production.yaml \
  -v $(pwd)/data:/qdrant/storage \
  ghcr.io/hanzoai/vector:latest
```

### Build from source

```bash
cargo build --release --bin qdrant
./target/release/qdrant --config-path config/config.yaml
```

### Build with GPU support

```bash
cargo build --release --bin qdrant --features gpu
```

## Core Concepts

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Hanzo Vector Node                      │
├──────────────┬───────────────┬───────────────────────────┤
│  REST API    │  gRPC API     │  Web UI                   │
│  (actix-web) │  (tonic)      │  (static files)           │
│  :6333       │  :6334        │  :6333/dashboard          │
├──────────────┴───────────────┴───────────────────────────┤
│                    Storage Layer                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Collection                                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │ │
│  │  │ Shard 0  │ │ Shard 1  │ │ Shard N  │           │ │
│  │  │┌────────┐│ │┌────────┐│ │┌────────┐│           │ │
│  │  ││Segment ││ ││Segment ││ ││Segment ││           │ │
│  │  ││ HNSW   ││ ││ HNSW   ││ ││ HNSW   ││           │ │
│  │  ││ Index   ││ ││ Index   ││ ││ Index   ││           │ │
│  │  │└────────┘│ │└────────┘│ │└────────┘│           │ │
│  │  └──────────┘ └──────────┘ └──────────┘           │ │
│  └─────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  WAL (Write-Ahead Log)  │  Snapshots  │  Gridstore      │
├──────────────────────────────────────────────────────────┤
│  Raft Consensus (P2P :6335) -- cluster mode only         │
└──────────────────────────────────────────────────────────┘
```

### Data Model

- **Collection**: A named group of points with a defined vector configuration (dimension, distance metric)
- **Point**: A record with an ID (u64 or UUID), one or more named vectors, and an optional JSON payload
- **Payload**: Arbitrary JSON metadata attached to a point, indexable for filtering
- **Shard**: A horizontal partition of a collection, distributable across cluster nodes
- **Segment**: An immutable storage unit within a shard containing vectors + indexes

### Distance Metrics

- **Cosine** -- angular similarity (default for embeddings)
- **Euclidean** -- L2 distance
- **Dot** -- inner product
- **Manhattan** -- L1 distance

### Quantization

- **Scalar** -- 8-bit quantization (4x memory reduction)
- **Product** -- PQ compression (up to 64x reduction)
- **Binary** -- 1-bit per dimension (32x reduction, fastest)

## Workspace Structure

```
hanzoai/vector (master branch)
├── Cargo.toml              # Root workspace, package: hanzo-vector v1.17.0
├── Cargo.lock
├── Dockerfile              # Multi-stage: chef → planner → builder → runtime
├── config/
│   ├── config.yaml         # Full reference config with all options
│   ├── production.yaml     # Production overrides
│   └── development.yaml    # Dev overrides
├── src/                    # Main binary source
│   ├── main.rs             # Entry point, CLI args, server startup
│   ├── consensus.rs        # Raft consensus implementation
│   ├── settings.rs         # Config loading and validation
│   ├── snapshots.rs        # Snapshot management
│   ├── startup.rs          # Server initialization
│   ├── greeting.rs         # Startup banner
│   ├── actix/              # REST API handlers (actix-web)
│   │   ├── api/            # Route handlers per resource
│   │   │   ├── collections_api.rs
│   │   │   ├── search_api.rs
│   │   │   ├── query_api.rs
│   │   │   ├── recommend_api.rs
│   │   │   ├── retrieve_api.rs
│   │   │   ├── update_api.rs
│   │   │   ├── count_api.rs
│   │   │   ├── cluster_api.rs
│   │   │   ├── snapshot_api.rs
│   │   │   ├── shards_api.rs
│   │   │   ├── local_shard_api.rs
│   │   │   ├── discovery_api.rs
│   │   │   ├── facet_api.rs
│   │   │   └── service_api.rs
│   │   ├── auth.rs         # API key + JWT RBAC auth
│   │   └── mod.rs          # Actix server config
│   ├── tonic/              # gRPC service implementations
│   ├── tracing/            # Tracing/logging setup
│   └── migrations/         # Storage format migrations
├── lib/                    # Workspace crates
│   ├── api/                # API types (REST + gRPC)
│   │   ├── src/grpc/proto/ # Protobuf definitions
│   │   │   ├── qdrant.proto
│   │   │   ├── collections.proto
│   │   │   ├── points.proto
│   │   │   ├── collections_service.proto
│   │   │   ├── points_service.proto
│   │   │   ├── snapshots_service.proto
│   │   │   ├── shard_snapshots_service.proto
│   │   │   ├── raft_service.proto
│   │   │   └── health_check.proto
│   │   └── src/rest/       # REST API types
│   ├── collection/         # Collection management (CRUD, optimization)
│   ├── segment/            # Core storage engine
│   │   └── src/
│   │       ├── index/          # HNSW index implementation
│   │       ├── vector_storage/ # Vector data storage
│   │       ├── payload_storage/# Payload storage + indexing
│   │       ├── id_tracker/     # Point ID mapping
│   │       ├── spaces/         # Distance metric implementations
│   │       ├── data_types/     # Vector type definitions
│   │       └── types.rs        # Core type definitions (170KB)
│   ├── storage/            # Storage orchestration layer
│   ├── shard/              # Shard management and transfer
│   ├── sparse/             # Sparse vector support
│   ├── quantization/       # Quantization (scalar/product/binary, C++ FFI)
│   ├── posting_list/       # Inverted index for sparse vectors
│   ├── gridstore/          # Column-oriented storage for payloads
│   ├── gpu/                # GPU acceleration (Vulkan, NVIDIA/AMD)
│   ├── edge/               # Embedded mode (no server, direct API)
│   │   └── python/         # Python bindings for edge mode
│   ├── trififo/            # Lock-free triple-buffer FIFO
│   ├── macros/             # Proc macros
│   └── common/             # Shared utilities
│       ├── common/         # Core common types
│       ├── cancel/         # Cancellation tokens
│       ├── dataset/        # Dataset loading utilities
│       └── issues/         # Issue tracking/reporting
├── openapi/                # OpenAPI 3.0 schema (ytt templates)
│   ├── openapi-main.ytt.yaml
│   ├── openapi-collections.ytt.yaml
│   ├── openapi-points.ytt.yaml
│   ├── openapi-service.ytt.yaml
│   ├── openapi-snapshots.ytt.yaml
│   ├── openapi-shard-snapshots.ytt.yaml
│   ├── openapi-shards.ytt.yaml
│   ├── openapi-cluster.ytt.yaml
│   └── schemas/
├── tests/                  # Integration tests
│   ├── basic_api_test.sh
│   ├── basic_grpc_test.sh
│   ├── basic_sparse_test.sh
│   ├── consensus_tests/
│   ├── e2e_tests/
│   └── openapi/
├── tools/                  # Build and dev scripts
│   ├── entrypoint.sh       # Docker entrypoint
│   ├── sync-web-ui.sh      # Download web dashboard
│   ├── compose/            # Docker compose files for cluster
│   └── schema2openapi/     # Schema generation tooling
└── pkg/
    └── appimage/           # AppImage packaging
```

## REST API Endpoints

All endpoints are on port 6333 by default. Auth via `api-key` header or JWT bearer token.

### Collections

```bash
# List collections
GET /collections

# Create collection
PUT /collections/{name}
{
  "vectors": { "size": 1536, "distance": "Cosine" },
  "optimizers_config": { "indexing_threshold": 10000 },
  "replication_factor": 2
}

# Get collection info
GET /collections/{name}

# Delete collection
DELETE /collections/{name}

# Update collection params
PATCH /collections/{name}
```

### Points

```bash
# Upsert points
PUT /collections/{name}/points
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, ...],
      "payload": { "city": "Berlin", "category": "tech" }
    }
  ]
}

# Get points by ID
POST /collections/{name}/points
{ "ids": [1, 2, 3], "with_payload": true, "with_vector": true }

# Delete points
POST /collections/{name}/points/delete
{ "points": [1, 2, 3] }

# Count points
POST /collections/{name}/points/count
{ "filter": { "must": [{ "key": "city", "match": { "value": "Berlin" } }] } }
```

### Search

```bash
# Vector search
POST /collections/{name}/points/search
{
  "vector": [0.1, 0.2, ...],
  "limit": 10,
  "filter": {
    "must": [{ "key": "city", "match": { "value": "Berlin" } }]
  },
  "with_payload": true
}

# Batch search
POST /collections/{name}/points/search/batch
{ "searches": [...] }

# Query (universal search endpoint)
POST /collections/{name}/points/query
{
  "query": [0.1, 0.2, ...],
  "limit": 10,
  "filter": { ... }
}

# Recommend
POST /collections/{name}/points/recommend
{
  "positive": [1, 2],
  "negative": [3],
  "limit": 10
}

# Discover (context-based search)
POST /collections/{name}/points/discover
```

### Snapshots

```bash
# Create snapshot
POST /collections/{name}/snapshots

# List snapshots
GET /collections/{name}/snapshots

# Download snapshot
GET /collections/{name}/snapshots/{snapshot_name}

# Full storage snapshot
POST /snapshots
```

### Cluster

```bash
# Cluster info
GET /cluster

# Collection cluster info
GET /collections/{name}/cluster

# Move shard
POST /collections/{name}/cluster
{ "move_shard": { "shard_id": 0, "from_peer_id": 1, "to_peer_id": 2 } }
```

### Service

```bash
# Health check
GET /healthz

# Readiness
GET /readyz

# Telemetry
GET /telemetry

# Metrics (Prometheus)
GET /metrics

# Locks
GET /locks
POST /locks
```

## gRPC Services

Port 6334 by default. Proto files in `lib/api/src/grpc/proto/`.

| Service | Proto | Description |
|---------|-------|-------------|
| `Collections` | `collections_service.proto` | CRUD for collections |
| `Points` | `points_service.proto` | CRUD, search, recommend, query for points |
| `Snapshots` | `snapshots_service.proto` | Snapshot management |
| `ShardSnapshots` | `shard_snapshots_service.proto` | Per-shard snapshot operations |
| `Qdrant` | `qdrant.proto` | Health check, version info |
| `QdrantInternal` | `qdrant_internal_service.proto` | Internal cluster operations |
| `Raft` | `raft_service.proto` | Raft consensus messages |

## Configuration

The config file (`config/config.yaml`) supports YAML with environment variable overrides. Key sections:

```yaml
storage:
  storage_path: ./storage       # Data directory
  snapshots_path: ./snapshots   # Snapshot directory
  on_disk_payload: true         # Keep payloads on disk (saves RAM)
  wal:
    wal_capacity_mb: 32         # WAL segment size
  hnsw_index:
    m: 16                       # HNSW edges per node
    ef_construct: 100           # Build-time neighbors
    full_scan_threshold_kb: 10000
    on_disk: false              # HNSW in RAM or disk
  optimizers:
    indexing_threshold_kb: 10000
    flush_interval_sec: 5
  collection:
    replication_factor: 1
    write_consistency_factor: 1

service:
  http_port: 6333
  grpc_port: 6334
  host: 0.0.0.0
  max_request_size_mb: 32
  enable_cors: true
  enable_tls: false
  # api_key: your_secret_key
  # jwt_rbac: true

cluster:
  enabled: false
  p2p:
    port: 6335
  consensus:
    tick_period_ms: 100

telemetry_disabled: false
```

Environment variable override pattern: `QDRANT__SERVICE__HTTP_PORT=6333` (double underscore for nesting).

## Distributed Deployment

Enable cluster mode with `cluster.enabled: true` and specify bootstrap peers:

```bash
# Node 1 (bootstrap)
./qdrant --uri http://node1:6335

# Node 2
./qdrant --uri http://node2:6335 --bootstrap http://node1:6335

# Node 3
./qdrant --uri http://node3:6335 --bootstrap http://node1:6335
```

Cluster features:
- **Raft consensus** for metadata coordination
- **Sharding**: Automatic or manual shard distribution
- **Replication**: Configurable replication factor per collection
- **Shard transfer methods**: `stream_records`, `snapshot`, `wal_delta`
- **Write consistency**: Configurable quorum (1 to all replicas)

## Debug Binaries

Available with `--features service_debug`:

```bash
# Generate JSON schema for all types
cargo run --features service_debug --bin schema_generator

# Inspect WAL contents
cargo run --features service_debug --bin wal_inspector -- <path>

# Pop entries from WAL
cargo run --features service_debug --bin wal_pop -- <path>

# Inspect segment data
cargo run --features service_debug --bin segment_inspector -- <path>
```

## Docker Build

The Dockerfile supports multi-platform builds with optional GPU:

```bash
# Standard build
docker build -t hanzo-vector .

# With GPU (NVIDIA)
docker build --build-arg GPU=nvidia -t hanzo-vector:gpu .

# With GPU (AMD)
docker build --build-arg GPU=amd -t hanzo-vector:gpu-amd .

# Custom profile
docker build --build-arg PROFILE=ci -t hanzo-vector:ci .

# With extra features
docker build --build-arg FEATURES=rocksdb -t hanzo-vector:rocksdb .
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `gpu` | GPU-accelerated HNSW (NVIDIA Vulkan / AMD ROCm) |
| `rocksdb` | RocksDB storage backend (alternative to default) |
| `tracing` | Distributed tracing support |
| `console` | tokio-console subscriber for runtime debugging |
| `tracy` | Tracy profiler integration |
| `stacktrace` | Stack trace capture on Linux |
| `staging` | Staging environment features |
| `data-consistency-check` | Runtime data integrity verification |
| `chaos-testing` | Chaos engineering hooks |

## Testing

```bash
# Unit tests
cargo test

# Integration tests (requires running instance)
bash tests/basic_api_test.sh
bash tests/basic_grpc_test.sh
bash tests/basic_sparse_test.sh

# Consensus tests
cd tests/consensus_tests && cargo test

# E2E tests
cd tests/e2e_tests && cargo test

# OpenAPI consistency check
bash tests/openapi_consistency_check.sh

# Python integration tests (uses uv)
cd tests && uv run pytest
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 6333 in use | Another instance running | `lsof -i :6333` and kill |
| OOM on large collections | Vectors in RAM | Set `on_disk: true` in vector config or use quantization |
| Slow search | No HNSW index built yet | Wait for optimizer or lower `indexing_threshold_kb` |
| gRPC connection refused | gRPC disabled | Set `grpc_port: 6334` in config |
| Cluster peer unreachable | P2P port blocked | Open port 6335, check TLS settings |
| GPU not detected | Missing Vulkan drivers | Install `vulkan-tools`, verify with `vulkaninfo` |
| Build fails on macOS | Missing protoc | `brew install protobuf` |

## Related Skills

- `hanzo/hanzo-database.md` - PostgreSQL (pgvector for simpler vector search)
- `hanzo/hanzo-storage.md` - S3-compatible object storage
- `hanzo/hanzo-llm-gateway.md` - LLM Gateway (generates embeddings for vector storage)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vector, embeddings, similarity-search, qdrant, hnsw, ai
**Prerequisites**: Rust, Docker, protobuf basics
