# Hanzo Storage - S3-Compatible Object Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Storage is a **high-performance, S3-compatible object storage server** for AI workloads. It is a fork of MinIO, rebranded and optimized for the Hanzo ecosystem. Written in Go. Live in production on Hanzo K8s clusters.

### Why Hanzo Storage?

- **S3 API Compatible**: Drop-in replacement for Amazon S3; works with any S3 client or SDK
- **Built for AI**: Optimized for large model artifacts, training datasets, and data pipelines
- **Erasure Coding**: Configurable data redundancy across drives and nodes
- **Server-Side Encryption**: SSE-S3 and SSE-KMS for data at rest and in transit
- **Multi-Tenancy**: Isolated namespaces and access boundaries for teams and services
- **Web Console**: Hanzo-branded console UI (`hanzoai/storage-console` fork)

### Tech Stack

- **Language**: Go (module: `github.com/minio/minio`)
- **Go Version**: 1.26
- **Console**: `github.com/hanzoai/storage-console` v1.7.7-hanzo.1
- **License**: AGPL v3

### OSS Base

Fork of [MinIO](https://github.com/minio/minio). Repo: `hanzoai/s3` (GitHub name is `s3`, branded as "Hanzo Storage").

## When to use

- Storing model artifacts, checkpoints, and training datasets
- Serving static assets and large files via S3 API
- Providing S3-compatible object storage for Hanzo services
- Replacing Amazon S3 in self-hosted or hybrid deployments
- Bucket lifecycle management (expiration, tiering)

## Hard requirements

1. **Go 1.24+** to build from source
2. **Docker** for container deployment
3. **NVMe/SSD storage** recommended for performance

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/s3` |
| Module | `github.com/minio/minio` (upstream path retained) |
| Go Version | 1.26 |
| License | AGPL v3 |
| API Port | 9000 |
| Console Port | 9001 |
| Docker Image | `ghcr.io/hanzoai/s3:latest` |
| Console Fork | `github.com/hanzoai/storage-console` |
| Dockerfile | `Dockerfile.hanzo` (production) |

## One-file quickstart

### Docker (Hanzo build)

```bash
docker build -f Dockerfile.hanzo -t hanzo-storage .
docker run -p 9000:9000 -p 9001:9001 \
  hanzo-storage server /data --console-address :9001
```

### Build from source

```bash
git clone https://github.com/hanzoai/s3.git
cd s3
go build -o hanzo-storage .
./hanzo-storage server /data --console-address :9001
```

### Verify with S3 client (mc)

```bash
mc alias set hanzo http://localhost:9000 minioadmin minioadmin
mc admin info hanzo
mc mb hanzo/my-bucket
mc cp ~/data/model.safetensors hanzo/my-bucket/
mc ls hanzo/my-bucket/
```

### Docker Compose

```yaml
# compose.yml
services:
  storage:
    image: ghcr.io/hanzoai/s3:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: "${S3_ROOT_USER}"
      MINIO_ROOT_PASSWORD: "${S3_ROOT_PASSWORD}"
    volumes:
      - storage-data:/data

volumes:
  storage-data:
```

## Core Concepts

### Architecture

```
┌──────────────┐     ┌──────────────────┐
│  S3 Clients  │────>│  Hanzo Storage   │
│  (any SDK)   │     │  (port 9000)     │
└──────────────┘     └────────┬─────────┘
                              │
┌──────────────┐     ┌────────┴─────────┐
│  Console UI  │────>│  Console Server  │
│  (browser)   │     │  (port 9001)     │
└──────────────┘     └──────────────────┘
```

### SDK Compatibility

Any S3-compatible SDK works. Purpose-built SDKs:

| Language | Package |
|----------|---------|
| Go | `github.com/minio/minio-go/v7` |
| JavaScript / TypeScript | `minio` (npm) |
| Python | `minio` (pip) |

Standard AWS SDKs (`aws-sdk-go`, `boto3`, `@aws-sdk/client-s3`) also work without modification.

### Key Internal Packages

```
s3/
  main.go                  # Entry point (calls cmd.Main)
  cmd/                     # CLI commands and server logic
  internal/
    auth/                  # Authentication
    bucket/                # Bucket management
    config/                # Configuration
    crypto/                # Encryption (SSE-S3, SSE-KMS)
    event/                 # Event notification
    grid/                  # Internal RPC grid
    hash/                  # Content hashing
    http/                  # HTTP server
    jwt/                   # JWT handling
    kms/                   # KMS integration
    s3select/              # S3 Select queries
    store/                 # Object store backend
  helm/                    # Helm charts
  workers/                 # Background workers
  buildscripts/            # Build automation
  dockerscripts/           # Docker entry scripts
```

### Environment Variables

```bash
# Hanzo-specific env var names (from Dockerfile.hanzo)
S3_ACCESS_KEY_FILE=access_key
S3_SECRET_KEY_FILE=secret_key
S3_ROOT_USER_FILE=access_key
S3_ROOT_PASSWORD_FILE=secret_key
S3_KMS_SECRET_KEY_FILE=kms_master_key
S3_CONFIG_ENV_FILE=config.env
MC_CONFIG_DIR=/tmp/.mc

# Standard MinIO env vars also supported
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

### Key Features

- **Erasure Coding**: Data protection with configurable redundancy
- **Bucket Policies**: S3-compatible policy documents for access control
- **Object Lifecycle**: Automated expiration, transition, and tiering rules
- **Event Notifications**: Kafka, NATS, MQTT, Elasticsearch, Redis, PostgreSQL, MySQL targets
- **S3 Select**: Query objects with SQL expressions
- **Versioning**: Object versioning with delete markers
- **Replication**: Cross-site bucket replication

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Default credentials | Using `minioadmin:minioadmin` | Set `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` |
| Console not loading | Console address not set | Add `--console-address :9001` |
| Build fails | Go version mismatch | Requires Go 1.24+ |
| Module path confusion | Module is `github.com/minio/minio` | Upstream module path retained for SDK compat |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-kms.md` - Secret management (SSE-KMS integration)
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-pubsub.md` - Event notification targets (NATS)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: s3, storage, object-storage, minio, ai-data
**Prerequisites**: Go 1.24+, Docker
