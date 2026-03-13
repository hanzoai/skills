# Hanzo Storage - S3-Compatible Object Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kv-go.md`, `hanzo/go-sdk.md`

## Overview

Hanzo Storage is a high-performance, S3-compatible object storage server for AI workloads. Fork of MinIO. This is a full server binary, not a client library. Any S3-compatible SDK (aws-sdk-go, boto3, minio-go) works against it.

**Note**: The Go module path is still `github.com/minio/minio` (not yet rebranded). Use standard MinIO/S3 client libraries to interact with it.

### OSS Base

Fork of **MinIO** (minio/minio). Repo: `hanzoai/s3`, branch: `main`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/minio/minio` (upstream, not yet rebranded) |
| Binary | `minio` |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/s3` |
| Branch | `main` |
| License | AGPL-3.0 |
| Protocol | S3 API (HTTP/REST) |
| Data port | 9000 |
| Console port | 9001 |
| Docker tag | `ghcr.io/hanzoai/s3:<version>` |

## Quick start

### Docker

```bash
docker build -t hanzo-storage .
docker run -p 9000:9000 -p 9001:9001 \
  hanzo-storage server /data --console-address :9001
```

Default credentials: `minioadmin:minioadmin` -- change immediately in production.

### Build from source

```bash
make build
./minio server /data --console-address :9001
```

### Verify connectivity

```bash
mc alias set hanzo http://localhost:9000 minioadmin minioadmin
mc admin info hanzo
mc mb hanzo/my-bucket
mc cp ~/data/model.safetensors hanzo/my-bucket/
mc ls hanzo/my-bucket/
```

## Key features

- Full S3 API compatibility (any S3 SDK works)
- Erasure coding with configurable redundancy
- Server-side encryption (SSE-S3, SSE-KMS)
- Bucket policies and IAM
- Object lifecycle management (expiration, transition, tiering)
- Multi-tenancy with isolated namespaces
- Event notifications (NATS, Kafka, AMQP, MQTT, NSQ, Elasticsearch, PostgreSQL, MySQL, Redis, etcd, webhooks)
- Built-in web console (port 9001)
- Prometheus metrics
- Site replication (multi-site HA)
- Decommissioning and rebalancing

## Client SDKs

Any S3-compatible SDK works. Purpose-built options:

| Language | Package |
|----------|---------|
| Go | `github.com/minio/minio-go/v7` |
| JavaScript | `minio` (npm) |
| Python | `minio` (pip) |
| AWS SDKs | `aws-sdk-go`, `boto3`, `@aws-sdk/client-s3` |

## Build and test

```bash
make build          # build binary
make test           # lint + unit tests
make verify         # integration verification
make install        # install to $GOPATH/bin
make docker         # build Docker image
```

## Makefile targets

| Target | Description |
|--------|-------------|
| `build` | Build `./minio` binary |
| `test` | Lint + unit tests |
| `install` | Build and install to `$GOPATH/bin` |
| `docker` | Build Docker image |
| `clean` | Remove build artifacts |
| `verify` | Full integration verification |
| `test-replication` | Multi-site replication tests |
| `test-iam` | IAM integration tests |

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: s3, storage, object-storage, minio, server
**Prerequisites**: Go 1.26+
