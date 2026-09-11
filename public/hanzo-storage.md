# Hanzo Storage - S3-Compatible Object Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Storage is a **high-performance, S3-compatible object storage server** for AI workloads. Derived from SeaweedFS and optimized for the Hanzo ecosystem. Written in Go. Live in production on Hanzo K8s clusters.

### Why Hanzo Storage?

- **S3 API Compatible**: Drop-in replacement for Amazon S3; works with any S3 client or SDK
- **Built for AI**: Optimized for large model artifacts, training datasets, and data pipelines
- **Erasure Coding**: Configurable data redundancy across drives and nodes
- **Server-Side Encryption**: SSE-S3 and SSE-KMS for data at rest and in transit
- **Multi-Tenancy**: Isolated namespaces and access boundaries for teams and services
- **CLI**: the `s3` binary's own shell — 153 commands covering fs, cluster, ec, volume, and mq

### Tech Stack

- **Language**: Go (module: `github.com/hanzoai/s3`)
- **Go Version**: 1.26.5
- **License**: Apache-2.0

### OSS Base

Derived from [SeaweedFS](https://github.com/seaweedfs/seaweedfs) (Apache-2.0), attribution retained in `LICENSE`. Repo: `hanzoai/s3`. `go.mod` names zero MinIO modules; the checksum libraries are our Apache-2.0 forks under `hanzos3/` (`crc64nvme`, `md5-simd`, `go`).

There is no separate CLI or console repo. `mc` and the MinIO console are not part of this stack — use the `s3` shell for administration.

## When to use

- Storing model artifacts, checkpoints, and training datasets
- Serving static assets and large files via S3 API
- Providing S3-compatible object storage for Hanzo services
- Replacing Amazon S3 in self-hosted or hybrid deployments
- Bucket lifecycle management (expiration, tiering)

## Hard requirements

1. **Go 1.26+** to build from source
2. **Docker** for container deployment
3. **NVMe/SSD storage** recommended for performance

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/s3` |
| Module | `github.com/hanzoai/s3` |
| Go Version | 1.26.5 |
| License | Apache-2.0 (SeaweedFS-derived) |
| S3 API Port | 9000 |
| Filer / Master / Volume | 8888 / 9333 / 8080 |
| Docker Image | `ghcr.io/hanzoai/s3:v1.0.14` (pin a semver; never `:latest`) |
| CLI | the `s3` shell (in-binary) |

## One-file quickstart

### Docker

```bash
docker run -p 9000:9000 -v s3-data:/data ghcr.io/hanzoai/s3:v1.0.14 \
  server -dir=/data -filer -s3 -s3.port=9000
```

### Build from source

```bash
git clone https://github.com/hanzoai/s3.git
cd s3 && go build -o s3 .
./s3 server -dir=/data -filer -s3 -s3.port=9000
```

### Administer with the `s3` shell

The shell is the CLI. There is no `mc`.

```bash
s3 shell
> s3.bucket.create -name my-bucket
> s3.bucket.list
> s3.accesskey.create -user alice
> fs.ls /buckets/my-bucket
> cluster.status
```

`fs.*` walks the filer namespace, `s3.*` manages buckets and access keys,
`cluster.*` and `ec.*` cover topology and erasure coding.

### Docker Compose

```yaml
# compose.yml
services:
  s3:
    image: ghcr.io/hanzoai/s3:v1.0.14
    ports:
      - "9000:9000"
    command: server -dir=/data -ip.bind=0.0.0.0 -filer -s3 -s3.port=9000
    environment:
      AWS_ACCESS_KEY_ID: "${S3_ACCESS_KEY}"
      AWS_SECRET_ACCESS_KEY: "${S3_SECRET_KEY}"
    volumes:
      - s3-data:/data

volumes:
  s3-data:
```

## Core Concepts

### Architecture

```
S3 clients (any SDK) ──> s3 gateway (:9000)
                              │
                          filer (:8888) ──> master (:9333) ──> volume (:8080)
```

Master tracks topology, volume stores chunks, filer holds the namespace, and the
S3 gateway maps buckets onto filer paths.

### SDK Compatibility

Standard AWS SDKs (`aws-sdk-go`, `boto3`, `@aws-sdk/client-s3`) work without
modification — prefer them. For Go, `github.com/hanzos3/go` is our Apache-2.0
client.

### Key Internal Packages

```
s3/
 main.go # Entry point (calls cmd.Main)
 cmd/ # CLI commands and server logic
 internal/
 auth/ # Authentication
 bucket/ # Bucket management
 config/ # Configuration
 crypto/ # Encryption (SSE-S3, SSE-KMS)
 event/ # Event notification
 grid/ # Internal RPC grid
 hash/ # Content hashing
 http/ # HTTP server
 jwt/ # JWT handling
 kms/ # KMS integration
 s3select/ # S3 Select queries
 store/ # Object store backend
 helm/ # Helm charts
 workers/ # Background workers
 buildscripts/ # Build automation
 dockerscripts/ # Docker entry scripts
```

### Environment Variables

```bash
# S3 credentials (standard AWS names)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Admin identities
S3_ADMIN_USER=...
S3_ADMIN_PASSWORD=...
S3_ADMIN_READONLY_USER=...
S3_ADMIN_READONLY_PASSWORD=...

# Optional
S3_BUCKET=...
S3_EXTERNAL_URL=...
```

Credentials come from KMS, never from literals in a manifest.

### Key Features

- **Erasure Coding**: `ec.encode` / `ec.rebuild` / `ec.balance` / `ec.scrub`
- **Bucket Policies**: `s3.bucket.access`, `s3.bucket.owner`, `s3.anonymous.*`
- **Object Lifecycle**: `s3.bucket.lifecycle` (expiration and tiering rules)
- **Quotas and Locking**: `s3.bucket.quota`, `s3.bucket.lock`
- **Versioning**: Object versioning with delete markers
- **Remote Tiering**: `remote.mount` / `remote.cache` against a backing store

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Reaching for `mc` | Habit from the MinIO era | Use `s3 shell`; `mc` is not part of this stack |
| Build fails | Go version mismatch | Requires Go 1.26+ |
| S3 API refuses a bucket | Bucket has no filer path yet | `s3.bucket.create -name <bucket>` in the shell |
| Volume full | `volumeSizeLimitMB` reached | Raise `-master.volumeSizeLimitMB` or add volume servers |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-kms.md` - Secret management (SSE-KMS integration)
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-pubsub.md` - Event notification targets (NATS)

---

**Category**: Hanzo Ecosystem
**Related**: s3, storage, object-storage, seaweedfs, ai-data
**Prerequisites**: Go 1.26+, Docker
