# Hanzo S3 - S3-Compatible Object Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo S3 is a **high-performance, S3-compatible object storage server** for AI workloads. Written in Go, it is the object-storage substrate for the Hanzo ecosystem and runs in production on Hanzo K8s clusters. This is a full server binary, not a client library — any S3 SDK works against it.

### Why Hanzo S3?

- **S3 API compatible**: Drop-in replacement for Amazon S3; works with any S3 client or SDK
- **Built for AI**: Optimized for large model artifacts, training datasets, and data pipelines
- **Erasure coding**: Reed-Solomon shards with rebuild and rebalance
- **Server-side encryption**: SSE-C, SSE-KMS, and SSE-S3
- **Object versioning**: Version IDs with delete markers
- **Lifecycle management**: Expiration and transition rules (ilm)

### Tech Stack

- **Language**: Go (module: `github.com/hanzoai/s3`)
- **Binary**: `s3`
- **Image**: `ghcr.io/hanzoai/s3` (published **linux/amd64 only**)
- **License**: Apache-2.0

### OSS Base

Derived from [SeaweedFS](https://github.com/seaweedfs/seaweedfs) (Apache-2.0). Repo: `hanzoai/s3`, branch `main`. Attribution to upstream is retained in the repo's `NOTICE`.

## When to use

- Storing model artifacts, checkpoints, and training datasets
- Serving static assets and large files via the S3 API
- Providing S3-compatible object storage for Hanzo services
- Replacing Amazon S3 in self-hosted or hybrid deployments
- Bucket lifecycle management (expiration, tiering)

## Hard requirements

1. **Go 1.26+** to build from source
2. **Docker** for container deployment
3. **linux/amd64** — the published image has no arm64 variant
4. **NVMe/SSD storage** recommended for performance

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/s3` |
| Module | `github.com/hanzoai/s3` |
| Binary | `s3` |
| License | Apache-2.0 |
| Image | `ghcr.io/hanzoai/s3:v1.0.14` |
| Entrypoint | `s3` |
| Default command | `server -s3 -s3.port=9000 -dir=/data` |
| S3 API port | 9000 |
| Filer port | 8888 |
| Master port | 9333 |
| Volume port | 8080 |
| Data directory | `/data` |
| In-cluster | `s3.hanzo.svc:9000` |
| Public | `s3.hanzo.ai` |

There is **no web console**. Management is via the S3 API and the `s3` CLI.

## One-file quickstart

### Docker

The image entrypoint is `s3` and its default command already starts the all-in-one
server, so no `command:` is needed.

```bash
docker run -p 9000:9000 \
  -e AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
  -e AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
  -v s3-data:/data \
  ghcr.io/hanzoai/s3:v1.0.14
```

The server reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and creates a static
admin identity from them. Supply them from KMS or your shell environment — never
hardcode credentials.

### Docker Compose

```yaml
# compose.yml
services:
  s3:
    image: ghcr.io/hanzoai/s3:v1.0.14
    ports:
      - "9000:9000"
    environment:
      AWS_ACCESS_KEY_ID: "${S3_ACCESS_KEY}"
      AWS_SECRET_ACCESS_KEY: "${S3_SECRET_KEY}"
    volumes:
      - s3-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  s3-data:
```

### Build from source

```bash
git clone https://github.com/hanzoai/s3.git
cd s3
CGO_ENABLED=0 go build -trimpath -o s3 ./s3
./s3 server -s3 -s3.port=9000 -dir=/data
```

### Verify connectivity

Standard AWS tooling works against any endpoint:

```bash
aws --endpoint-url http://localhost:9000 s3 mb s3://my-bucket
aws --endpoint-url http://localhost:9000 s3 cp ~/data/model.safetensors s3://my-bucket/
aws --endpoint-url http://localhost:9000 s3 ls s3://my-bucket/
```

The `ghcr.io/hanzoai/s3-cli` image ships the `s3` client for alias-style workflows:

```bash
s3 alias set local http://localhost:9000 "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}"
s3 ls local/my-bucket/
```

## Core concepts

### Architecture

The all-in-one `server` command runs master, volume, filer, and the S3 gateway in a
single process. Each exposes its own port:

```
┌──────────────┐      ┌──────────────────────────────┐
│  S3 clients  │─────>│  S3 gateway      (port 9000) │
│  (any SDK)   │      ├──────────────────────────────┤
└──────────────┘      │  Filer           (port 8888) │
                      │  Master          (port 9333) │
                      │  Volume          (port 8080) │
                      └──────────────────────────────┘
```

### SDK compatibility

Use the standard AWS SDK for your language, pointed at the Hanzo S3 endpoint:

| Language | Package |
|----------|---------|
| Python | `boto3` |
| Go | `github.com/aws/aws-sdk-go-v2` |
| JavaScript / TypeScript | `@aws-sdk/client-s3` |

No Hanzo-specific client library is required — configure an endpoint override and
the standard credential environment variables.

### Key packages

```
s3/
  s3api/          # S3 REST API handlers (SSE, versioning, lifecycle)
  command/        # CLI commands (server, master, filer, volume, admin)
  server/         # HTTP servers and admin handlers
  filer/          # Filer metadata store
  iam/, iamapi/   # Identity and access management
  kms/            # KMS integration for SSE-KMS
  notification/   # Event notification
  replication/    # Cross-store replication
  shell/          # Admin shell (ec.encode, ec.rebuild, ec.balance)
  stats/          # Prometheus metrics
```

### Credentials

| Variable | Purpose |
|----------|---------|
| `AWS_ACCESS_KEY_ID` | Admin identity access key |
| `AWS_SECRET_ACCESS_KEY` | Admin identity secret key |

These are the standard AWS names every S3 client already understands. Store them in
KMS (`hanzo/hanzo-kms.md`) and inject them as environment variables — never commit
credentials, and never ship a default key pair.

### Health checks

The S3 API port serves all three probes:

| Endpoint | Purpose |
|----------|---------|
| `GET :9000/healthz` | Liveness |
| `GET :9000/readyz` | Readiness |
| `GET :9000/status` | Status detail |

### Metrics

Prometheus metrics are exposed when `-metricsPort=<port>` is set (disabled when `0`).
All series use the `Hanzo` namespace:

| Metric | Meaning |
|--------|---------|
| `Hanzo_s3_request_total{type,code,bucket}` | Request count by operation, status, bucket |
| `Hanzo_s3_request_seconds` | Request latency histogram |
| `Hanzo_s3_time_to_first_byte_millisecond` | Time to first byte |
| `Hanzo_s3_bucket_traffic_received_bytes_total` | Ingress bytes per bucket |
| `Hanzo_s3_bucket_traffic_sent_bytes_total` | Egress bytes per bucket |
| `Hanzo_volumeServer_total_disk_size` | Disk used by volumes |
| `Hanzo_volumeServer_volumes` | Volume/shard count |
| `Hanzo_volumeServer_disk_error_status` | Disk error status |

### Erasure coding

Erasure coding is driven from the admin shell:

```bash
s3 shell
> ec.encode -collection my-collection    # encode volumes into EC shards
> ec.rebuild                             # rebuild missing shards
> ec.balance                             # rebalance shards across nodes
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `no matching manifest` on pull | Image is linux/amd64 only | Run on an amd64 host, or `--platform linux/amd64` under emulation |
| 403 on every request | Credentials not picked up | Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; the server logs `Added admin identity from AWS environment variables` on success |
| Health check never passes | Probing a vendor-specific health path | Use `/healthz` on port 9000 |
| Console URL 404s | There is no web console | Manage buckets via the S3 API or the `s3` CLI |
| Metrics endpoint empty | Metrics disabled by default | Start with `-metricsPort=9327` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-kms.md` - Secret management (SSE-KMS integration)
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-pubsub.md` - Event notification targets (NATS)

---

**Last Updated**: 2026-08-03
**Category**: Hanzo Ecosystem
**Related**: s3, storage, object-storage, seaweedfs, server
**Prerequisites**: Go 1.26+, Docker
