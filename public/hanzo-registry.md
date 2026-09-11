# Hanzo Registry - Private Docker Container Registry

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Registry is a **private Docker container registry** running Docker Distribution (registry:2) on hanzo-k8s, authenticated via Hanzo IAM token-based auth. It carries container images and Helm charts, org-namespaced as `<host>/<org>/<app>`. Live at `oci.hanzo.ai`.

### Why Hanzo Registry?

- **IAM-integrated auth**: Token-based auth via `hanzo.id/v1/iam/registry/token`
- **S3-backed storage**: object storage on hanzoai/s3, no PVC
- **CORS-enabled**: Supports browser-based registry operations
- **Delete support**: Image deletion enabled for cleanup workflows
- **K8s-native**: Runs as a Deployment in the `hanzo` namespace on hanzo-k8s

### Tech Stack

- **Image**: `registry:2` (Docker Distribution)
- **Auth**: JWT token-based via Hanzo IAM (Hanzo IAM)
- **Storage**: S3-backed (hanzoai/s3)
- **Port**: 5000 (ClusterIP service)
- **CI**: GitHub Actions deploy workflow with KMS-sourced credentials

### OSS Base

Repo: `hanzoai/registry` (Apache 2.0).

## When to use

- Pushing/pulling private Docker images for Hanzo services
- Hosting container images that should not be on public registries
- Integrating container workflows with Hanzo IAM authentication
- Self-hosting a Docker registry with OIDC-based access control

## Hard requirements

1. **Kubernetes cluster** (hanzo-k8s) with the `hanzo` namespace
2. **Hanzo IAM** at hanzo.id for token-based authentication
3. **Signing certificate** (`signing.crt` / `signing.key`) as K8s secret `registry-signing-key`
4. **S3 bucket** on hanzoai/s3 with credentials in the `s3-credentials` secret

## Quick reference

| Item | Value |
|------|-------|
| Endpoint | `oci.hanzo.ai` |
| Repository path | `oci.hanzo.ai/<org>/<app>` |
| Internal port | 5000 (ClusterIP) |
| Auth realm | `https://hanzo.id/v1/iam/registry/token` |
| Auth service | `oci.hanzo.ai` |
| Token issuer | `hanzo-iam` |
| Storage | S3 (hanzoai/s3) |
| K8s namespace | `hanzo` |
| Repo | `github.com/hanzoai/registry` |
| License | Apache 2.0 |

## One-file quickstart

### Push and pull images

```bash
# Login (uses Hanzo IAM credentials)
docker login oci.hanzo.ai

# Push an image — repositories are org-namespaced
docker tag myapp:latest oci.hanzo.ai/hanzoai/myapp:latest
docker push oci.hanzo.ai/hanzoai/myapp:latest

# Pull an image
docker pull oci.hanzo.ai/hanzoai/myapp:latest
```

### First-time setup

```bash
# Generate a self-signed signing certificate (10-year validity)
make generate-cert

# Create the K8s secret from local cert files
make create-secret

# Deploy to hanzo-k8s
make deploy
```

### Operations

```bash
# Check deployment status
make status

# Tail logs
make logs

# Restart pods
make restart
```

## Core Concepts

### Architecture

```
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Docker Client │────>│ oci.hanzo.ai     │────>│ Hanzo IAM │
│ (push/pull) │ │ (registry:2) │ │ (token realm) │
└─────────────────┘ └────────┬─────────┘ └─────────────────┘
 │
 ┌──────┴─────────┐
 │ hanzoai/s3 │
 │ (object store) │
 └────────────────┘
```

### Auth Flow

1. Docker client attempts to push/pull from `oci.hanzo.ai`
2. Registry returns 401 with token realm URL (`hanzo.id/v1/iam/registry/token`)
3. Client requests token from IAM, providing credentials
4. IAM validates credentials and returns a signed JWT (issuer: `hanzo-iam`)
5. Client retries with JWT in Authorization header
6. Registry validates JWT signature against `signing.crt` mounted from K8s secret

### Registry Configuration

The auth and http stanzas of `config.yml` — the storage driver is S3, configured
from the `s3-credentials` secret:

```yaml
version: 0.1
http:
  addr: :5000
  headers:
    X-Content-Type-Options: [nosniff]
    Access-Control-Allow-Origin: ['https://oci.hanzo.ai']
    Access-Control-Allow-Methods: ['HEAD', 'GET', 'OPTIONS', 'DELETE']
auth:
  token:
    realm: https://hanzo.id/v1/iam/registry/token
    service: oci.hanzo.ai
    issuer: hanzo-iam
    rootcertbundle: /etc/registry-signing/signing.crt
```

### K8s Resources

- **Deployment**: 1 replica, `registry:2` image, 100m/128Mi request, 500m/512Mi limit
- **Service**: ClusterIP on port 5000
- **Secret**: `registry-signing-key` with `signing.crt` mounted to `/etc/registry-signing/`
- **Secret**: `s3-credentials` (KMS-synced) for the object store

### CI/CD

The `deploy.yml` workflow triggers on push to `main` (when `k8s/` or `config.yml` change) or manual dispatch:
1. Fetches DO_API_TOKEN from KMS via Universal Auth
2. Configures kubectl via `doctl kubernetes cluster kubeconfig save hanzo-k8s`
3. Applies K8s manifests (`kubectl apply -f k8s/`)
4. Restarts and waits for rollout

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 on push/pull | Missing or expired IAM token | `docker login oci.hanzo.ai` |
| Certificate error | `signing.crt` not mounted | Verify `registry-signing-key` secret exists |
| Push denied to a bare name | Repositories are org-namespaced | Tag as `oci.hanzo.ai/<org>/<app>` |
| CORS errors | Browser request blocked | Check `Access-Control-Allow-Origin` in config.yml |

## Related Skills

- `hanzo/hanzo-id.md` - IAM and authentication (token realm)
- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-universe.md` - Production K8s infrastructure
- `hanzo/hanzo-kms.md` - Secret management (deploy credentials)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: registry, docker, containers, iam
**Prerequisites**: Docker CLI, Kubernetes, Hanzo IAM credentials
