# Hanzo Registry - Private Docker Container Registry

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Registry is a **private Docker container registry** running Docker Distribution (registry:2) on hanzo-k8s, authenticated via Hanzo IAM token-based auth. It provides a self-hosted alternative to Docker Hub and GHCR for internal container images. Live at `registry.hanzo.ai`.

### Why Hanzo Registry?

- **IAM-integrated auth**: Token-based auth via `iam.hanzo.ai/api/registry/token`
- **Persistent storage**: 50Gi PVC on DigitalOcean Block Storage
- **CORS-enabled**: Supports browser-based registry operations
- **Delete support**: Image deletion enabled for cleanup workflows
- **K8s-native**: Runs as a Deployment in the `hanzo` namespace on hanzo-k8s

### Tech Stack

- **Image**: `registry:2` (Docker Distribution)
- **Auth**: JWT token-based via Hanzo IAM (Casdoor)
- **Storage**: Filesystem-backed with 50Gi PVC (`do-block-storage`)
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
2. **Hanzo IAM** at iam.hanzo.ai for token-based authentication
3. **Signing certificate** (`signing.crt` / `signing.key`) as K8s secret `registry-signing-key`
4. **PVC**: 50Gi `do-block-storage` class PersistentVolumeClaim

## Quick reference

| Item | Value |
|------|-------|
| Endpoint | `registry.hanzo.ai` |
| Internal port | 5000 (ClusterIP) |
| Auth realm | `https://iam.hanzo.ai/api/registry/token` |
| Token issuer | `hanzo-iam` |
| Storage | 50Gi PVC (`registry-data`) |
| K8s namespace | `hanzo` |
| Repo | `github.com/hanzoai/registry` |
| License | Apache 2.0 |

## One-file quickstart

### Push and pull images

```bash
# Login (uses Hanzo IAM credentials)
docker login registry.hanzo.ai

# Push an image
docker tag myapp:latest registry.hanzo.ai/myapp:latest
docker push registry.hanzo.ai/myapp:latest

# Pull an image
docker pull registry.hanzo.ai/myapp:latest
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
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Docker Client   │────>│  registry.hanzo.ai│────>│  Hanzo IAM      │
│  (push/pull)     │     │  (registry:2)     │     │  (token realm)  │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                          ┌──────┴─────────┐
                          │  50Gi PVC       │
                          │  (DO Block)     │
                          └────────────────┘
```

### Auth Flow

1. Docker client attempts to push/pull from `registry.hanzo.ai`
2. Registry returns 401 with token realm URL (`iam.hanzo.ai/api/registry/token`)
3. Client requests token from IAM, providing credentials
4. IAM validates credentials and returns a signed JWT (issuer: `hanzo-iam`)
5. Client retries with JWT in Authorization header
6. Registry validates JWT signature against `signing.crt` mounted from K8s secret

### Registry Configuration

```yaml
# config.yml
version: 0.1
storage:
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true
http:
  addr: :5000
  headers:
    X-Content-Type-Options: [nosniff]
    Access-Control-Allow-Origin: ['https://registry.hanzo.ai']
    Access-Control-Allow-Methods: ['HEAD', 'GET', 'OPTIONS', 'DELETE']
auth:
  token:
    realm: https://iam.hanzo.ai/api/registry/token
    service: registry.hanzo.ai
    issuer: hanzo-iam
    rootcertbundle: /etc/registry-signing/signing.crt
```

### K8s Resources

- **Deployment**: 1 replica, `registry:2` image, 100m/128Mi request, 500m/512Mi limit
- **Service**: ClusterIP on port 5000
- **PVC**: 50Gi `do-block-storage` class (`registry-data`)
- **Secret**: `registry-signing-key` with `signing.crt` mounted to `/etc/registry-signing/`

### CI/CD

The `deploy.yml` workflow triggers on push to `main` (when `k8s/` or `config.yml` change) or manual dispatch:
1. Fetches DO_API_TOKEN from KMS via Universal Auth
2. Configures kubectl via `doctl kubernetes cluster kubeconfig save hanzo-k8s`
3. Applies K8s manifests (`kubectl apply -f k8s/`)
4. Restarts and waits for rollout

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 on push/pull | Missing or expired IAM token | `docker login registry.hanzo.ai` |
| Certificate error | `signing.crt` not mounted | Verify `registry-signing-key` secret exists |
| Storage full | 50Gi PVC exhausted | Delete unused images or resize PVC |
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
