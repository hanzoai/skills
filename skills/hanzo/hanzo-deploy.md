# Hanzo Deploy - Deployment Guide

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-k8s.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`

## Overview

Hanzo deploys all services via Kubernetes manifests stored in the **universe** repo (`hanzoai/universe`). Universe is the single source of truth for all infrastructure. Branch `main` is production, branch `dev` is staging. All images are built by CI/CD on self-hosted runners and pushed to GHCR. Never build images locally unless explicitly requested.

## When to use

- Deploying any Hanzo service to production or staging
- Setting up CI/CD pipelines for Hanzo services
- Understanding the deployment workflow
- Adding a new service to the Hanzo infrastructure

## Hard requirements

1. **Universe is the source of truth**: All K8s manifests live in `~/work/hanzo/universe/infra/k8s/`
2. **main = production, dev = staging**: Never push untested changes directly to main
3. **Self-hosted runners only**: Never use GitHub Actions billing. All builds run on the 7-node runner pool in hanzo-k8s
4. **GHCR for images**: `ghcr.io/hanzoai/<service>:<tag>`, always `--platform linux/amd64`
5. **KMS for secrets**: All secrets via KMSSecret CRDs synced from kms.hanzo.ai. Never hardcode secrets in manifests
6. **No local builds**: CI/CD builds for all architectures. Do not build images on dev machines
7. **Kustomize, not Helm**: `kubectl kustomize . | kubectl apply -f -` for idempotent deploy
8. **No nginx, no caddy**: Hanzo Ingress (Traefik) or the static plugin handles serving

## Quick reference

| Item | Value |
|------|-------|
| Universe repo | `github.com/hanzoai/universe` (private) |
| Local path | `~/work/hanzo/universe/infra/k8s/` |
| Prod cluster | `do-sfo3-hanzo-k8s` (22 nodes) |
| Staging cluster | `do-sfo3-hanzo-dev-k8s` |
| Registry | `ghcr.io/hanzoai/*` |
| DO registry | `registry.digitalocean.com/hanzo` (5/5 repo limit, avoid) |
| Runner pool | 7x32GB nodes in hanzo-k8s |
| KMS | `kms.hanzo.ai` |

## Deployment workflow

### Standard service deploy

```bash
# 1. Edit manifests in universe
cd ~/work/hanzo/universe/infra/k8s/<service>

# 2. Apply to production
kubectl --context do-sfo3-hanzo-k8s kustomize . | kubectl apply -f -

# 3. Verify
kubectl --context do-sfo3-hanzo-k8s -n hanzo get pods -l app=<service>
kubectl --context do-sfo3-hanzo-k8s -n hanzo logs -l app=<service> --tail=50
```

### New service checklist

1. Create directory in `universe/infra/k8s/<service>/`
2. Write `kustomization.yaml`, `deployment.yaml`, `service.yaml`
3. Add `KMSSecret` CRD in `secrets.yaml` for any secrets
4. Add Ingress resource with `ingressClassName: hanzo`
5. Add Cloudflare DNS record pointing to cluster LB (`165.232.146.176` for hanzo-k8s)
6. Set up CI/CD workflow in service repo to build+push image to GHCR
7. Apply manifests: `kubectl kustomize . | kubectl apply -f -`

### Image build (CI/CD)

All images are built in CI/CD pipelines, never locally.

```yaml
# Typical .github/workflows/build.yml
name: Build and Push
on:
  push:
    branches: [main, develop]
jobs:
  build:
    runs-on: self-hosted  # MUST use self-hosted runners
    steps:
      - uses: actions/checkout@v4
      - name: Login to GHCR
        run: echo "${{ secrets.GHCR_TOKEN }}" | docker login ghcr.io -u hanzoai --password-stdin
      - name: Build and push
        run: |
          docker buildx build \
            --platform linux/amd64 \
            --push \
            -t ghcr.io/hanzoai/<service>:latest \
            -t ghcr.io/hanzoai/<service>:${{ github.sha }} \
            .
```

### Secret management

```yaml
# secrets.yaml -- KMSSecret CRD
apiVersion: kms.hanzo.ai/v1
kind: KMSSecret
metadata:
  name: my-service-secrets
  namespace: hanzo
spec:
  project: my-service
  environment: production
  syncInterval: 5m
  secretRef:
    name: my-service-secrets
  secrets:
    - DATABASE_URL
    - API_KEY
    - JWT_SECRET
```

## Universe directory structure

```
universe/infra/k8s/
  namespace.yaml         # hanzo namespace
  cluster-issuer.yaml    # Let's Encrypt issuer
  platform-ingress.yaml  # Platform-level ingress
  rbac/                  # RBAC for service accounts
  base/                  # Shared base manifests
  # --- Services ---
  app/                   # hanzo.app
  billing/               # billing.hanzo.ai
  bot/                   # bot gateway
  chat/                  # chat.hanzo.ai
  cloud/                 # cloud.hanzo.ai (Casibase)
  commerce/              # commerce API
  console/               # console.hanzo.ai (Langfuse)
  dns/                   # CoreDNS
  flow/                  # workflow builder
  gateway/               # API gateway (KrakenD)
  iam/                   # hanzo.id (Casdoor)
  kms/                   # kms.hanzo.ai (Infisical)
  monitoring/            # Prometheus/Grafana
  o11y/                  # SigNoz observability
  paas/                  # platform.hanzo.ai
  registry/              # container registry
  search/                # search service
  sql/                   # PostgreSQL + ZAP sidecar
  storage/               # MinIO
  team/                  # hanzo.team
  vector/                # vector DB
  zen/                   # Zen model serving
  zt/                    # zero-trust (OpenZiti)
  # ... 40+ service directories
```

## Cluster topology

### hanzo-k8s (production)

- **Location**: DigitalOcean SFO3
- **Nodes**: 22 (15 workers + 7 runners)
- **Context**: `do-sfo3-hanzo-k8s`
- **LB IP**: `165.232.146.176`
- **Services**: All Hanzo services (IAM, KMS, Platform, Cloud, Console, Gateway, Commerce, etc.)

### lux-k8s (Lux blockchain)

- **Location**: DigitalOcean SFO3
- **Nodes**: Separate cluster for Lux blockchain
- **Context**: `do-sfo3-lux-k8s`
- **LB IP**: `24.144.69.101`
- **Services**: 15 validators, gateway, markets, lux-cloud-web

## Gotchas

1. **GHA billing is blocked** for hanzoai org. All CI/CD must use self-hosted runners.
2. **Runner pool waste**: 7x32GB nodes at 17-26% utilization ($1,764/mo waste). Plan to scale to 2.
3. **DO registry limit**: 5/5 repo limit on `registry.digitalocean.com/hanzo`. Use GHCR exclusively.
4. **Local TLS errors**: Pushing to GHCR from local can fail with TLS errors. Use the buildx K8s driver.
5. **DO App Platform is DECOMMISSIONED**: All services are K8s-native as of Feb 2026. No managed databases.
6. **In-cluster PostgreSQL**: `postgres.hanzo.svc` -- no managed DB services.

## Related Skills

- `hanzo/hanzo-k8s.md` -- K8s cluster details
- `hanzo/hanzo-kms.md` -- Secret management
- `hanzo/hanzo-ingress.md` -- Ingress routing
- `hanzo/hanzo-platform.md` -- PaaS for app deployment

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: deployment, ci-cd, kubernetes, universe, ghcr
**Prerequisites**: kubectl, K8s cluster access, GHCR credentials
