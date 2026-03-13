# Hanzo Charts - Helm Charts for Hanzo Services

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-operator.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo Charts is the **official Helm chart repository** for all Hanzo services and infrastructure. Contains 16 charts covering core services (IAM, KMS, Gateway), AI services (Cloud, Agents, LLM), platform services (Console, GitOps, Platform), business services (Commerce, Analytics), and infrastructure (Storage, Bootnode, MPC, Datastore, Edge, Net).

### Why Hanzo Charts?

- **16 charts**: Complete coverage of every Hanzo service
- **Unified configuration**: Global values for domain, registry, storage class, TLS
- **Upstream tracking**: Charts for KrakenD, vLLM, Agnost, Dokploy with version pinning
- **Chart Testing**: CI with `ct` (chart-testing) for linting and validation
- **Helm repo**: Published to `https://charts.hanzo.ai`

### Tech Stack

- **Helm** 3.12+
- **Kubernetes** 1.25+
- **Chart Testing**: `ct.yaml` configuration
- **CI**: GitHub Actions for lint/test/publish
- **Language**: Go Template (Helm templates)

## When to use

- Deploying individual Hanzo services via Helm
- Installing a full Hanzo stack with a single umbrella chart
- Customizing service configuration via values.yaml overrides
- Templating K8s manifests for Hanzo infrastructure

## Hard requirements

1. **Helm** 3.12+
2. **Kubernetes** 1.25+ cluster
3. Access to `ghcr.io/hanzoai/*` container images

## Quick reference

| Item | Value |
|------|-------|
| Repo URL | `https://charts.hanzo.ai` |
| Chart count | 16 |
| Source | `github.com/hanzoai/charts` |
| Branch | `main` |
| License | Apache 2.0 |
| Chart testing | `ct.yaml` at repo root |

## Available Charts

### Core Services

| Chart | Path | Description |
|-------|------|-------------|
| `iam` | `charts/iam/` | Identity and Access Management (Casdoor-based) |
| `kms` | `charts/kms/` | Key Management Service (Infisical-based) |
| `gateway` | `charts/gateway/` | API Gateway (KrakenD-based) |

### AI Services

| Chart | Path | Description |
|-------|------|-------------|
| `cloud` | `charts/cloud/` | Multi-tenant AI/MCP platform |
| `agents` | `charts/agents/` | AI Agent orchestration |
| `llm` | `charts/llm/` | LLM serving (vLLM-based) |

### Platform Services

| Chart | Path | Description |
|-------|------|-------------|
| `console` | `charts/console/` | Admin Console UI |
| `gitops` | `charts/gitops/` | K8s CI/CD (Agnost-based) |
| `platform` | `charts/platform/` | Local dev platform (Dokploy-based) |

### Business Services

| Chart | Path | Description |
|-------|------|-------------|
| `commerce` | `charts/commerce/` | Multi-tenant Commerce |
| `analytics` | `charts/analytics/` | Analytics service |

### Infrastructure

| Chart | Path | Description |
|-------|------|-------------|
| `storage` | `charts/storage/` | S3-compatible object storage |
| `bootnode` | `charts/bootnode/` | Blockchain infrastructure |
| `mpc` | `charts/mpc/` | Threshold signatures (TSS) |
| `net` | `charts/net/` | Network infrastructure |
| `edge` | `charts/edge/` | Edge compute |

## Repository Structure

```
charts/
  agents/        # Chart.yaml, templates/, values.yaml
  analytics/
  bootnode/
  cloud/
  commerce/
  console/
  edge/
  gateway/
  gitops/
  iam/
  kms/
  llm/
  mpc/
  net/
  platform/
  storage/
.github/
  workflows/     # CI for lint, test, publish
ct.yaml          # Chart Testing configuration
README.md
LICENSE          # Apache 2.0
```

Each chart directory contains:
- `Chart.yaml` -- chart metadata, version, dependencies
- `values.yaml` -- default configuration values
- `templates/` -- Helm templates for K8s resources

## Installation

```bash
# Add Hanzo Helm repo
helm repo add hanzo https://charts.hanzo.ai
helm repo update

# Install individual chart
helm install hanzo-iam hanzo/iam -n hanzo --create-namespace
helm install hanzo-gateway hanzo/gateway -n hanzo
helm install hanzo-cloud hanzo/cloud -n hanzo

# Install full stack (umbrella)
helm install hanzo hanzo/stack -n hanzo --create-namespace \
  --set global.domain=yourdomain.com \
  --set iam.enabled=true \
  --set gateway.enabled=true \
  --set cloud.enabled=true \
  --set console.enabled=true
```

## Global Values

```yaml
global:
  domain: hanzo.ai
  imageRegistry: ghcr.io/hanzoai
  storageClass: do-block-storage
  tls:
    enabled: true
    issuer: letsencrypt-prod
```

## Development

```bash
# Lint all charts
helm lint charts/*

# Template a chart (debug)
helm template test charts/iam --debug

# Install with dry-run
helm install test charts/iam --dry-run --debug

# Run chart-testing
ct lint --config ct.yaml
ct install --config ct.yaml
```

## Architecture

```
                +-----------------+
                | hanzo-gateway   | (KrakenD)
                |  LoadBalancer   |
                +--------+--------+
                         |
     +-------------------+-------------------+
     |                   |                   |
+----v----+        +-----v-----+       +-----v-----+
|hanzo-iam|        |hanzo-cloud|       |hanzo-api  |
| (Auth)  |        | (AI/MCP)  |       | (Services)|
+----+----+        +-----+-----+       +-----+-----+
     |                   |                   |
     +-------------------+-------------------+
                         |
                +--------v--------+
                |hanzo-datastore  | (Shared)
                |PostgreSQL/Redis |
                +-----------------+
```

## Observability

- **VictoriaMetrics** for metrics (not Prometheus server -- charts expose /metrics endpoints)
- **Grafana** for dashboards
- **Hanzo Datastore** for logs and analytics

## Upstream Tracking

| Hanzo Chart | Upstream | Version |
|-------------|----------|---------|
| `gateway` | KrakenD CE | Latest 2.x |
| `gitops` | Agnost GitOps | hanzoai/gitops fork |
| `platform` | Dokploy | Latest main |
| `llm` | vLLM | Latest stable |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Chart not found | Repo not added | `helm repo add hanzo https://charts.hanzo.ai` |
| Template error | Missing values | Check `values.yaml` defaults |
| Image pull fail | Registry auth | Configure imagePullSecrets |
| CRD conflicts | Operator also managing | Use either operator OR charts, not both |

## Related Skills

- `hanzo/hanzo-operator.md` - K8s operator (CRD alternative to Helm)
- `hanzo/hanzo-universe.md` - Universe manifests (raw Kustomize)
- `hanzo/hanzo-stack.md` - Docker Compose local stack

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: helm, kubernetes, charts, deployment, infrastructure
**Prerequisites**: Helm 3, Kubernetes basics
