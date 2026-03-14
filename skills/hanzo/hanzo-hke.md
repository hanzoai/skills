# Hanzo HKE - Managed Kubernetes Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-universe.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-cloud.md`

## Overview

HKE (Hanzo Kubernetes Engine) is a planned **managed Kubernetes cluster service** with built-in DevOps workflows. The repository at `github.com/hanzoai/hke` was created on 2026-01-20 and is currently an empty placeholder -- no code, no branches, no files, no default branch. It is reserved for future development as Hanzo's dedicated K8s provisioning and lifecycle management product.

HKE is intended to complement the existing Hanzo infrastructure stack. Today, production Kubernetes is managed manually via `kubectl` and Kustomize manifests in the `universe` repo (`github.com/hanzoai/universe`), while application deployments go through the PaaS platform (`github.com/hanzoai/paas`). HKE will formalize cluster lifecycle management as a first-class service.

## Current Status

| Property | Value |
|----------|-------|
| Repo | `github.com/hanzoai/hke` |
| Created | 2026-01-20 |
| Status | **Empty** (no source code) |
| Default branch | None |
| License | None |
| Visibility | Public |
| Archived | No |
| Description | Managed Kubernetes clusters with built-in DevOps workflows |

The repository has been created and named but contains zero files. There are no releases, issues, pull requests, or CI workflows.

## Intended Purpose

Based on the repository description and the broader Hanzo infrastructure architecture:

### Cluster Provisioning
- Automated Kubernetes cluster creation across cloud providers (DigitalOcean DOKS, AWS EKS, GCP GKE)
- Standardized node pool configuration with resource presets
- Multi-region cluster topology support
- Cluster lifecycle management (create, scale, upgrade, destroy)

### Built-in DevOps Workflows
- Integrated CI/CD pipeline definitions per cluster
- GitOps-driven deployment workflows
- Automated image building and registry integration (GHCR, Hanzo Registry)
- Rolling, blue-green, and canary deployment strategies

### Multi-Tenant Cluster Management
- Organization-scoped cluster isolation via Hanzo IAM (`hanzo.id`)
- RBAC policy templates mapped to IAM roles
- Namespace provisioning per tenant with resource quotas
- Network policy enforcement between tenants

### Observability Integration
- Pre-configured monitoring stack (VictoriaMetrics, Grafana, Loki)
- Distributed tracing via Tempo
- OpenTelemetry Collector sidecar injection
- Health check and readiness probe templates

## Architecture Context

### Where HKE Fits in the Hanzo Stack

```
User Request
    |
    v
HKE API (future)        -- Cluster provisioning & lifecycle
    |
    v
Cloud Provider API       -- DOKS, EKS, GKE
    |
    v
K8s Clusters             -- Managed by HKE
    |
    +-- Universe Manifests   -- Service deployments (infra/k8s/)
    +-- PaaS Platform        -- Application deployments
    +-- KMS                  -- Secrets via KMSSecret CRDs
    +-- IAM                  -- Identity via OIDC JWT
```

### Current Production Infrastructure (without HKE)

Today, Hanzo runs two DOKS clusters managed manually:

| Cluster | IP | Purpose |
|---------|-----|---------|
| **hanzo-k8s** | `24.199.76.156` | All Hanzo services (IAM, KMS, Gateway, Console, Commerce, Chat, App) |
| **lux-k8s** | `24.144.69.101` | Lux blockchain (15 validators, KrakenD, markets, cloud) |

Cluster creation uses `doctl kubernetes cluster create`. Deployments use `kubectl kustomize . | kubectl apply -f -` from `universe/infra/k8s/`. Terraform configs exist in `universe/infra/terraform/` for DigitalOcean infrastructure but cluster provisioning is not yet automated through a service API.

## Expected Tech Stack

Based on Hanzo conventions and the existing infrastructure patterns:

| Layer | Expected Technology |
|-------|-------------------|
| Language | Go (consistent with Hanzo systems services) |
| API | REST + gRPC |
| Auth | Hanzo IAM OIDC (JWT with `owner` claim for org scoping) |
| Database | PostgreSQL via `hanzoai/sql` |
| Cache | Valkey/Redis via `hanzoai/kv` |
| Secrets | KMS (`kms.hanzo.ai`) via KMSSecret CRDs |
| Container Image | `ghcr.io/hanzoai/hke:<tag>` (`--platform linux/amd64`) |
| Manifests | Kustomize (not Helm, per Hanzo convention) |
| CI/CD | GitHub Actions or PaaS Tekton pipelines |
| IaC | Terraform (DigitalOcean provider `~> 2.34`) |

## When to Use

- **Not yet**: This repo has no code. It is a placeholder.
- For current K8s manifest management, use `hanzo/hanzo-universe.md`
- For application deployments on existing clusters, use `hanzo/hanzo-platform.md`
- For secrets management on K8s, use `hanzo/hanzo-kms.md`
- For cloud dashboard and AI model API, use `hanzo/hanzo-cloud.md`

## Related Hanzo Infrastructure

| Service | Repo | Purpose |
|---------|------|---------|
| Universe | `github.com/hanzoai/universe` | Production K8s manifests, Helm charts, Terraform, compose files |
| PaaS | `github.com/hanzoai/paas` | Application deployment platform (Agnost fork) |
| Platform | `github.com/hanzoai/platform` | PaaS UI (Dokploy fork) |
| KMS | `github.com/hanzoai/kms` | Secrets management (Infisical fork) |
| IAM | `github.com/hanzoai/iam` | Identity and access management (Casdoor fork) |
| Cloud | `github.com/hanzoai/cloud` | Cloud API backend |

## Related Skills

- `hanzo/hanzo-universe.md` -- Production K8s infrastructure monorepo
- `hanzo/hanzo-platform.md` -- PaaS platform (current deployment tool)
- `hanzo/hanzo-cloud.md` -- Cloud dashboard and AI model API
- `hanzo/hanzo-kms.md` -- Secret management with KMSSecret CRDs

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kubernetes, managed-k8s, devops, clusters, provisioning
**Prerequisites**: None (repo is empty)
