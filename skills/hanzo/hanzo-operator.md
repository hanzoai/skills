# Hanzo Operator - Kubernetes Operator for Hanzo Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-charts.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-platform.md`

## Overview

Hanzo Operator is a **unified Kubernetes operator** that manages all Hanzo production infrastructure declaratively via 7 CRDs under the `hanzo.ai/v1alpha1` API group. Built with Kubebuilder v4 and controller-runtime, it reconciles high-level Hanzo resource specs into the full set of underlying K8s objects (Deployments, StatefulSets, Services, Ingress, HPA, PDB, NetworkPolicy, CronJobs, KMSSecret CRs).

### Why Hanzo Operator?

- **Declarative infrastructure**: Define an entire Hanzo service, datastore, or network in one CR
- **7 CRDs**: HanzoService, HanzoDatastore, HanzoGateway, HanzoMPC, HanzoNetwork, HanzoIngress, HanzoPlatform
- **Idempotent reconciliation**: Uses `ctrl.CreateOrUpdate` with `MutateFuncFor` for safe convergence
- **KMS integration**: Automatically creates `kms.hanzo.ai/v1alpha1 KMSSecret` CRs for secret management
- **Owner references**: Full GC cascade -- deleting the parent CR cleans up all children
- **Phase lifecycle**: Pending, Creating, Running, Degraded, Deleting

### Tech Stack

- **Language**: Go 1.26
- **Framework**: Kubebuilder v4, controller-runtime v0.23.1
- **K8s API**: k8s.io/* v0.35.0
- **Image**: `ghcr.io/hanzoai/operator:latest`
- **Namespace**: `hanzo-operator-system`
- **Stats**: 28 Go source files, ~7,400 lines, 13,022 lines CRD YAML

## When to use

- Deploying Hanzo services to Kubernetes declaratively
- Managing API gateways (KrakenD-based) via HanzoGateway
- Provisioning datastores (PostgreSQL, Redis, MongoDB) via HanzoDatastore
- Running blockchain validator networks via HanzoNetwork
- Managing multi-party computation infrastructure via HanzoMPC
- Orchestrating a full Hanzo platform stack via HanzoPlatform

## Hard requirements

1. **Kubernetes** v1.11.3+ cluster with CRD support
2. **cert-manager** for TLS (used by HanzoIngress)
3. **KMS operator** (`kms.hanzo.ai`) for KMSSecret reconciliation
4. **RBAC permissions**: core, apps, autoscaling, batch, networking, policy, hanzo.ai/*, kms.hanzo.ai/*

## Quick reference

| Item | Value |
|------|-------|
| Language | Go 1.26 |
| Framework | Kubebuilder v4, controller-runtime v0.23.1 |
| API Group | `hanzo.ai/v1alpha1` |
| CRD Count | 7 |
| Image | `ghcr.io/hanzoai/operator:latest` |
| Namespace | `hanzo-operator-system` |
| Repo | `github.com/hanzoai/operator` |
| Branch | `main` |
| License | Apache 2.0 |

## CRD Reference

| CRD | Short Name | Creates |
|-----|------------|---------|
| **HanzoService** | `hsvc` | Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, KMSSecret |
| **HanzoDatastore** | `hds` | StatefulSet, headless Service, PVC, CronJob (backup), KMSSecret |
| **HanzoGateway** | `hgw` | Deployment, Service, ConfigMap (KrakenD config), Ingress |
| **HanzoMPC** | `hmpc` | StatefulSet, headless Service, Dashboard Deployment, Cache Deployment, Ingress |
| **HanzoNetwork** | `hnet` | StatefulSet (validators), Deployments (bootnode/indexer/explorer/bridge), Services, ConfigMaps |
| **HanzoIngress** | `hing` | Multiple Ingress resources with cert-manager TLS |
| **HanzoPlatform** | `hplat` | Child CRDs (composes all of the above) |

## Repository Structure

```
api/v1alpha1/
  common_types.go              # Shared types (ContainerSpec, IngressSpec, etc.)
  hanzoservice_types.go        # HanzoService CRD spec
  hanzodatastore_types.go      # HanzoDatastore CRD spec
  hanzogateway_types.go        # HanzoGateway CRD spec
  hanzompc_types.go            # HanzoMPC CRD spec
  hanzonetwork_types.go        # HanzoNetwork CRD spec
  hanzoingress_types.go        # HanzoIngress CRD spec
  hanzoplatform_types.go       # HanzoPlatform CRD spec
  types.go                     # Additional shared types
  zz_generated.deepcopy.go     # Generated DeepCopy methods
cmd/
  main.go                      # Manager entry point
internal/
  controller/
    hanzoservice_controller.go   # HanzoService reconciler
    hanzodatastore_controller.go # HanzoDatastore reconciler
    hanzogateway_controller.go   # HanzoGateway reconciler
    hanzompc_controller.go       # HanzoMPC reconciler
    hanzonetwork_controller.go   # HanzoNetwork reconciler
    hanzoingress_controller.go   # HanzoIngress reconciler
    hanzoplatform_controller.go  # HanzoPlatform reconciler
    predicates.go                # Event filters (createOrUpdatePred, etc.)
    helpers.go                   # Shared controller utilities
  manifests/                     # K8s object builders (builder, labels, mutate)
  status/                        # Condition management
  metrics/                       # Prometheus metrics
  config/                        # Feature gates
config/
  crd/bases/                     # Generated CRD YAML (13k lines)
  rbac/                          # ClusterRole, bindings
  manager/                       # Deployment template
  samples/                       # Sample CRs for all 7 CRDs
Dockerfile                       # Multi-stage Go build (Alpine)
Makefile                         # Build, test, deploy targets
```

## Build and Deploy

```bash
# Regenerate CRDs and DeepCopy
make manifests generate

# Build local binary
make build

# Run tests
make test

# Build Docker image
make docker-build IMG=ghcr.io/hanzoai/operator:latest

# Build and push (multi-platform)
docker buildx build --platform linux/amd64 --push -t ghcr.io/hanzoai/operator:latest .

# Install CRDs into cluster
make install

# Deploy operator
make deploy IMG=ghcr.io/hanzoai/operator:latest

# Apply sample CRs
kubectl apply -k config/samples/
```

### Production Deployment

Universe manifests at `~/work/hanzo/universe/infra/k8s/hanzo-operator/`:

```bash
kubectl apply -k universe/infra/k8s/hanzo-operator/
```

## Key Patterns

- **Predicate filtering**: `createOrUpdatePred`, `updateOrDeletePred`, `statusChangePred` control which events trigger reconciliation
- **CreateOrUpdate with MutateFuncFor**: Idempotent object reconciliation -- creates if missing, patches if changed
- **Owner references**: All child objects have owner refs for automatic garbage collection
- **Phase lifecycle**: Resources transition through Pending, Creating, Running, Degraded, Deleting
- **KMSSecret delegation**: Operator creates `kms.hanzo.ai/v1alpha1 KMSSecret` CRs; the existing KMS operator handles actual secret sync from kms.hanzo.ai

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| RBAC errors on deploy | Missing cluster-admin | Grant cluster-admin or use `make deploy` |
| CRD not found | CRDs not installed | `make install` first |
| KMSSecret not syncing | KMS operator not running | Deploy KMS operator to cluster |
| Reconcile loop | Spec drift | Check object mutate functions in manifests/ |
| Image pull errors | GHCR auth | Ensure imagePullSecrets configured |

## Related Skills

- `hanzo/hanzo-charts.md` - Helm charts (alternative to operator CRDs)
- `hanzo/hanzo-universe.md` - Universe manifests (where operator is deployed)
- `hanzo/hanzo-kms.md` - KMS for secret management (KMSSecret CRs)
- `hanzo/hanzo-platform.md` - PaaS platform (managed by HanzoPlatform CRD)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kubernetes, operator, crds, kubebuilder, infrastructure, go
**Prerequisites**: Kubernetes, Go, Kubebuilder concepts
