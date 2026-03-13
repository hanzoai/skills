# Hanzo Nchain - Composable Blockchain Node Infrastructure Operator

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-hke.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-cloud.md`

## Overview

Nchain is a **Kubernetes operator for deploying and scaling blockchain nodes**. Built with kubebuilder and controller-runtime, it defines custom resources (CRDs) for managing multi-protocol blockchain infrastructure declaratively. Supports Lux, Ethereum, Bitcoin, Cosmos, Solana, and Substrate via a pluggable protocol driver interface.

### Why Nchain?

- **Multi-protocol** -- Single operator handles Lux, Ethereum, Bitcoin, Cosmos, Solana, Substrate, and generic nodes
- **Composable CRDs** -- Network (top-level composer) creates NodeClusters, Chains, Indexers, Explorers, Bridges, and Gateways
- **Protocol drivers** -- Pluggable `Driver` interface generates container commands, env vars, volumes, health checks per chain
- **Production features** -- Rolling canary upgrades, seed restore from snapshots, automated snapshot schedules, startup gates, KMS secret sync
- **K8s-native** -- Standard kubebuilder patterns, leader election, health/readiness probes, secure metrics

### Tech Stack

- **Language**: Go 1.25
- **Framework**: kubebuilder / controller-runtime v0.23
- **CRD API**: `nchain.hanzo.ai/v1alpha1`
- **Image**: `ghcr.io/hanzoai/nchain:latest`
- **Binary**: `manager` (single binary operator)

## When to use

- Deploying blockchain validator/fullnode/archive clusters on Kubernetes
- Managing multi-chain networks (Lux subnets, Ethereum L2s, Cosmos zones)
- Running block explorers, indexers, and API gateways alongside nodes
- Automating chain data snapshots and seed-based bootstrap
- Operating cross-chain bridges under K8s lifecycle management

## Hard requirements

1. **Kubernetes cluster** with CRD support (v1.28+)
2. **controller-gen** for CRD manifest generation
3. **Go 1.25+** for building from source

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/nchain` |
| Go version | 1.25 |
| API group | `nchain.hanzo.ai/v1alpha1` |
| Image | `ghcr.io/hanzoai/nchain:latest` |
| Default branch | `main` |
| Health probe | `:8081/healthz` |
| Readiness probe | `:8081/readyz` |
| Leader election ID | `nchain.hanzo.ai` |
| Repo | `github.com/hanzoai/nchain` |

## One-file quickstart

### Deploy the operator

```bash
# Build and push
make docker-push IMG=ghcr.io/hanzoai/nchain:latest

# Generate CRDs and deploy
make deploy
```

### Create a Lux devnet

```yaml
apiVersion: nchain.hanzo.ai/v1alpha1
kind: NodeCluster
metadata:
  name: lux-devnet-validators
  namespace: lux-devnet
spec:
  protocol: lux
  networkID: "3"
  replicas: 5
  role: validator
  image:
    repository: ghcr.io/luxfi/node
    tag: v1.23.23
  ports:
    rpc: 9650
    p2p: 9651
  storage:
    storageClassName: do-block-storage
    size: 50Gi
  resources:
    requests:
      cpu: "1"
      memory: 2Gi
    limits:
      cpu: "2"
      memory: 4Gi
  consensus:
    algorithm: snow
    params:
      sybilProtectionEnabled: false
      allowPrivateIPs: true
```

### Create a full Network (composer)

```yaml
apiVersion: nchain.hanzo.ai/v1alpha1
kind: Network
metadata:
  name: lux-mainnet
spec:
  protocol: lux
  networkID: "1"
  clusters:
    - name: validators
      spec:
        replicas: 5
        role: validator
        image:
          repository: ghcr.io/luxfi/node
          tag: v1.23.23
  gateways:
    - name: api
      spec:
        replicas: 2
        nodeClusterRef: lux-mainnet-validators
        autoRoutes: true
        ingress:
          enabled: true
          hosts:
            - api.lux.network
          tls: true
```

## Core Concepts

### CRD Hierarchy

```
Network (top-level composer)
  |-- NodeCluster    Stateful node pods (validator, fullnode, archive, bootnode, sentry)
  |-- Chain          Blockchain/subnet/L2 definitions (genesis, chainID, VMID)
  |-- Indexer        Block indexing services
  |-- Explorer       Block explorer UIs (Blockscout, etc.)
  |-- Bridge         Cross-chain bridge relayers
  |-- Gateway        API gateway / RPC load balancer with rate limits
  |-- Cloud          Cloud management platform (bootnode API + web UI)
```

### Controllers (8 reconcilers)

| Controller | CRD | Purpose |
|-----------|-----|---------|
| `NodeClusterReconciler` | NodeCluster | Manages StatefulSets for node pods |
| `ChainReconciler` | Chain | Tracks blockchain lifecycle and block height |
| `IndexerReconciler` | Indexer | Deploys and configures indexing services |
| `ExplorerReconciler` | Explorer | Deploys block explorer instances |
| `BridgeReconciler` | Bridge | Manages cross-chain bridge relayers |
| `GatewayReconciler` | Gateway | Deploys API gateway with auto-routing and rate limits |
| `NetworkReconciler` | Network | Top-level composer creating all child resources |
| `CloudReconciler` | Cloud | Cloud management platform deployment |

### Protocol Drivers

Each protocol implements the `Driver` interface:

```go
type Driver interface {
    Name() string
    DefaultImage() string
    BuildCommand(spec *NodeClusterSpec) (command, args []string)
    BuildEnv(spec *NodeClusterSpec) []corev1.EnvVar
    HealthEndpoint(spec *NodeClusterSpec) (path string, port int32)
    DefaultPorts() PortConfig
    RecommendedResources(role string) (requests, limits corev1.ResourceList)
    // ... plus volumes, init containers, config maps
}
```

| Protocol | File | Default Image |
|----------|------|---------------|
| Lux | `internal/protocol/lux.go` | `ghcr.io/luxfi/node` |
| Ethereum | `internal/protocol/ethereum.go` | Geth-based |
| Bitcoin | `internal/protocol/bitcoin.go` | Bitcoin Core |
| Cosmos | `internal/protocol/cosmos.go` | Cosmos SDK |
| Solana | `internal/protocol/solana.go` | Solana validator |
| Generic | `internal/protocol/generic.go` | User-specified |

### NodeCluster Roles

| Role | Description |
|------|-------------|
| `validator` | Full consensus participation (default) |
| `fullnode` | Validates but doesn't propose blocks |
| `archive` | Full historical state retention |
| `bootnode` | Network bootstrap / peer discovery |
| `sentry` | DDoS protection proxy for validators |

## Directory structure

```
github.com/hanzoai/nchain/
    Dockerfile                          # Multi-stage build (distroless)
    Makefile                            # build, test, manifests, docker-build, deploy
    api/v1alpha1/
        chain_types.go                  # Chain CRD (blockchain/subnet/L2)
        nodecluster_types.go            # NodeCluster CRD (node pod management)
        network_types.go                # Network CRD (top-level composer)
        bridge_types.go                 # Bridge CRD (cross-chain relayer)
        explorer_types.go               # Explorer CRD (block explorer)
        gateway_types.go                # Gateway CRD (API gateway)
        indexer_types.go                # Indexer CRD (block indexing)
        cloud_types.go                  # Cloud CRD (management platform)
        common_types.go                 # Shared types (ports, storage, consensus, etc.)
        groupversion_info.go            # API group registration
        zz_generated_deepcopy.go        # Auto-generated DeepCopy methods
    cmd/
        main.go                         # Operator entrypoint (registers all 8 controllers)
    config/
        crd/                            # Generated CRD YAML manifests
        samples/
            lux-mainnet.yaml            # Full Lux mainnet Network example
            nodecluster-standalone.yaml # Standalone NodeCluster + Gateway
            ethereum-node.yaml          # Ethereum node example
            cloud-multi-brand.yaml      # Multi-brand Cloud deployment
    internal/
        controller/
            nodecluster_controller.go   # NodeCluster reconciler
            chain_controller.go         # Chain reconciler
            network_controller.go       # Network reconciler
            bridge_controller.go        # Bridge reconciler
            explorer_controller.go      # Explorer reconciler
            gateway_controller.go       # Gateway reconciler
            indexer_controller.go       # Indexer reconciler
            cloud_controller.go         # Cloud reconciler
            helpers.go                  # Shared controller utilities
            predicates.go               # Event filter predicates
        manifests/
            builder.go                  # K8s manifest builder (StatefulSet, Service, etc.)
            labels.go                   # Label helpers
            mutate.go                   # CreateOrUpdate mutation helpers
        protocol/
            driver.go                   # Driver interface definition
            registry.go                 # Protocol driver registry
            lux.go                      # Lux protocol driver
            ethereum.go                 # Ethereum protocol driver
            bitcoin.go                  # Bitcoin protocol driver
            cosmos.go                   # Cosmos protocol driver
            solana.go                   # Solana protocol driver
            generic.go                  # Generic fallback driver
        status/                         # Status tracking utilities
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| CRDs not found | Manifests not applied | Run `make manifests && make deploy` |
| Leader election contention | Multiple operator replicas | Only one replica should run, or enable `--leader-elect` |
| Nodes stuck in Pending | Missing StorageClass | Verify `storageClassName` exists in cluster |
| Protocol not recognized | Invalid protocol enum | Must be one of: lux, ethereum, bitcoin, cosmos, substrate, generic |
| Snapshot restore fails | S3 endpoint unreachable | Check `seedRestore.objectStoreURL` and network policies |

## Related Skills

- `hanzo/hanzo-hke.md` - Managed Kubernetes Engine (placeholder)
- `hanzo/hanzo-platform.md` - PaaS platform for service deployment
- `hanzo/hanzo-cloud.md` - Cloud dashboard and console

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kubernetes, operator, blockchain, lux, ethereum, bitcoin, cosmos, solana, infrastructure
**Prerequisites**: Kubernetes 1.28+, Go 1.25+, controller-gen
