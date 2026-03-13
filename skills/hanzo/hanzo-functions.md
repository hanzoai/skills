# Hanzo Functions - Serverless Compute Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-runtime.md`

## Overview

Hanzo Functions is a **Kubernetes-native serverless platform** for event-driven workloads with GPU support. Go codebase (module `github.com/fission/fission`, Fission fork), runs as a set of K8s controllers and pods. Supports Python, Go, Node.js, Rust, and custom container runtimes with auto-scaling, cold start optimization, and canary deployments. License: Apache-2.0.

### Why Hanzo Functions?

- **Kubernetes native**: Uses CRDs for function, environment, trigger, and package resources
- **GPU support**: NVIDIA GPU scheduling for AI/ML inference functions
- **Multiple runtimes**: Python 3.9-3.12, Go 1.20-1.22, Node.js 18-22, Rust, custom containers
- **Event sources**: HTTP, Kafka, RabbitMQ, Cron, Kubernetes watches, custom triggers
- **Cold start optimization**: Sub-100ms cold starts with pool-based prewarming
- **Canary releases**: Blue/green deployments with traffic splitting

### Tech Stack

- **Language**: Go 1.24+
- **Build**: Make + goreleaser
- **K8s**: CRDs, controller-runtime, KEDA (autoscaling)
- **CLI**: spf13/cobra (`fission-cli`)
- **Helm**: `charts/fission-all`
- **CI**: GitHub Actions, Skaffold
- **Observability**: OpenTelemetry, Prometheus, InfluxDB

### OSS Base

Repo: `hanzoai/functions` (Fission fork). Default branch: `main`.

## When to use

- Deploy event-driven functions on Kubernetes
- AI/ML inference with GPU-accelerated serverless functions
- Webhook handlers, API endpoints, or scheduled jobs
- Message queue processing (Kafka, RabbitMQ triggers)
- Kubernetes event-driven automation (watch triggers)

## Hard requirements

1. **Kubernetes cluster** (1.28+) with CRD support
2. **Helm 3** for installation
3. **Container registry** accessible from the cluster (for function builds)
4. **NVIDIA GPU operator** (optional, for GPU functions)

## Quick reference

| Item | Value |
|------|-------|
| Go Module | `github.com/fission/fission` |
| Go Version | 1.24+ |
| CLI Binary | `fission` |
| Helm Chart | `charts/fission-all` |
| License | Apache-2.0 |
| Repo | `github.com/hanzoai/functions` |
| Default Branch | `main` |
| CRDs | `crds/v1/` |
| Skaffold | `skaffold.yaml` |

## One-file quickstart

### Helm install

```bash
helm repo add hanzo https://charts.hanzo.ai
helm install functions hanzo/functions
```

### CLI install

```bash
# macOS
brew install hanzoai/tap/hanzo-fn

# Linux
curl -sSL https://get.hanzo.ai/fn | bash
```

### Deploy a function

```python
# handler.py
def handler(context, event):
    name = event.body.get("name", "World")
    return {"statusCode": 200, "body": f"Hello, {name}!"}
```

```bash
fission env create --name python --image fission/python-env:latest
fission function create --name hello --env python --code handler.py
fission function test --name hello
```

## Core Concepts

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Hanzo Functions                         │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────────┐  │
│  │ Router   │──>│ Executor │──>│ Function Pods           │  │
│  │ (gateway)│   │          │   │  ┌─────┐ ┌─────┐       │  │
│  └──────────┘   └──────────┘   │  │ Fn  │ │ Fn  │ ...   │  │
│       │                        │  └─────┘ └─────┘       │  │
│       │         ┌──────────┐   └────────────────────────┘  │
│       └────────>│ KEDA     │                                │
│                 │ Scaler   │   ┌────────────────────────┐  │
│                 └──────────┘   │ Builder                 │  │
│                                │ (source -> container)   │  │
│  ┌──────────┐                  └────────────────────────┘  │
│  │ Fetcher  │  ┌──────────┐   ┌────────────────────────┐  │
│  │ (code    │  │ Storage  │   │ TimerTrigger / MQ      │  │
│  │  loader) │  │ Service  │   │ / KubeWatcher          │  │
│  └──────────┘  └──────────┘   └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### CRD Resources

| CRD | Purpose |
|-----|---------|
| `Function` | Function code, entry point, resource limits |
| `Environment` | Runtime image (Python, Go, Node.js, etc.) |
| `Package` | Source or deployment archive |
| `HTTPTrigger` | HTTP route -> function mapping |
| `MessageQueueTrigger` | Kafka/RabbitMQ -> function |
| `TimeTrigger` | Cron schedule -> function |
| `KubernetesWatchTrigger` | K8s resource events -> function |
| `CanaryConfig` | Traffic splitting for canary deployments |

### Components (cmd/)

| Binary | Purpose |
|--------|---------|
| `fission-bundle` | Main server (router, executor, builder mgr, all-in-one) |
| `fission-cli` | CLI tool (`fission` command) |
| `builder` | Builds source packages into deployable containers |
| `fetcher` | Fetches function code into pods at runtime |
| `reporter` | Usage reporting |
| `preupgradechecks` | Pre-upgrade validation |

### Supported Runtimes

| Runtime | Versions | GPU Support |
|---------|----------|-------------|
| Python | 3.9, 3.10, 3.11, 3.12 | Yes |
| Go | 1.20, 1.21, 1.22 | Yes |
| Node.js | 18, 20, 22 | No |
| Rust | 1.75+ | Yes |
| Custom | Any Docker image | Yes |

### Directory Structure

```
functions/
  Makefile               # Build: check, build-fission-cli, codegen, release
  .goreleaser.yml        # GoReleaser config for multi-platform builds
  skaffold.yaml          # Skaffold dev/deploy config
  kind.yaml              # KinD cluster config for local dev
  go.mod                 # github.com/fission/fission, Go 1.24
  cmd/
    fission-bundle/      # Main server binary
    fission-cli/         # CLI binary
    builder/             # Source builder
    fetcher/             # Code fetcher
    reporter/            # Usage reporter
    preupgradechecks/    # Upgrade validator
  pkg/
    apis/                # CRD type definitions (core/v1)
    router/              # HTTP request routing
    executor/            # Function pod management
    builder/             # Build pipeline
    buildermgr/          # Builder manager
    fetcher/             # Code fetch logic
    fission-cli/         # CLI command implementations
    mqtrigger/           # Message queue trigger handler
    kubewatcher/         # K8s watch trigger handler
    timer/               # Cron trigger handler
    cache/               # In-memory caching
    storagesvc/          # Archive storage
    throttler/           # Request throttling
    webhook/             # Admission webhooks
    generated/           # Generated client code
    crd/                 # CRD client helpers
    healthcheck/         # Health/readiness probes
    utils/               # Shared utilities
    plugin/              # Plugin interface
    publisher/           # Event publisher
    tracker/             # Request tracking
  crds/v1/               # CRD YAML manifests
  charts/fission-all/    # Helm chart
  hack/                  # Code generation, release scripts
  test/                  # Integration and e2e tests
  tools/                 # Doc generation, CRD ref docs
  .github/               # GitHub Actions workflows
```

### Development

```bash
# Run tests
make test-run

# Build CLI
make build-fission-cli

# Generate CRDs
make generate-crds

# Code generation (client, deepcopy)
make codegen

# Local dev with Skaffold + KinD
make skaffold-deploy SKAFFOLD_PROFILE=kind
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Function cold start slow | Pool not prewarmed | Increase `minScale` on function or enable pool manager |
| Build failures | Registry not accessible | Check `REGISTRY_URL` in Helm values, verify push access |
| CRD not found | CRDs not installed | Run `kubectl apply -k crds/v1` |
| GPU function pending | No GPU nodes | Verify NVIDIA GPU operator installed, nodes have GPU labels |
| MQ trigger not firing | Kafka/RabbitMQ connection | Check MQ trigger config, verify broker connectivity |
| CLI connection refused | Router not running | Verify fission-router pod is healthy, check port-forward |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-universe.md` - K8s infrastructure
- `hanzo/hanzo-runtime.md` - Container runtime
- `hanzo/hanzo-stream.md` - Kafka wire protocol (MQ trigger source)
- `hanzo/hanzo-pubsub.md` - NATS messaging (event source)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: serverless, faas, functions, kubernetes, gpu, event-driven, fission
**Prerequisites**: Kubernetes 1.28+, Helm 3, container registry
