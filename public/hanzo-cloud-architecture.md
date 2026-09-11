---
name: hanzo-cloud-architecture
description: The canonical Hanzo Cloud architecture — one Go binary on zip, ZAP-only transport, plugin registry, embedded console, three deployment modes. Read this before touching any hanzoai service.
---

# Hanzo Cloud Architecture — One Binary, ZAP-Native, Plugin-Composed

**Category**: Hanzo Ecosystem
**Canonical spec**: HIP-0106 (one binary), HIP-0114 (ZAP transport), HIP-0116 (plugin/VM model), HIP-0117 (cloud-in-a-box), HIP-0004 (AI gateway) — `github.com/hanzoai/hips`
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-zap.md`, `hanzo/hanzo-gateway.md`, `hanzo/hanzo-o11y.md`, luxfi/skills `lux-zap-substrate`

## When to Use This Skill

- Working anywhere in `hanzoai/cloud` or any service it embeds (IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai, ...)
- Adding a new Hanzo service or subsystem
- Choosing a transport, RPC layer, or storage backend for Hanzo code
- Deploying Hanzo Cloud (laptop, k3s bootstrap, or BYO Kubernetes)
- Reading any older doc that mentions multi-pod services or gRPC — this skill is the current truth

## The Architecture in One Paragraph

`hanzoai/cloud` is ONE Go binary that embeds every Hanzo-native subsystem —
IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai, and the rest — as
packages behind a single plugin registry. The core is a **977-LOC shim** on
**zip** (`github.com/hanzoai/zip`, the ZAP-native high-performance server,
Fiber v3/fasthttp, HIP-0105); the subsystems are **74,561 LOC of plugins**
under `clients/*` — a 76:1 plugin-to-core ratio. Transport is **ZAP and
nothing else** (HIP-0114): zero gRPC, zero protobuf in Hanzo code. The same
artifact powers `api.hanzo.ai` and every white-label resold cloud surface;
brand, enabled subsystems, and tenant scope are deployment configuration.

## Status: shipped vs decided

| Piece | Status |
|-------|--------|
| One binary (`hanzoai/cloud`, subsystems embedded) | **SHIPPED** |
| ZAP-only transport, zero gRPC/protobuf in Hanzo code | **SHIPPED** |
| zip as the one server framework | **SHIPPED** |
| Console SPA `go:embed`'d (303.6 MiB one-binary, fail-hard Dockerfile) | **SHIPPED** |
| console.hanzo.ai per-product metered usage + 402 balance-gating | **SHIPPED** |
| `helm install` (charts at `cloud/helm/cloud`) | **SHIPPED** |
| Plugin **VM** formalization (out-of-process VMs via lpm, HIP-0116) | **DECIDED** (HIP Draft, staged) |
| `cloud cluster init` k3s bootstrap (HIP-0117 mode 2) | **DECIDED** (HIP Draft, staged) |
| Visor per-tenant cross-provider autoscaling | **DECIDED** (staged) |

## Core Concepts

### 1. The plugin registry (`cloud.Register`)

Every subsystem registers into the cloud registry via `init()`:

```go
// build.go — the whole extension API
func Register(name string, order int, mount MountFunc)
func RegisterWithShutdown(name string, order int, mount MountFunc, shutdown ShutdownFunc)
```

- Subsystems live in `clients/<name>/` (e.g. `clients/kmssvc`, `clients/o11y`,
  `clients/console`, `clients/tasksvc`, `clients/notify`, `clients/kafka`).
- The subsystem set is defined ONCE in the `subsystems` bundle; `cmd/cloud`
  and `cmd/hanzo` blank-import it. `cloud.Serve(nil)` honors `cfg.Enable`
  from flags/env.
- `cmd/cloud` is the full-surface entrypoint; production deployments name
  their subsystem set explicitly. Embedded subsystems ship
  **disabled-by-default and fail closed**: a subsystem without its required
  config refuses to mount rather than degrading silently.

### 2. ZAP-only transport (HIP-0114)

- ZERO gRPC and ZERO protobuf anywhere in Hanzo code. The transport between
  subsystems, plugins, and services is ZAP — zero-copy, ZAP-typed Go
  interfaces in-process, ZAP RPC over the wire when split.
- The ONLY protobuf in the stack is `zap2pb` (`~/work/zap/zap2pb`, ~30 LOC)
  at the OTel interop edge (OTLP/OpAMP). `pb2zap` / `zap2pb` / `zapc` handle
  the pb↔zap boundary; services hold only ZAP types.
- zip serves ZAP as the primary listener with HTTP as an extra:
  `app.Listen(":9653", "http://:8080")`.

```go
// ❌ NEVER in Hanzo code
import "google.golang.org/grpc"        // no gRPC
import "google.golang.org/protobuf"    // no protobuf

// ✅ The one server framework, ZAP-typed handlers
import "github.com/hanzoai/zip"
```

### 3. Storage and consensus

- **SQLite-primary, per-tenant** — the default store for every subsystem.
  PostgreSQL only for production multi-instance needs; never default to it
  for local dev.
- **datastore** (ClickHouse) is the ONE non-Go exception — telemetry and
  analytics columns.
- **Consensus/replication** = Lux Quasar (`luxfi/consensus`) + `luxfi/zapdb`.
  NO ZooKeeper, NO raft libraries, NO etcd anywhere in the stack.

### 4. Plugin/VM model (HIP-0116 — decided, staged)

Every `hanzoai/<repo>` service builds BOTH ways from one codebase:

1. **Standalone binary** — its own repo, its own release.
2. **Plugin VM** for the cloud host — the `luxfi/node` VM-host model: an
   independently-built binary the host supervises and dials **over ZAP**.
   Distributed and installed via `luxfi/lpm`.

NOT build tags. NOT gRPC. NOT dlopen. A plugin VM is simply a subsystem
that lives in another process; in-process and out-of-process resolve through
the same ZAP-typed contract. Each VM tracks its own replication state on
zapdb under Quasar.

### 5. Cloud-in-a-Box — one binary, three modes (HIP-0117)

| Mode | Command | Topology |
|------|---------|----------|
| 1 | `cloud serve` | Single process, NO Kubernetes. All subsystems in-process, per-tenant SQLite, datastore as subprocess or gracefully absent. Laptop/edge/single-node. |
| 2 | `cloud cluster init` | Binary FETCHES k3s (does not embed it), bootstraps the cluster, installs the operator, hands reconciliation to `services.hanzo.ai` CRs. Bare metal → HA in one command. (staged) |
| 3 | `helm install` | BYO Kubernetes: `cloud/helm/cloud` deploys the same image into an existing cluster. |

### 6. The edges: gateway, console, visor

- **gateway** (HIP-0004, `api.hanzo.ai`) — the unified AI provider interface:
  one OpenAI-compatible surface over 100+ providers, BYO AI/keys.
- **console** (`console.hanzo.ai`) — the unified dashboard: per-product
  metered usage and 402 balance-gating. The console SPA is `go:embed`'d into
  the cloud binary (real `hanzoai/console` static build; the Dockerfile
  fails hard if the SPA is missing — no placeholder builds).
- **visor** — fleet autoscaler across any nodes and any cloud provider
  (BYO provider). 8 Kubernetes clusters operated today.

### 7. CI/CD

- **Gitea Actions on git.hanzo.ai is the native CI** — GH-Actions-compatible,
  runs the same `.github/workflows` on Hanzo runners. GitHub is the OSS
  mirror. No GitHub-hosted builders, ever.
- Images: `ghcr.io/hanzoai/*` for Hanzo, `ghcr.io/luxfi/*` for Lux — never mix.
- The operator reconciles `services.hanzo.ai` CRs; devs ship via CI/CD, not
  kubectl.

## Quick Reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/cloud` (`~/work/hanzo/cloud`) |
| Core | 977-LOC shim on zip + `cloud.Register` registry |
| Plugins | `clients/*` — 74,561 LOC, 76:1 plugin:core |
| Server framework | `github.com/hanzoai/zip` (`~/work/zap/zip`) |
| Transport | ZAP only (HIP-0114); ZAP port 9653 |
| pb boundary | `zap2pb`/`pb2zap`/`zapc` at the OTel edge only |
| Storage | SQLite per-tenant; datastore (ClickHouse) for telemetry |
| Consensus | Lux Quasar + zapdb — no ZooKeeper/raft/etcd |
| Entrypoints | `cmd/cloud` (full surface), `cmd/hanzo` (subcommand dispatcher) |
| Helm chart | `cloud/helm/cloud` |
| One-binary size | 303.6 MiB with embedded console SPA |
| AI surface | gateway HIP-0004 at api.hanzo.ai |
| Metering | console.hanzo.ai, 402 balance-gating |

### Key Guidelines

```
✅ DO: add functionality as a clients/* plugin registered via cloud.Register
✅ DO: hold ZAP types end-to-end; convert to pb only at the OTel edge
✅ DO: default to SQLite per-tenant; fail closed when config is missing
✅ DO: build each service standalone AND as a subsystem (HIP-0116)

❌ DON'T: introduce gRPC, protobuf, or a second RPC stack
❌ DON'T: add a new standalone pod for something the binary already embeds
❌ DON'T: reach for ZooKeeper/etcd/raft — Quasar + zapdb is the substrate
❌ DON'T: build images locally — Gitea Actions CI on our runners only
```

## Anti-Patterns

❌ **"Let's split this subsystem into its own microservice"** — the split
already exists: it's the same code as a plugin VM over ZAP (HIP-0116).
Process topology is deployment configuration, not architecture.

❌ **"Add a proto file for this internal API"** — Hanzo code holds ZAP types
only. If you are writing `.proto`, you are at the OTel edge or you are wrong.

❌ **"docker compose up postgres for local dev"** — `cloud serve` runs the
whole cloud from one binary with SQLite. That IS local dev.

## Related Skills

- `skills/hanzo/hanzo-cloud.md` — the cloud binary operationally (config, URLs, deploy)
- `skills/hanzo/hanzo-zap.md` — ZAP protocol details and the MCP mapping
- `skills/hanzo/hanzo-gateway.md` — the HIP-0004 AI gateway at api.hanzo.ai
- `skills/hanzo/hanzo-o11y.md` — observability, embedded o11y subsystem
- luxfi/skills `lux-zap-substrate` — the Lux substrate (Quasar + zapdb + zip + lpm) Hanzo builds on
