# Hanzo Cloud - The Unified One-Binary Cloud

**Category**: Hanzo Ecosystem
**Canonical spec**: HIP-0106 (one binary), HIP-0114 (ZAP), HIP-0116 (plugin VMs), HIP-0117 (deploy modes)
**Related Skills**: `hanzo-cloud-architecture/SKILL.md` (read first), `hanzo/hanzo-zap.md`, `hanzo/hanzo-gateway.md`, `hanzo/hanzo-console.md`, `hanzo/hanzo-iam.md`

## Overview

`hanzoai/cloud` is **ONE Go binary** that embeds every Hanzo-native subsystem
— IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai (the former LLM
control plane), and the rest — behind a single plugin registry. It is built
on **zip** (`github.com/hanzoai/zip`, the ZAP-native high-performance server):
a **977-LOC shim** plus `cloud.Register(name, order, mount)` and **74,561 LOC
of plugins** under `clients/*` (76:1 plugin:core). The console SPA is
`go:embed`'d (real `hanzoai/console` static build; 303.6 MiB one-binary;
fail-hard Dockerfile). The same artifact powers `api.hanzo.ai`,
`api.lux.cloud`, `api.zoo.cloud`, and every white-label surface — brand,
enabled subsystems, and tenant scope are deployment configuration.

**Naming note**: the former "AI provider management platform" content that
used to live at `hanzoai/cloud` was renamed to `hanzoai/ai` and mounts as the
`ai` subsystem inside this binary. White-label fork target is `hanzoai/cloud`.

## When to use

- Working on any embedded subsystem (`clients/*`) or the cloud core
- Standing up a full Hanzo cloud locally (`cloud serve` — no Kubernetes)
- Deploying or white-labeling a cloud surface
- Managing AI providers/keys (the `ai` subsystem + gateway HIP-0004)
- Wiring per-product metered usage and 402 balance-gating via console

## Hard requirements

1. **ZAP only** (HIP-0114): zero gRPC, zero protobuf in Hanzo code. The pb
   boundary is `zap2pb`/`pb2zap`/`zapc` at the OTel edge exclusively.
2. **zip is the one server framework** — no Beego, no Gin, no echo, no
   second framework.
3. **SQLite-primary per-tenant** storage; PostgreSQL only for production
   multi-instance. Never default to PostgreSQL for local dev.
4. **Subsystems fail closed**: an embed without its required config refuses
   to mount. Disabled-by-default in production; deployments enable
   explicitly via `cfg.Enable` (flags/env).
5. **IAM identity from hanzo.id**; org from the JWT `owner` claim; all data
   scoped to org.

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/cloud` (`~/work/hanzo/cloud`) |
| Entrypoints | `cmd/cloud` (full surface), `cmd/hanzo` (subcommand dispatcher) |
| Registry | `cloud.Register` / `cloud.RegisterWithShutdown` (`build.go`) |
| Subsystem set | `subsystems` bundle (one source of truth, blank-imported) |
| Plugins | `clients/*` (admin, ai, billing, console, iamsvc, kmssvc, kafka, notify, o11y, tasksvc, ...) |
| Server | `github.com/hanzoai/zip` — ZAP primary + HTTP extra, one `Listen` |
| Console | `go:embed` SPA (`webui.go`, `clients/console`) |
| Helm chart | `cloud/helm/cloud` |
| Image | `ghcr.io/hanzoai/cloud` (Gitea Actions CI, GH mirror) |
| API surface | `api.hanzo.ai` (`/v1/...`, never `/api/` prefixes) |
| Dashboard | `console.hanzo.ai` (usage metering, 402 balance-gating) |

## Architecture

```
                 ONE binary: cloud
┌───────────────────────────────────────────────────┐
│ zip (ZAP-native server, Fiber v3/fasthttp)        │
│ cloud core: 977-LOC shim + Register() registry    │
├───────────────────────────────────────────────────┤
│ clients/* plugins (74,561 LOC)                    │
│  iamsvc │ kmssvc │ o11y │ tasksvc │ notify        │
│  pubsub │ kafka  │ console (embedded SPA) │ ai    │
│  billing │ admin │ analytics │ ...                │
├───────────────────────────────────────────────────┤
│ SQLite per-tenant   │ datastore (ClickHouse)      │
│ Quasar + zapdb (replication — no ZK/raft/etcd)    │
└───────────────────────────────────────────────────┘
        transport everywhere: ZAP (HIP-0114)
```

Each `hanzoai/<repo>` service still builds standalone; inside cloud it is a
subsystem, and (HIP-0116, staged) the same code runs as an out-of-process
**plugin VM** dialed over ZAP and distributed via `luxfi/lpm`. In-process vs
out-of-process is deployment topology, not architecture.

## Adding a subsystem

```go
// clients/example/example.go
package example

import "github.com/hanzoai/cloud"

func init() {
    cloud.Register("example", 50, mount) // name, mount order, mount func
}

func mount(app cloud.App, cfg cloud.Config) error {
    if cfg.ExampleDSN == "" {
        return cloud.ErrNotConfigured // fail closed, never degrade silently
    }
    app.Get("/v1/example/:id", handler) // /v1/, never /api/
    return nil
}
```

Then add it to the `subsystems` bundle — the ONE place the set is defined.

## Running

```bash
# The whole cloud on a laptop — no Kubernetes, SQLite per-tenant (HIP-0117 mode 1)
cloud serve

# Bootstrap a k3s HA cluster + operator + services.hanzo.ai CRs (mode 2, staged)
cloud cluster init

# BYO Kubernetes (mode 3)
helm install cloud ./helm/cloud
```

## API key usage (the `ai` subsystem + gateway)

```bash
export HANZO_API_KEY=sk-...

curl https://api.hanzo.ai/v1/chat/completions \
 -H "Authorization: Bearer ${HANZO_API_KEY}" \
 -H "Content-Type: application/json" \
 -d '{"model": "zen-70b", "messages": [{"role": "user", "content": "Hello"}]}'
```

The gateway (HIP-0004) is the unified AI provider interface — 100+ providers,
BYO keys. Usage is metered per product in console; exhausted balance returns
**402** at the gate.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Subsystem missing from a deploy | Not in `cfg.Enable` | Enable it explicitly; embeds are disabled-by-default |
| Subsystem refuses to mount | Required config absent | That is fail-closed working as designed — supply the config |
| Console SPA 404 | Binary built without embed | Rebuild via CI; the Dockerfile fails hard when the SPA is missing |
| Postgres in local dev | Wrong default | `cloud serve` uses per-tenant SQLite; do not add Postgres |
| gRPC/proto imports in review | Stale pattern | Rewrite on ZAP types; pb only at the OTel edge (`zap2pb`) |

## Related Skills

- `hanzo-cloud-architecture/SKILL.md` — the canonical architecture skill
- `hanzo/hanzo-zap.md` — ZAP transport and MCP mapping
- `hanzo/hanzo-gateway.md` — HIP-0004 AI gateway at api.hanzo.ai
- `hanzo/hanzo-console.md` — dashboard, metering, LLM observability
- `hanzo/hanzo-iam.md` — identity (embedded `iamsvc`)

---

**Category**: Hanzo Ecosystem
**Related**: cloud, one-binary, zip, zap, plugins, subsystems, white-label, console
**Prerequisites**: Go, the hanzo-cloud-architecture skill
