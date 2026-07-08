---
name: discover-hanzo-cloud-architecture
description: Automatically discover the Hanzo Cloud one-binary ZAP-native architecture skill when working on hanzoai services, transports, or deployment topology.
---

# Hanzo Cloud Architecture Discovery

Gateway to the canonical "how the one-binary ZAP-native plugin cloud works" skill.

## When This Skill Activates

This skill auto-activates when you're working with:
- `hanzoai/cloud` or any embedded subsystem (IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai)
- ZAP transport, zip server, ZAP-RPC, zap2pb / pb2zap / zapc
- Adding a new Hanzo service, subsystem, or plugin
- `cloud serve`, `cloud cluster init`, helm deploys of Hanzo Cloud
- Anything that tempts you toward gRPC, protobuf, ZooKeeper, etcd, or a new pod

## Load the Skill

```bash
cat skills/hanzo-cloud-architecture/SKILL.md
```

## One-Line Truths (current architecture)

- ONE Go binary (`hanzoai/cloud`) embeds every subsystem; 977-LOC core on zip, 74,561 LOC of `clients/*` plugins.
- Transport is ZAP only (HIP-0114) — zero gRPC, zero protobuf in Hanzo code; pb exists only at the OTel edge via `zap2pb`.
- Storage: SQLite-primary per-tenant; datastore (ClickHouse) is the one non-Go exception.
- Consensus/replication: Lux Quasar + zapdb — never ZooKeeper/raft/etcd.
- Services build standalone AND as ZAP-native plugin VMs distributed via `luxfi/lpm` (HIP-0116).
- Deploy modes: `cloud serve` (no k8s) | `cloud cluster init` (fetch k3s + operator CRs) | `helm install` (BYO k8s) — HIP-0117.
- Full spec: HIPs 0106, 0114, 0116, 0117, 0004 in `github.com/hanzoai/hips`.

## Related Categories

- `skills/hanzo/INDEX.md` — per-service Hanzo skills
- `skills/observability/INDEX.md` — o11y (embedded subsystem)
- `skills/distributed-systems/INDEX.md` — consensus theory (production uses Quasar + zapdb)
