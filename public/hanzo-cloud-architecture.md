---
name: hanzo-cloud-architecture
description: The canonical Hanzo Cloud architecture — one repo, host + per-app plugin binaries on zip, ZAP-only transport, router-projected OpenAPI, the receipt release train. Read this before touching any hanzoai service.
---

# Hanzo Cloud Architecture — Host + Plugins, ZAP-Native, Router-Projected

**Category**: Hanzo Ecosystem
**Canonical spec**: HIP-0106 (one binary), HIP-0114 (ZAP transport), HIP-0116 (plugin/VM model), HIP-0117 (cloud-in-a-box), HIP-0004 (AI gateway) — `github.com/hanzoai/hips`
**Live truth**: `~/work/hanzo/cloud/LLM.md` — 5,800 lines, actively corrected in place. It OUTRANKS this skill wherever they disagree; this skill is the orientation, that file is the record.

## When to Use This Skill

- Working anywhere in the cloud repo or any app it composes (iam, kms, ai, commerce, o11y, git, platform, …)
- Adding a subsystem, choosing a transport or storage backend, or shipping a release
- Reading any older doc that mentions `clients/*`, a 76:1 plugin ratio, gRPC, or multi-pod services — all superseded

## The Architecture in One Paragraph

`cloud` is ONE Go module that builds a LIGHT HOST (`cmd/cloud`, links zip + the
manifest and none of the apps) plus ONE BINARY PER APP (`plugin/<app>/main.go`,
~144 apps under `apps/<name>/`). `manifest/apps.go` is the hand-authored routing
table — every app, its `/v1/<segment>` prefixes, in routing order; ai is the
`/v1` REMAINDER, last, answering what no earlier app claimed. Transport is ZAP
and nothing else (HIP-0114): zero gRPC, zero protobuf in Hanzo code (`zap2pb` at
the OTel edge is the one exception). The same artifact serves api.hanzo.ai and
every white-label surface; brand and enabled apps are deployment configuration.

## Repo topology (as of 2026-08-18 — verify against LLM.md "Provenance")

- **Product line + version register**: `git.hanzo.ai/hanzo-inc/cloud` (Gitea, the
  release train runs beside it; its v1.801.* tags are the receipts).
- **Working mirror**: `github.com/hanzo-inc/cloud` — the checkout at
  `~/work/hanzo/cloud` tracks this as `origin`. Ship = green gate locally, push
  main to BOTH.
- **Public OSS core**: `github.com/hanzoai/cloud` — a re-rooted, dual-licensed
  (Apache-2.0 OR MIT) line sharing NO ancestor with the product.
  `git.hanzo.ai/hanzoai/cloud` is its read-only pull mirror. The receipt car
  publishes each release's record there as a GitHub Release; the tag GitHub
  mints for it sits on the OSS-core tip, a different line. **NEVER
  `git fetch --tags` from the hanzoai/cloud remotes** — those tag names poison
  `git describe`, internal/lineage and pin ordering.

## Build & test — the traps

- **NEVER `go build ./...`** — it links 100+ plugin binaries at ~4.5 GiB each.
- MODULE mode only: `GOWORK=off`, never workspace mode.
- `make build` — the light host. `make plugin APP=<x>` — the one app you edit.
  `make ship` — host + every app. `make test` — the release gate (closure-check,
  zipdoc-check, `go test ./...`, then the fleet drift gate, ~35 min).
  `make test-fast` — everything but the drift gate; inner loop only.
- Dependency bumps: `make bump M=<mod>@<ver>` — moving a dep and regenerating the
  documents it moves are ONE action; `make closure-check` (seconds) names stale
  documents and prints the exact fix.

## The OpenAPI document pipeline (openapi/, describe.go, mk/fleet.mk)

The spec is a PROJECTION OF THE LIVE ROUTER; `openapi.yaml` at the root is a
golden of it. Four facts are declared beside the routes: bodies (`Register`),
prose (`Describe`), audience (`Public` → x-public, default-deny), credential
(`security.go`, default-REQUIRE, `Open()` per-op). Every operation must state
what it does for the caller — a route with no prose FAILS the gate (the sentence
lives on the handler's Go doc comment, or `openapi.Describe` beside the mount
for routes an embedded module registers). Each app's committed subset
(`plugin/<app>/openapi.json`) is produced by `<binary> describe <dir>` from that
app's own mount, and carries ONLY addresses the fleet delivers to that app —
describe prunes any path whose `manifest.OwnerOf` (longest-prefix, the routing
table's own answer) names a sibling. The fleet document is the WEAVE of subsets;
one address claimed by two apps is a red gate, never a picked winner.

## The release train (.hanzo/workflows/cicd.yml + hanzo.yml)

One workflow, one `needs:` graph: gate → image → live → reach → fanout →
receipt, on Hanzo runners against git.hanzo.ai. NO GitHub-hosted builders, no
local image builds. A v* tag is a RECEIPT minted only after build + smoke pass —
never a build trigger. The rollout car moves the universe pin
(`universe/charts/app/values/hanzo/cloud.yaml`); cd reconciles it. It does not
roll back — it blocks and resumes (workflow_dispatch at the same sha; every car
is idempotent on (version, digest)). Ask production what it runs:
`curl -s https://api.hanzo.ai/v1/health` → `{"revision": …}`.

## Core laws (unchanged)

- ZAP-only transport; zip (`github.com/hanzoai/zip`) is the one server framework.
- SQLite-primary per tenant (hanzoai/sqlite); datastore (ClickHouse) for
  telemetry columns only. No postgres for local dev, no ZooKeeper/etcd/raft —
  Quasar + zapdb is the substrate.
- Apps fail closed: a subsystem without its required config refuses to mount.
- Calling an app that is not deployed returns an ERROR, never a zero value
  (absent_test.go — the Register* seam list is guarded).
- IAM is all auth (`hanzoai/iam` embedded); KMS for secrets; `/v1/` only.
- Console SPA is a prebuilt input the Dockerfile pulls; fail-hard if missing.

### Key Guidelines

```
✅ DO: read ~/work/hanzo/cloud/LLM.md before structural work — it is the record
✅ DO: add an app under apps/<name>, declare it in manifest/apps.go, make generate
✅ DO: write every route's sentence where the handler lives; regenerate documents
✅ DO: run `make test` before pushing — a red push blocks the train for everyone

❌ DON'T: go build ./..., go work, build images locally, or add a second CI lane
❌ DON'T: introduce gRPC/protobuf, a second RPC stack, or a second auth layer
❌ DON'T: fetch tags from the public OSS repo, or read its tags as receipts
❌ DON'T: publish an address the manifest routes to a sibling app
```

## Related Skills

- `skills/hanzo/hanzo-cloud.md` — the cloud binary operationally (config, URLs, deploy)
- `skills/hanzo/hanzo-zap.md` — ZAP protocol details and the MCP mapping
- `skills/hanzo/hanzo-gateway.md` — the HIP-0004 AI gateway at api.hanzo.ai
- luxfi/skills `lux-zap-substrate` — the Lux substrate (Quasar + zapdb + zip + lpm)
