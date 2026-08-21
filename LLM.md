# LLM.md - Hanzo Skills

## Overview

A skill is a directory: {name}/SKILL.md says what it covers and names the files beside it. 45 skills, 678 files covering the entire Hanzo AI ecosystem.

## The Architecture (current truth)

Hanzo runs as ONE Go binary: `hanzoai/cloud` embeds IAM, KMS, o11y, tasks,
notify, pubsub, kafka, console, and ai as `clients/*` plugins on **zip** (the
ZAP-native server) behind the `cloud.Register` registry. Transport is **ZAP
only** (HIP-0114) — zero gRPC, zero protobuf in Hanzo code (pb exists solely
at the OTel edge via `zap2pb`). Storage is SQLite-primary per-tenant plus
datastore (ClickHouse); coordination is Lux Quasar + zapdb — never
ZooKeeper/raft/etcd. Deploy modes (HIP-0117): `cloud serve` (no k8s),
`cloud cluster init` (k3s + operator CRs), `helm install` (BYO k8s).
Canonical skill: `skills/hanzo-cloud-architecture/SKILL.md`.
Full spec: HIPs 0106, 0114, 0116, 0117, 0004.

## Structure

```
skills/
  skills/
    hanzo-cloud-architecture/  # THE architecture skill (read first)
      SKILL.md
    hanzo-api/          # 121 product skills — how to CALL api.hanzo.ai
      INDEX.md          # every product, grouped into 13 domains
      <product>.md      # one per x-app: families, operations, real curls
                        # GENERATED: scripts/generate_hanzo_api_skills.py
                        # (source: https://api.hanzo.ai/v1/openapi.json)
    hanzo/              # 131 Hanzo ecosystem skills (flat markdown)
      INDEX.md          # Catalog and decision tree
      hanzo-k8s.md      # K8s infrastructure
      hanzo-deploy.md   # Deployment guide
      hanzo-platform.md # PaaS (Dokploy fork)
      hanzo-id.md       # Client auth flows — @hanzo/iam SDK (PKCE, one way)
      hanzo-iam.md      # IAM server — hanzoai/iam (clean rewrite; retired fork = iam-v1)
      hanzo-identity.md # On-chain DID (distinct product, not the IAM login)
      hanzo-kms.md      # Secrets (luxfi/kms MPC)
      hanzo-ingress.md  # Ingress (Traefik fork)
      hanzo-cloud.md    # THE unified one-binary cloud (HIP-0106)
      hanzo-console.md  # LLM observability (Langfuse)
      hanzo-chat.md     # Chat UI (LibreChat fork)
      hanzo-team.md     # Collaboration (Huly fork)
      hanzo-bot.md      # Multi-channel bot skill
      hanzo-mcp.md      # Model Context Protocol
      hanzo-zap.md      # ZAP — THE transport (HIP-0114) + DB bridge
      hanzo-o11y.md     # Observability (embedded o11y subsystem)
      ... (131 total)
    # 80+ other skill directories (api, cloud, ml, etc.)
  commands/             # Custom commands
  scripts/              # Helper scripts
  tests/                # Test suite
```

## The site — hanzoskills.com

`public/` is the site, byte for byte. It is a **Cloudflare Pages** project
(direct upload), not a Worker and not ingress+s3 — the live host returns
`access-control-allow-origin: *` and a charset-suffixed content-type, which
Workers Static Assets does not emit. Compare `/llms.txt` against hanzo.codes
(a Worker) to tell them apart; the `/` response alone does not distinguish them.

```
public/          index.html + 537 <name>.html + <name>.md + all.md, llms.txt, favicon.svg
scripts/site.py  the shared chrome: style, nav, hero, copy-prompt menu, agent bar
scripts/marks.py mark geometry, from hanzoai/design assets/{providers,logos}
tests/site.mjs   browser checks — mark, menu, bar, computed border-radius
wrangler.toml    pages_build_output_dir = public
```

- `scripts/site.py` is the ONE definition of the chrome and rewrites every page.
  It is idempotent, and CI fails if `public/` does not match it.
- Publishing is `.github/workflows/deploy.yml` on push to main. It resolves the
  Pages project **by hostname** rather than by a remembered name, then asserts
  the live host serves the new bytes. Do not publish by hand.
- Cloudflare injects its analytics beacon at the edge; `site.py` strips it so it
  never enters source.
- Controls are pills (`--r: 999px`) and surfaces are `--r-soft: 14px`, matching
  hanzo.ai, whose buttons compute to a full pill at 44px tall.
- `public/` content is a snapshot of the corpus, not generated from `skills/`.
  The markdown→html generator that produced it has no home in this repo.

### Publishing is blocked on the KMS machine identity

The deploy runs green through the browser checks and stops at the credential
read. KMS answers `/v1/kms/auth/login` with **502** for this repo's
`KMS_CLIENT_ID`/`KMS_CLIENT_SECRET`, from both `cloud.hanzo.svc:8000` and
`api.hanzo.ai`. A wrong client id/secret returns **401**, so 502 means the
identity is found and then fails to resolve server-side — a KMS provisioning
fault, not a network or workflow one. Cloud logs show the split cleanly:
internal callers 502, external callers with bad creds 401.

To unblock, someone with KMS admin needs to reissue the machine identity for
this repo and confirm `CF_API_TOKEN` and `CF_ACCOUNT_ID` are sealed at
`hanzo/deploy/cf-pages` env=prod. Nothing in this repo needs to change.

GitHub Actions is not an alternative here: `hanzoai/skills` is private and the
org's Actions billing is failing, so private-repo jobs never start. Public
repos (hanzo.codes) still run.

## Key Skills by Domain

### Infrastructure
- `hanzo-k8s.md` -- Two DOKS clusters: hanzo-k8s (prod, 22 nodes) and lux-k8s (blockchain)
- `hanzo-deploy.md` -- Universe repo is source of truth. main=prod, dev=staging
- `hanzo-ingress.md` -- Traefik fork, IngressClass `hanzo`, auto TLS
- `hanzo-static.md` -- Traefik static plugin, replaces nginx/caddy

### Identity & Security
- `hanzo-id.md` -- RFC 6749/OIDC, multi-org (hanzo/lux/zoo/pars), Cloudflare Worker
- `hanzo-kms.md` -- KMSSecret CRDs for K8s sync, 9 Vault subsystems
- `hanzo-zt.md` -- Zero-trust (OpenZiti fork), dark services, mTLS

### AI Services
- `hanzo-cloud.md` -- THE unified one-binary cloud (HIP-0106); `ai` subsystem = provider mgmt
- `hanzo-skill.md` -- HIP-0004 unified AI provider interface at api.hanzo.ai, BYO keys
- `hanzo-console.md` -- LLM traces, costs, prompts + per-product metering, 402 balance-gating
- `hanzo-chat.md` -- Multi-model chat UI (14 Zen + 100+ third-party)
- `hanzo-mcp.md` -- 260+ tools, 13 HIP-0300 unified tools
- `hanzo-bot.md` -- 50+ messaging adapters, ACP skill

### Data
- `hanzo-zap.md` -- ZAP: THE transport (HIP-0114) + SQL/KV/Datastore/DocumentDB→MCP bridge
- `hanzo-o11y.md` -- Infrastructure APM, embedded in cloud; ClickHouse keeper-free, OTLP-over-ZAP

## Hard Rules

1. ONE binary, ZAP only -- no gRPC, no protobuf in Hanzo code (pb only at the OTel edge via zap2pb)
2. KMS for all secrets -- never plaintext in manifests
3. GHCR for images -- `ghcr.io/hanzoai/*`, `--platform linux/amd64`; never mix with `ghcr.io/luxfi/*`
4. Kustomize, not Helm (exception: `cloud/helm/cloud` for BYO-k8s installs)
5. No nginx, no caddy -- Hanzo Ingress only
6. CI = Gitea Actions on git.hanzo.ai (own runners); GitHub = OSS mirror; no GH-hosted builders
7. IAM for all auth -- RFC 6749 endpoints, implicit flow preferred
8. Universe repo is source of truth for K8s manifests; operator reconciles `services.hanzo.ai` CRs
9. SQLite-primary per-tenant; PostgreSQL only for prod multi-instance; never default to PG locally
10. Quasar + zapdb for coordination -- no ZooKeeper, no raft, no etcd
11. Zen models: qwen3+ only, never reference upstream names
12. luxfi packages only, never go-ethereum or ava-labs
13. compose.yml not docker-compose.yml

## Production Topology

8 Kubernetes clusters operated; visor autoscales fleets across providers.
Post-consolidation, the standalone notify/tasks/o11y/pubsub/... pods are
retired — they run as subsystems inside the `cloud` binary. The surviving
distinct surfaces:

```
hanzo-k8s (do-sfo3-hanzo-k8s, LB: 165.232.146.176)
  +-- cloud (ONE binary: IAM/KMS/o11y/tasks/notify/pubsub/kafka/console/ai)
  |     hanzo.id, kms.hanzo.ai, console.hanzo.ai, o11y.hanzo.ai, ...
  +-- Gateway (api.hanzo.ai -- HIP-0004 unified AI interface)
  +-- Chat (chat.hanzo.ai), Team (hanzo.team), Bot, DNS, ZT, ...
  +-- SQLite per-tenant (in-binary) + PostgreSQL for multi-instance
  +-- datastore/ClickHouse (telemetry, keeper-free)
  +-- Valkey (redis.hanzo.svc), MinIO (minio.hanzo.svc)

lux-k8s (do-sfo3-lux-k8s, LB: 24.144.69.101)
  +-- validators
  +-- Gateway (api.lux.network)
  +-- Markets, Cloud (white-label of the same binary)
```
