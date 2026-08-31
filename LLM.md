# LLM.md - Hanzo Skills

## Overview

A skill is a directory: {name}/SKILL.md says what it covers and names the files beside it. 59 skills, 780 markdown files covering the entire Hanzo AI ecosystem. Forty-five are ours; fourteen are Anthropic's, redistributed under Apache-2.0 — see NOTICE and the section below.

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

## The fourteen skills that are not ours

`academy-guide`, `algorithmic-art`, `brand-guidelines`, `canvas-design`,
`claude-api`, `discernment-nudge`, `frontend-design`, `internal-comms`,
`mcp-builder`, `skill-creator`, `slack-gif-creator`, `theme-factory`,
`webapp-testing` and `web-artifacts-builder` come from
github.com/anthropics/skills under Apache-2.0, merged at upstream
`3b3fad96af16a10759d930941b4520ba0c40edae`. They sit flat in `skills/` like
every other skill, each with the upstream `LICENSE.txt` beside its
`SKILL.md`. **That LICENSE.txt is load-bearing — the repository's MIT
licence does not cover those directories.** Never delete one to tidy up, and
carry it through any move. `NOTICE` at the root is the attribution and lists
the two frontmatter changes we made; `THIRD_PARTY_NOTICES.md` is Anthropic's,
verbatim.

Re-syncing means copying the directory again and re-applying exactly two
things: the `license:` frontmatter line, which upstream leaves unnamed, and
`claude-api`'s description, which upstream ships at 1068 characters against a
1024 cap that makes it unloadable as-is.

**Five upstream skills are absent on purpose.** `docx`, `pdf`, `pptx` and
`xlsx` are source-available, not open source: their LICENSE.txt forbids
copying them, keeping them outside Anthropic's services, deriving from them
and distributing them, so merging them here would violate it four ways.
`doc-coauthoring` ships with no LICENSE.txt and the upstream repository has
no root licence, so nothing grants us the right — that one is a question for
Anthropic rather than a settled no. Do not add any of the five because they
look like a gap.

## The site — hanzoskills.com

`public/` is the site, byte for byte. It serves off the **Hanzo Sites plane** —
an S3 prefix plus a host route, no pod, no image, no replicas. Cloudflare is DNS
and CDN for the zone and nothing else: there is no Pages project, no Worker, and
no Cloudflare credential anywhere in the lane.

```
public/          index.html + 537 <name>.html + <name>.md + all.md, llms.txt, favicon.svg
scripts/site.py  the shared chrome: style, nav, hero, copy-prompt menu, agent bar
scripts/marks.py mark geometry, from hanzoai/design assets/{providers,logos}
tests/site.mjs   browser checks — mark, menu, bar, computed border-radius
```

- `scripts/site.py` is the ONE definition of the chrome and rewrites every page.
  It is idempotent, and CI fails if `public/` does not match it.
- Publishing is `.hanzo/workflows/deploy.yml` on push to main — `.hanzo`, not
  `.github`: the forge reads the first workflow dir that exists and this repo is
  private, so GitHub-hosted jobs never start anyway. It calls
  `hanzoai/ci/.github/actions/site@v1`, which is `bin/site`: enqueue → upload the
  objects straight to S3 → complete. The bytes never pass through the API. Do not
  publish by hand, and do not transcribe those three calls here.
- ONE credential, `KMS_CLIENT_ID`/`KMS_CLIENT_SECRET` (org-level on git.hanzo.ai),
  which the action trades at `kms.hanzo.ai` for the deploy key.
- Controls are pills (`--r: 999px`) and surfaces are `--r-soft: 14px`, matching
  hanzo.ai, whose buttons compute to a full pill at 44px tall.
- `public/` content is a snapshot of the corpus, not generated from `skills/`.
  The markdown→html generator that produced it has no home in this repo.

### Where the site lives, outside this repo

Two files in `hanzo/universe`, both on paths that self-heal:

```
charts/app/values/hanzo/static-sites.yaml   hanzoskills-headers + hanzoskills-static
                                            (root s3://hanzo-sites/hanzo/hanzoskills)
                                            and the apex ‖ www route
infra/k8s/ingress/wildcard-certs.yaml       *.hanzoskills.com + hanzoskills.com
```

Response headers are declared THERE, not here: staticFiles reads objects out of
S3 and never a `_headers` file, so a header this site wants is a middleware or it
does not exist. DNS is apex + www `A -> 129.212.164.5`, proxied.

The robots.txt in `public/` is the one that is served, and that is the point of
the move — the zone used to answer with a Cloudflare-managed file that named no
crawler at all, on a site whose whole purpose is being read by agents.

### The plane sends no Content-Type for `.md`

Measured on the live host: `.html`, `.txt`, `.xml` and `.svg` are correct, and
the 539 markdown twins arrive `200` with the right bytes and **no `Content-Type`
header at all**. With `nosniff` set, a browser will not render those inline.

It is an ingress defect, not a site one, and not the obvious cause. The
middleware sets a type only when `mime.TypeByExtension` knows the extension and
otherwise leaves `http.ServeContent` to sniff — and that path, reproduced
faithfully, returns `text/plain`. So the mime table alone does not explain it:
something later in the response chain drops the sniffed value. The stored S3 type
is not the answer either — `objectInfo` carries no content type, so the serving
path never reads what `bin/site` stored.

The robust fix is one line, `mime.AddExtensionType(".md", "text/markdown; charset=utf-8")`,
which makes the middleware `Set` the header explicitly and never reach the sniff.
It needs an `hanzoai/ingress` release and a roll of the shared fleet ingress,
so it is not something to slip in beside a site change.

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
