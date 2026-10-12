# LLM.md - Hanzo Skills

## Overview

A skill is a directory: {name}/SKILL.md says what it covers and names the files beside it. 224 skills, 1,103 markdown files covering the entire Hanzo AI ecosystem. 210 are ours; fourteen are Anthropic's, redistributed under Apache-2.0 — see NOTICE and the section below.

## Install

One skill, into any agent, with the skills.sh CLI. The first form fetches the page the site publishes and
needs no GitHub access; the second clones this repository and copies `skills/<name>/`:

```
npx skills add https://hanzoskills.com/kai.md                 # asks for agents and scope
npx skills add hanzoai/skills --skill kai -a claude-code -y    # Claude Code, this project, no prompts
```

`-g` installs for the user instead of the project; `--list` shows every skill the clone can install.
Verified with skills 1.7.0: the URL form anonymously against the live `cloud-host.md` and against `kai.md`
served from a local build; the clone against github main, 224 skills found, `kai` written to
`.claude/skills/kai/SKILL.md`, byte for byte the file in `skills/kai/`.

Claude Code also takes `kai` as a plugin of its own. `.claude-plugin/marketplace.json` lists it with
`"source": "./skills/kai"`, `"strict": false` and `"skills": ["./"]`, so the plugin is that one directory
and there is no second copy of SKILL.md:

```
claude plugin marketplace add hanzoai/skills
claude plugin install kai@hanzoai-skills
```

Verified on Claude Code 2.1.284: `claude plugin details kai@hanzoai-skills` lists one skill, `kai`, and the
plugin cache holds only its SKILL.md. The `hanzo-skills` plugin (`"source": "./"`, the whole repository)
does not install there: `.claude-plugin/plugin.json` gives `author` as a string where Claude Code requires
an object, and `claude plugin validate .` fails on it.

github.com/hanzoai/skills is private, so the clone and the plugin need GitHub access to it. Without that (no
git credentials, `gh` login or SSH key) the clone fails with `Authentication failed`. The URL form takes any
page the site publishes, `https://hanzoskills.com/<name>.md`; `kai.md` is served once the site deploys.

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
    kai/                # building with Kai decisions; installs on its own (see Install)
      SKILL.md
    hanzo-api/          # product skills — how to CALL api.hanzo.ai; INDEX.md lists 121, ai, kms and sandbox are on disk
      INDEX.md          # every product, grouped into 13 domains
      <product>.md      # one per x-app: families, operations, real curls
                        # GENERATED: scripts/generate_hanzo_api_skills.py
                        # (source: https://api.hanzo.ai/v1/openapi.json)
    hanzo/              # 134 Hanzo ecosystem skills (flat markdown)
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
      ... (134 total)
    # 220 other skill directories (api, cloud, ml, etc.)
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
public/          index.html + 1,244 <name>.html + <name>.md + all.md, llms.txt, favicon.svg, fonts/
scripts/build.py markdown->html generator: parses skills + core HIPs, formats index & categories
scripts/site.py  the shared chrome: style, nav, hero, copy-prompt menu, agent bar, Zen fonts
scripts/marks.py mark geometry, from hanzoai/design assets/{providers,logos}
tests/site.mjs   browser checks — mark, menu, bar, computed border-radius, Zen fonts
```

- `scripts/build.py` compiles `skills/` (including 134 `hanzo/*` skills) and all 275 HIPs into `public/`:
  1,182 pages. It deletes nothing, so the other 62 pages in `public/` are left from earlier builds.
  The HIPs come from `HIPS_DIR`, whose default is a macOS path; anywhere else point it at a
  `hanzoai/hips` checkout at origin/main, or the index, llms.txt and all.md drop every HIP.
  A name several files share (`tool-use`, `streaming`, `batches`, `files-api`, `reference`,
  `experiments`, `affiliate-programs`) is published once, from whichever file `os.walk` reaches
  first, and that order differs between filesystems: a Linux build and a macOS build of the same
  tree pick different files for those pages.
  Standards / HIPs are organized across 5 architectural domains explaining all standards:
  `standards-core` (Core AI & Models), `standards-infra` (Cloud & Native Infrastructure),
  `standards-interface` (Protocols & Realtime SSE), `standards-security` (IAM & Cryptography),
  and `standards-apps` (Platform & Applications).
  The `hanzo` category (led by `hanzo-base`, `hanzo-cloud`, `hanzo-migrate-backend-to-go`, `hanzo-coursework-fullstack-go-cloud`, `hanzo-tutorial-realtime-sse-backend`, `hanzo-examples-native-cloud`, `hanzo-stack`) is sorted first.
- The site enforces native compiled code only (Go, Rust, C++), migrating away from Python/Next.js/Node/Rails/Java.
  Developers have a copy of the entire platform: run a local cloud (`hanzo up cloud`), extend with `/v1/*`,
  build on Hanzo Base (`github.com/hanzoai/base`) with realtime native Server-Sent Events (SSE), analytics,
  and insights, and scale to the full Hanzo AI Cloud (`api.hanzo.ai`, Zen qwen3+ only). Hanzo AI is Techstars '17.
- Typography: Canonical Hanzo `Zen` and `Zen Mono` variable font families (`public/fonts/Zen-Variable.woff2`, `Zen-Italic-Variable.woff2`, `ZenMono-Variable.woff2`, `ZenMono-Italic-Variable.woff2`, `ZenPixel-Square.woff2`) loaded via `@font-face` with `font-display: swap`.
- `scripts/site.py` is the ONE definition of the chrome and rewrites every page.
  It is idempotent, and CI fails if `public/` does not match it.
- The home page opens on two boxes, from `INSTALL`, `PASTE`, `NEXT` and `AFTER` in
  `scripts/site.py`: the install command for the visitor's OS (tabs; every OS
  without JS) and the line to paste into an agent. `site.py` writes the same
  block into `public/index.md`. Windows is one `INSTALL` row once
  `hanzo.sh/install.ps1` serves a script (404 on 2026-10-12).
- Publishing is `.github/workflows/deploy.yml` on push to main, beside `ci.yml`.
  That is the only workflow directory here: the forge reads the first one that
  exists, and the org's own `linux-amd64` runners serve both planes. It calls
  `hanzoai/ci/.github/actions/site@v1`, which is `bin/site`: enqueue → upload the
  objects straight to S3 → complete. The bytes never pass through the API. Do not
  publish by hand, and do not transcribe those three calls here.
- GitHub Actions is off for hanzoai/skills: `gh api repos/hanzoai/skills/actions/permissions`
  answers `{"enabled": false}`, and the last runs, CI and deploy for ef27aaf on 2026-09-09, were
  cancelled. Until a repo admin turns it back on, a push to main runs neither workflow and
  publishes nothing.
- ONE credential, `KMS_CLIENT_ID`/`KMS_CLIENT_SECRET` (org-level on git.hanzo.ai),
  which the action trades at `kms.hanzo.ai` for the deploy key.
- Controls are pills (`--r: 999px`) and surfaces are `--r-soft: 14px`, matching
  hanzo.ai, whose buttons compute to a full pill at 44px tall.

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

### Content types

The 1,246 markdown files (the 1,244 page twins, `index.md` and `all.md`) are served
`text/markdown; charset=utf-8`; `.html`, `.txt`, `.xml` and `.svg` carry their own types.
Measured on the live host for `index.md`, `hanzo-base.md`, `all.md` and `hip-0004.md`.

`scripts/build.py` renders headings, lists, quotes, code and paragraphs. It has no table
rule: a markdown table on a page arrives as one paragraph of pipes in the HTML, and intact
in the `.md` twin.

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
