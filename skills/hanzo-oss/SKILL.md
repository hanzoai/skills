---
name: hanzo-oss
description: Complete progressive directory of all Hanzo Open-Source repositories (hanzoai/* and zap-proto/*) — core Go/Rust/C++ runtimes, AI inference, IAM, KMS, Base, UI design system, MCP, and business suite.
---

# Hanzo Open-Source Ecosystem (`hanzoai/*` & `zap-proto/*`)

> This is a progressive repository directory, not proof of each dependency's open-source licence or production readiness. Some model weights and internal services have separate proprietary rights.
> Query the machine-readable catalog at `https://hanzoskills.com/oss/index.json` and use the `hanzo-oss-legos` skill to choose, verify, and compose components for a customer solution.

## 1. Core Cloud, Data & Transport (Go / Rust)

| Repository | Local Path | Language | Purpose & Entrypoint |
|---|---|---|---|
| [`hanzoai/cloud`](https://github.com/hanzoai/cloud) | `~/work/hanzo/cloud` | Go | Unified cloud host binary serving `api.hanzo.ai` (`/v1/*`) |
| [`hanzoai/base`](https://github.com/hanzoai/base) | `~/work/hanzo/base` | Go | Per-org SQLite backend with S3 replication, IAM auth, and live SSE |
| [`zap-proto/zap`](https://github.com/zap-proto/zap) | `~/work/hanzo/zap` | Rust / Go | Zero-copy binary RPC & local unix-socket router (`zapd`, HIP-0114) |
| [`zap-proto/zip`](https://github.com/zap-proto/zip) | `~/work/hanzo/zip` | Go | Typed service framework emitting HTTP `/v1/*`, OpenAPI, and MCP |
| [`hanzoai/iam`](https://github.com/hanzoai/iam) | `~/work/hanzo/iam` | Go | Sovereign OIDC/OAuth2 identity provider (`hanzo.id`, HIP-0026) |
| [`hanzoai/kms`](https://github.com/hanzoai/kms) | `~/work/hanzo/kms` | Go | Secret & key management (`kms.hanzo.ai`, HIP-0027) |
| [`hanzoai/universe`](https://github.com/hanzoai/universe) | `~/work/hanzo/universe` | YAML / Helm | Declarative Kubernetes GitOps state reconciled by `cd.hanzo.ai` |

## 2. AI Inference, Models & Memory (Rust / C++ / Python / TS)

| Repository | Local Path | Language | Purpose & Entrypoint |
|---|---|---|---|
| [`hanzoai/engine`](https://github.com/hanzoai/engine) | `~/work/hanzo/engine` | Rust / C++ | High-performance CUDA & Metal LLM inference (`hanzo-engine serve`) |
| [`hanzoai/llm`](https://github.com/hanzoai/llm) | `~/work/hanzo/llm` | Go / Python | Unified LLM gateway (`api.hanzo.ai/v1/chat/completions`, HIP-0004) |
| [`hanzoai/zen`](https://github.com/hanzoai/zen) | `~/work/hanzo/zen` | Rust / Python | Zen model family weights, training, and evaluation (`hip-0039`) |
| [`hanzoai/bot`](https://github.com/hanzoai/bot) | `~/work/hanzo/bot` | TS / Docker | Durable agent brain (`~/.hanzo/brain/brain.db`) + sandbox images (`exec`, `dev`, `desktop`, `browser`, `avd`, `admin`) |

## 3. Agent Tooling, CLI & Browser Automation

| Repository | Local Path | Language | Purpose & Entrypoint |
|---|---|---|---|
| [`hanzoai/cli`](https://github.com/hanzoai/cli) | `~/work/hanzo/cli` | Go | Sovereign `hanzo` CLI covering 188 capability groups |
| [`hanzoai/dev`](https://github.com/hanzoai/dev) | `~/work/hanzo/dev` | Rust | Native `dev` coding agent CLI |
| [`hanzoai/mcp`](https://github.com/hanzoai/mcp) | `~/work/hanzo/mcp` | TS / Rust | `@hanzo/mcp` server implementing the 13 HIP-0300 tools |
| [`hanzoai/extension`](https://github.com/hanzoai/extension) | `~/work/hanzo/extension` | TS | Hanzo AI Browser Extension (`dist/chrome`) + `hanzo-browser` sandbox & ZAP social bridge |
| [`hanzoai/skills`](https://github.com/hanzoai/skills) | `~/work/hanzo/skills` | Markdown / Python | `hanzoskills.com` progressive skills, MCP schemas, and HIP standards |

## 4. UI System & Business Applications

| Repository | Local Path | Language | Purpose & Entrypoint |
|---|---|---|---|
| [`hanzoai/ui`](https://github.com/hanzoai/ui) | `~/work/hanzo/ui` | TS / React | `@hanzo/ui`, `@hanzo/gui`, `@hanzo/design`, `@hanzo/appearance` |
| [`hanzoai/team`](https://github.com/hanzoai/team) | `~/work/hanzo/team` | TS / React | `hanzo.team` unified workspace shell & `<BusinessHeader>` |
| [`hanzoai/console`](https://github.com/hanzoai/console) | `~/work/hanzo/console` | TS / React | `crm.hanzo.ai` & `console.hanzo.ai` (Audience, CRM, Twenty-style MIT UI) |
| [`hanzoai/cms`](https://github.com/hanzoai/cms) | `~/work/hanzo/cms` | Go / TS | `cms.hanzo.ai` content & blog management with multi-channel syndication |
| [`hanzoai/post`](https://github.com/hanzoai/post) | `~/work/hanzo/post` | TS / NestJS / Next | `post.hanzo.ai` (`postiz.hanzo.ai`) private social & blog scheduling + AI copilot |

## 5. Composing Solution Stacks

The catalogue is a starting point, not a list of mandatory dependencies. Read `hanzo-solution-composer` for requirements/evidence and design output, `hanzo-oss-legos` for a capability-to-repository matrix and licence checks, `hanzo-decision-evaluation` for model benchmarking, and `hanzo-eu-deployment` for customer-owned private hosting. The industry-neutral reference pattern is `hanzo-enterprise-blueprint`.

Additional public repository candidates in the machine-readable index include `hanzoai/kai`, `hanzoai/ml`, `hanzoai/s3`, `hanzoai/vfs`, `hanzoai/visor`, `hanzoai/insights`, `hanzoai/ads`, `hanzoai/social`, `hanzoai/zen`, `hanzoai/ai`, `hanzoai/vector`, `hanzoai/commerce`, `hanzoai/o11y`, `hanzoai/telemetry`, and `hanzoai/flow`. Visibility alone is insufficient: check exact repo revisions, LICENSE/NOTICE and model-weight terms before calling anything OSS or ready to deploy.
