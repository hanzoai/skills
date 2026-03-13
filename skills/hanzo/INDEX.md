# Hanzo Ecosystem Skills

**Complete guide to Hanzo's AI infrastructure, services, and developer tools**

## Overview

Hanzo is a comprehensive AI ecosystem — from frontier models (Zen MoDE) to production infrastructure (PaaS, KMS, IAM) to developer tools (SDKs, CLI, extensions). 126 skills cover every component with progressive disclosure.

### Philosophy

- **Local-First**: Run AI on your infrastructure for privacy and control
- **MCP-Native**: Built for Model Context Protocol agentic workflows
- **Higher-Level Abstractions**: Use Hanzo "legos" instead of line-by-line code
- **Production-Ready**: Battle-tested components and patterns

## Skill Catalog

### Core AI Infrastructure (12 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-engine.md` | Rust inference & embedding engine (mistral.rs fork) | `cat skills/hanzo/hanzo-engine.md` |
| `hanzo-vllm.md` | Rust-based LLM inference engine (candle-vllm fork) | `cat skills/hanzo/hanzo-vllm.md` |
| `hanzo-node.md` | Rust AI agent node with P2P/MCP/WASM (v1.1.20, 30+ crates) | `cat skills/hanzo/hanzo-node.md` |
| `hanzo-network.md` | Distributed AI agent topology | `cat skills/hanzo/hanzo-network.md` |
| `hanzo-llm-gateway.md` | Unified LLM proxy (100+ providers, LiteLLM v1.82.1 fork) | `cat skills/hanzo/hanzo-llm-gateway.md` |
| `hanzo-mcp.md` | MCP server — 13 HIP-0300 unified tools (`@hanzo/mcp` v2.4.1) | `cat skills/hanzo/hanzo-mcp.md` |
| `hanzo-agent.md` | Multi-agent SDK (`hanzoai` v0.0.4, OpenAI agents fork) | `cat skills/hanzo/hanzo-agent.md` |
| `hanzo-aci.md` | Agent Computer Interface (tool/action abstraction) | `cat skills/hanzo/hanzo-aci.md` |
| `hanzo-tools.md` | Unified AI agent tool registry (88 tools, Rust) | `cat skills/hanzo/hanzo-tools.md` |
| `hanzo-computer.md` | Desktop automation & computer use | `cat skills/hanzo/hanzo-computer.md` |
| `hanzo-operative.md` | Computer use agent (Anthropic fork, v0.1.1) | `cat skills/hanzo/hanzo-operative.md` |
| `hanzo-memory.md` | Memory API server for AI agents | `cat skills/hanzo/hanzo-memory.md` |

### Cloud Services & Apps (11 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-app.md` | AI-powered developer platform & desktop assistant | `cat skills/hanzo/hanzo-app.md` |
| `hanzo-chat.md` | Hanzo Chat (LibreChat v0.8.3-rc1 fork) at chat.hanzo.ai | `cat skills/hanzo/hanzo-chat.md` |
| `hanzo-cloud.md` | AI provider management (Go/Beego, PostgreSQL, Casibase fork) | `cat skills/hanzo/hanzo-cloud.md` |
| `hanzo-console.md` | AI observability, tracing, cost tracking | `cat skills/hanzo/hanzo-console.md` |
| `hanzo-commerce-api.md` | Billing, payments, subscriptions API | `cat skills/hanzo/hanzo-commerce-api.md` |
| `hanzo-analytics.md` | Analytics & tracking service | `cat skills/hanzo/hanzo-analytics.md` |
| `hanzo-billing.md` | Billing & subscription management | `cat skills/hanzo/hanzo-billing.md` |
| `hanzo-web3.md` | Blockchain API (100+ EVM/non-EVM chains) | `cat skills/hanzo/hanzo-web3.md` |
| `hanzo-web3-gateway.md` | Keyless blockchain via x402 micropayments | `cat skills/hanzo/hanzo-web3-gateway.md` |
| `hanzo-website.md` | Corporate website (hanzo.ai) | `cat skills/hanzo/hanzo-website.md` |
| `hanzo-store.md` | AI app store / marketplace | `cat skills/hanzo/hanzo-store.md` |

### Platform & Deployment (16 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-platform.md` | PaaS (Dokploy fork, Hono.js API + Next.js 16) | `cat skills/hanzo/hanzo-platform.md` |
| `hanzo-gateway.md` | API gateway (KrakenD/Lura, 147+ endpoints, Go) | `cat skills/hanzo/hanzo-gateway.md` |
| `hanzo-ingress.md` | Ingress controller (Traefik fork, Go) | `cat skills/hanzo/hanzo-ingress.md` |
| `hanzo-tunnel.md` | WebSocket tunnel & bot agent bridge (Rust + Python) | `cat skills/hanzo/hanzo-tunnel.md` |
| `hanzo-zt.md` | Zero-trust network fabric (OpenZiti fork, Go, mTLS/PKI) | `cat skills/hanzo/hanzo-zt.md` |
| `hanzo-zrok.md` | Zero-trust sharing platform (OpenZiti/zrok fork, Go) | `cat skills/hanzo/hanzo-zrok.md` |
| `hanzo-dns.md` | Programmable DNS (CoreDNS fork, Go) | `cat skills/hanzo/hanzo-dns.md` |
| `hanzo-operator.md` | K8s operator for Hanzo services | `cat skills/hanzo/hanzo-operator.md` |
| `hanzo-charts.md` | Helm charts for all Hanzo services | `cat skills/hanzo/hanzo-charts.md` |
| `hanzo-hke.md` | Managed Kubernetes Engine | `cat skills/hanzo/hanzo-hke.md` |
| `hanzo-vm.md` | Cloud OS & VM management | `cat skills/hanzo/hanzo-vm.md` |
| `hanzo-functions.md` | Serverless functions platform | `cat skills/hanzo/hanzo-functions.md` |
| `hanzo-stack.md` | Full local dev environment | `cat skills/hanzo/hanzo-stack.md` |
| `hanzo-runtime.md` | Runtime environment for AI workloads | `cat skills/hanzo/hanzo-runtime.md` |
| `hanzo-universe.md` | Production K8s infrastructure (private) | `cat skills/hanzo/hanzo-universe.md` |
| `hanzo-terraform.md` | Terraform provider for Hanzo | `cat skills/hanzo/hanzo-terraform.md` |

### SDKs & Libraries (15 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `python-sdk.md` | Python SDK (`hanzoai` v2.2.0, Stainless, 9 packages) | `cat skills/hanzo/python-sdk.md` |
| `js-sdk.md` | TypeScript SDK (`hanzoai` v0.1.0-alpha.2, Stainless) | `cat skills/hanzo/js-sdk.md` |
| `go-sdk.md` | Go SDK (`github.com/hanzoai/go-sdk` v0.1.0-alpha.4) | `cat skills/hanzo/go-sdk.md` |
| `rust-sdk.md` | Rust SDK (v1.1.12, 14 crates: agents, MCP, PQC, guard) | `cat skills/hanzo/rust-sdk.md` |
| `hanzo-sdk.md` | Unified multi-language CLI & client library | `cat skills/hanzo/hanzo-sdk.md` |
| `hanzo-openapi.md` | Unified API spec for 26 cloud services | `cat skills/hanzo/hanzo-openapi.md` |
| `hanzo-trpc-openapi.md` | tRPC-to-OpenAPI adapter | `cat skills/hanzo/hanzo-trpc-openapi.md` |
| `hanzo-base.md` | Shared base library & utilities | `cat skills/hanzo/hanzo-base.md` |
| `hanzo-orm.md` | Go generics ORM (`github.com/hanzoai/orm`) | `cat skills/hanzo/hanzo-orm.md` |
| `hanzo-log.md` | Structured logging library (Go) | `cat skills/hanzo/hanzo-log.md` |
| `hanzo-kv-go.md` | KV store Go client | `cat skills/hanzo/hanzo-kv-go.md` |
| `hanzo-pubsub-go.md` | PubSub Go client | `cat skills/hanzo/hanzo-pubsub-go.md` |
| `hanzo-search-go.md` | Search Go client | `cat skills/hanzo/hanzo-search-go.md` |
| `hanzo-vector-go.md` | Vector store Go client | `cat skills/hanzo/hanzo-vector-go.md` |
| `hanzo-edge.md` | Edge computing & CDN | `cat skills/hanzo/hanzo-edge.md` |

### Identity & Security (7 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-iam.md` | IAM server (Casdoor fork, Go/Beego, OAuth2/OIDC/SAML/LDAP) | `cat skills/hanzo/hanzo-iam.md` |
| `hanzo-id.md` | Custom login UI (Next.js, white-label) | `cat skills/hanzo/hanzo-id.md` |
| `hanzo-identity.md` | Identity service & certificate management | `cat skills/hanzo/hanzo-identity.md` |
| `hanzo-kms.md` | Secret management (Infisical fork + 9 Vault subsystems) | `cat skills/hanzo/hanzo-kms.md` |
| `hanzo-vault.md` | PCI-compliant card tokenization (Go) | `cat skills/hanzo/hanzo-vault.md` |
| `hanzo-guard.md` | Security middleware & threat detection | `cat skills/hanzo/hanzo-guard.md` |
| `hanzo-mpc.md` | Multi-party computation for distributed wallets | `cat skills/hanzo/hanzo-mpc.md` |

### Developer Tools (10 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-extension.md` | Browser & IDE extensions v1.8.0 (pnpm monorepo) | `cat skills/hanzo/hanzo-extension.md` |
| `hanzo-cli.md` | Command-line interface | `cat skills/hanzo/hanzo-cli.md` |
| `hanzo-dev.md` | Terminal AI agent (Codex CLI fork, Bazel, @hanzo/dev) | `cat skills/hanzo/hanzo-dev.md` |
| `hanzo-code.md` | Web-based code editor | `cat skills/hanzo/hanzo-code.md` |
| `hanzo-editor.md` | Web IDE / rich text editor | `cat skills/hanzo/hanzo-editor.md` |
| `hanzo-live.md` | Real-time generative AI pipelines | `cat skills/hanzo/hanzo-live.md` |
| `hanzo-repl.md` | Interactive REPL | `cat skills/hanzo/hanzo-repl.md` |
| `hanzo-relay.md` | Client relay for Hanzo apps | `cat skills/hanzo/hanzo-relay.md` |
| `hanzo-skill.md` | Skill installer CLI | `cat skills/hanzo/hanzo-skill.md` |
| `hanzo-playground.md` | AI playground | `cat skills/hanzo/hanzo-playground.md` |

### AI Models & Training (8 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `zenlm.md` | Zen frontier models (MoDE, 600M-480B params) | `cat skills/hanzo/zenlm.md` |
| `hanzo-gym.md` | Unified training platform (Zoo Labs) | `cat skills/hanzo/hanzo-gym.md` |
| `hanzo-jin.md` | Multimodal LLM (text/vision/audio/3D) | `cat skills/hanzo/hanzo-jin.md` |
| `hanzo-kensho.md` | Image generation model (17B params) | `cat skills/hanzo/hanzo-kensho.md` |
| `hanzo-koe.md` | Text-to-dialogue model | `cat skills/hanzo/hanzo-koe.md` |
| `hanzo-mugen.md` | Audio generation (PyTorch) | `cat skills/hanzo/hanzo-mugen.md` |
| `hanzo-sho.md` | Text diffusion engine | `cat skills/hanzo/hanzo-sho.md` |
| `hanzo-ml.md` | ML utilities & shared libraries | `cat skills/hanzo/hanzo-ml.md` |

### ML Infrastructure (3 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-candle.md` | Rust ML framework (HF candle fork) | `cat skills/hanzo/hanzo-candle.md` |
| `hanzo-ane.md` | Apple Neural Engine training (M1+) | `cat skills/hanzo/hanzo-ane.md` |
| `hanzo-evm.md` | Rust EVM (reth v1.11.0 fork, Rust 2024, MSRV 1.93) | `cat skills/hanzo/hanzo-evm.md` |

### Data & Observability (15 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-database.md` | PostgreSQL, Redis, pgvector | `cat skills/hanzo/hanzo-database.md` |
| `hanzo-sql.md` | PostgreSQL with AI extensions | `cat skills/hanzo/hanzo-sql.md` |
| `hanzo-kv.md` | High-performance key-value store (Valkey fork) | `cat skills/hanzo/hanzo-kv.md` |
| `hanzo-documentdb.md` | MongoDB-compatible database | `cat skills/hanzo/hanzo-documentdb.md` |
| `hanzo-datastore.md` | Vector database integration | `cat skills/hanzo/hanzo-datastore.md` |
| `hanzo-vector.md` | Vector search engine (Qdrant v1.17.0 fork, Rust, HNSW) | `cat skills/hanzo/hanzo-vector.md` |
| `hanzo-s3.md` | S3-compatible object storage | `cat skills/hanzo/hanzo-s3.md` |
| `hanzo-storage.md` | Object storage (MinIO) | `cat skills/hanzo/hanzo-storage.md` |
| `hanzo-pubsub.md` | Pub/sub messaging service (NATS) | `cat skills/hanzo/hanzo-pubsub.md` |
| `hanzo-stream.md` | Event streaming (Kafka gateway) | `cat skills/hanzo/hanzo-stream.md` |
| `hanzo-logs.md` | Scalable log aggregation (Loki fork) | `cat skills/hanzo/hanzo-logs.md` |
| `hanzo-metrics.md` | Time-series database (VictoriaMetrics fork) | `cat skills/hanzo/hanzo-metrics.md` |
| `hanzo-o11y.md` | Observability platform (SigNoz fork, ClickHouse, OTEL) | `cat skills/hanzo/hanzo-o11y.md` |
| `hanzo-sentry.md` | Error tracking & perf monitoring (Sentry 24.2.0 fork) | `cat skills/hanzo/hanzo-sentry.md` |
| `hanzo-insights.md` | Product analytics (PostHog fork, InsightsQL) | `cat skills/hanzo/hanzo-insights.md` |

### Commerce & Finance (14 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-commerce.md` | Multi-tenant e-commerce (Go, v1.36.5, 100+ models) | `cat skills/hanzo/hanzo-commerce.md` |
| `hanzo-checkout.md` | Payment checkout JS library (96 stars) | `cat skills/hanzo/hanzo-checkout.md` |
| `hanzo-form.md` | Form handling library (16 stars) | `cat skills/hanzo/hanzo-form.md` |
| `hanzo-gochimp.md` | MailChimp API Go client (68 stars) | `cat skills/hanzo/hanzo-gochimp.md` |
| `hanzo-authorizenet.md` | AuthorizeNet Go client | `cat skills/hanzo/hanzo-authorizenet.md` |
| `hanzo-ledger.md` | Double-entry financial ledger (Formance fork, Go) | `cat skills/hanzo/hanzo-ledger.md` |
| `hanzo-treasury.md` | Programmable financial infrastructure (Formance Stack) | `cat skills/hanzo/hanzo-treasury.md` |
| `hanzo-numscript.md` | Financial transaction DSL | `cat skills/hanzo/hanzo-numscript.md` |
| `hanzo-reconciliation.md` | Transaction auto-matching | `cat skills/hanzo/hanzo-reconciliation.md` |
| `hanzo-payments.md` | Payment processing & routing | `cat skills/hanzo/hanzo-payments.md` |
| `hanzo-sign.md` | Document/crypto signing service | `cat skills/hanzo/hanzo-sign.md` |
| `hanzo-registry.md` | Container/package registry | `cat skills/hanzo/hanzo-registry.md` |
| `hanzo-explorer.md` | EVM blockchain explorer (Blockscout fork) | `cat skills/hanzo/hanzo-explorer.md` |
| `hanzo-faucet.md` | Hanzo Network token faucet | `cat skills/hanzo/hanzo-faucet.md` |

### Apps & Design (8 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-brand.md` | Brand guidelines, colors, logo | `cat skills/hanzo/hanzo-brand.md` |
| `hanzo-bot.md` | Bot framework (~40 extensions, OpenClaw fork) | `cat skills/hanzo/hanzo-bot.md` |
| `hanzo-ui.md` | React component library (shadcn/ui fork, 161+ components) | `cat skills/hanzo/hanzo-ui.md` |
| `hanzo-desktop.md` | AI desktop app (Shinkai fork, Tauri + React Native) | `cat skills/hanzo/hanzo-desktop.md` |
| `hanzo-docs.md` | Documentation site (fumadocs fork, Next.js/MDX) | `cat skills/hanzo/hanzo-docs.md` |
| `hanzo-studio.md` | Visual AI engine (ComfyUI fork, port 8188) | `cat skills/hanzo/hanzo-studio.md` |
| `hanzo-flow.md` | Visual workflow builder (Langflow fork, Python) | `cat skills/hanzo/hanzo-flow.md` |
| `hanzo-search.md` | Search engine (Meilisearch v1.37.0 fork, Rust) | `cat skills/hanzo/hanzo-search.md` |

### Infrastructure Utilities (6 skills)

| Skill | Description | Load |
|-------|-------------|------|
| `hanzo-zap.md` | ZAP binary protocol bridge/sidecar | `cat skills/hanzo/hanzo-zap.md` |
| `hanzo-extract.md` | Content extraction & sanitization | `cat skills/hanzo/hanzo-extract.md` |
| `hanzo-suite.md` | Test/development suite | `cat skills/hanzo/hanzo-suite.md` |
| `hanzo-dataroom.md` | Document sharing (Papermark fork) | `cat skills/hanzo/hanzo-dataroom.md` |
| `hanzo-contracts.md` | Foundry contracts (AIToken, AIFaucet, HanzoRegistry) | `cat skills/hanzo/hanzo-contracts.md` |
| `hanzo-nchain.md` | Blockchain node operator | `cat skills/hanzo/hanzo-nchain.md` |

### Guides (1 skill)

| Skill | Description | Load |
|-------|-------------|------|
| `ai-development-guide.md` | How to use AI with Hanzo + Lux | `cat skills/hanzo/ai-development-guide.md` |

## Decision Tree

```
What do you need?
├── Call LLMs → hanzo-chat.md (UI) or python-sdk.md (SDK)
├── Run locally → hanzo-engine.md + hanzo-vllm.md + hanzo-node.md
├── Build agents → hanzo-agent.md + hanzo-mcp.md + hanzo-tools.md + hanzo-aci.md
├── Deploy apps → hanzo-platform.md (PaaS) or hanzo-stack.md (local)
├── Serverless → hanzo-functions.md
├── Kubernetes → hanzo-operator.md + hanzo-charts.md + hanzo-hke.md + hanzo-terraform.md
├── API routing → hanzo-gateway.md (KrakenD) or hanzo-ingress.md
├── Zero-trust networking → hanzo-zt.md (fabric) + hanzo-zrok.md (sharing)
├── DNS → hanzo-dns.md (CoreDNS fork)
├── Auth/Identity → hanzo-iam.md (server) + hanzo-id.md (UI) + hanzo-identity.md
├── Secrets → hanzo-kms.md (9 Vault subsystems)
├── MPC wallets → hanzo-mpc.md (distributed key management)
├── Payments → hanzo-payments.md + hanzo-checkout.md + hanzo-authorizenet.md
├── Treasury → hanzo-treasury.md + hanzo-ledger.md + hanzo-numscript.md
├── Commerce → hanzo-commerce.md (core) + hanzo-billing.md + hanzo-form.md
├── Visual AI → hanzo-studio.md (ComfyUI) or hanzo-flow.md (Langflow)
├── Image gen → hanzo-kensho.md (17B)
├── Audio gen → hanzo-mugen.md + hanzo-koe.md
├── Search → hanzo-search.md (Meilisearch, Rust)
├── Storage → hanzo-s3.md + hanzo-storage.md + hanzo-vector.md + hanzo-pubsub.md + hanzo-stream.md
├── Key-value → hanzo-kv.md + hanzo-kv-go.md
├── SQL/Database → hanzo-sql.md + hanzo-database.md + hanzo-documentdb.md + hanzo-orm.md
├── IDE tools → hanzo-extension.md + hanzo-code.md + hanzo-editor.md
├── Blockchain → hanzo-web3.md + hanzo-contracts.md + hanzo-evm.md + hanzo-explorer.md
├── Monitoring → hanzo-o11y.md + hanzo-sentry.md + hanzo-logs.md + hanzo-metrics.md
├── Product analytics → hanzo-insights.md (PostHog fork, InsightsQL)
├── API specs → hanzo-openapi.md + hanzo-sdk.md + hanzo-trpc-openapi.md
├── Signing → hanzo-sign.md
├── Security → hanzo-guard.md + rust-sdk.md (hanzo-guard crate)
├── ML training → hanzo-gym.md + hanzo-ane.md + hanzo-candle.md
└── Prod infra → hanzo-universe.md (private)
```

## Related Ecosystems

- **Lux Network** (`github.com/luxfi/skills`) — Blockchain infrastructure (31 skills)
- **Zoo Foundation** — Decentralized AI research

---

**Last Updated**: 2026-03-13
**Total Skills**: 126
**Gateway**: `discover-hanzo/SKILL.md`
