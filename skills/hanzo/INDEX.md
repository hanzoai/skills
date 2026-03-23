# Hanzo Ecosystem Skills

**Complete guide to Hanzo AI infrastructure, services, and developer tools**

## Overview

Hanzo is a comprehensive AI ecosystem -- from frontier models (Zen MoDE) to production infrastructure (PaaS, KMS, IAM) to developer tools (SDKs, CLI, MCP). 131 skills cover every component.

### Key Rules

- **KMS for all secrets** -- never hardcode in manifests, never plaintext
- **GHCR for images** -- `ghcr.io/hanzoai/<service>`, always `--platform linux/amd64`
- **Kustomize, not Helm** -- `kubectl kustomize . | kubectl apply -f -`
- **No nginx, no caddy** -- Hanzo Ingress (Traefik) handles all routing
- **Self-hosted runners only** -- never GitHub Actions billing
- **IAM for all auth** -- hanzo.id, RFC 6749/OIDC endpoints only
- **Universe is source of truth** -- all K8s manifests in `hanzoai/universe`
- **Zen models: qwen3+ only** -- never qwen2, never reference upstream names
- **luxfi packages** -- never go-ethereum, never ava-labs

### Production Clusters

| Cluster | Context | LB IP | Purpose |
|---------|---------|-------|---------|
| hanzo-k8s | `do-sfo3-hanzo-k8s` | `165.232.146.176` | All Hanzo services (22 nodes) |
| lux-k8s | `do-sfo3-lux-k8s` | `24.144.69.101` | Lux blockchain |

## Skill Catalog

### Infrastructure & Deployment (8 skills)

**Hanzo K8s** (`hanzo-k8s.md`)
K8s infrastructure. hanzo-k8s (prod, 22 nodes), lux-k8s (blockchain). DOKS on DigitalOcean SFO3. Kustomize deploys.

**Hanzo Deploy** (`hanzo-deploy.md`)
Deployment guide. Universe as source of truth, main=prod, dev=staging. Self-hosted runners, GHCR images, KMS secrets.

**Hanzo Ingress** (`hanzo-ingress.md`)
L7 reverse proxy (Traefik fork). IngressClass `hanzo`, auto TLS via Let's Encrypt, middleware (rate limit, auth, compress).

**Hanzo Static** (`hanzo-static.md`)
Static file server plugin for Traefik. SPA serving, cache headers, gzip. No nginx/caddy.

**Hanzo DNS** (`hanzo-dns.md`)
Programmable DNS (CoreDNS fork). Custom `hanzoapi` plugin. Internal resolution + Cloudflare external.

**Hanzo ZT** (`hanzo-zt.md`)
Zero-trust network fabric (OpenZiti fork). mTLS/PKI, dark services, smart routing, end-to-end encryption.

**Hanzo Universe** (`hanzo-universe.md`)
Production K8s infrastructure manifests (private repo). 40+ service directories.

**Hanzo Terraform** (`hanzo-terraform.md`)
Terraform provider for Hanzo infrastructure.

### Identity & Security (7 skills)

**Hanzo ID** (`hanzo-id.md`)
Identity at hanzo.id. Multi-org IAM (Casdoor), OIDC/OAuth2, 4 orgs (hanzo/lux/zoo/pars). RFC 6749 endpoints. Cloudflare Worker edge auth.

**Hanzo KMS** (`hanzo-kms.md`)
Secrets at kms.hanzo.ai (Infisical fork). 9 Vault subsystems. KMSSecret CRDs for K8s sync. Universal Auth.

**Hanzo IAM** (`hanzo-iam.md`)
IAM server (Casdoor fork, Go/Beego). OAuth2/OIDC/SAML/LDAP backend.

**Hanzo Identity** (`hanzo-identity.md`)
Identity service and certificate management.

**Hanzo Vault** (`hanzo-vault.md`)
PCI-compliant card tokenization (Go).

**Hanzo Guard** (`hanzo-guard.md`)
Security middleware and threat detection.

**Hanzo MPC** (`hanzo-mpc.md`)
Multi-party computation for distributed wallets.

### Cloud Services & Apps (12 skills)

**Hanzo Cloud** (`hanzo-cloud.md`)
AI provider management at cloud.hanzo.ai (Casibase fork). OpenAI-compatible API at api.hanzo.ai. 100+ providers, billing, team management.

**Hanzo Console** (`hanzo-console.md`)
Observability at console.hanzo.ai (Langfuse fork). LLM traces, token usage, cost tracking, prompt management, datasets.

**Hanzo Platform** (`hanzo-platform.md`)
PaaS at platform.hanzo.ai (Dokploy fork). Git-push deploys, Hono.js API, Next.js 16 frontend.

**Hanzo Chat** (`hanzo-chat.md`)
Chat at chat.hanzo.ai (LibreChat fork). Multi-model, 14 Zen + 100+ third-party, agents, RAG, MCP tools.

**Hanzo Team** (`hanzo-team.md`)
Collaboration at hanzo.team (Huly fork). Project management, chat, docs. IAM-only login, Svelte frontend.

**Hanzo Bot** (`hanzo-bot.md`)
Multi-channel AI gateway. 50+ messaging adapters (Discord, Telegram, Slack, WhatsApp, etc.). ACP gateway.

**Hanzo Playground** (`hanzo-playground.md`)
Bot control plane at app.hanzo.bot. Node registry, workflows, memory scopes, DID/VC audit.

**Hanzo Billing** (`hanzo-billing.md`)
Billing portal at billing.hanzo.ai. Subscriptions, Web3 payments (7 chains), multi-org, OIDC/PKCE auth.

**Hanzo App** (`hanzo-app.md`)
AI-powered developer platform and desktop assistant.

**Hanzo Website** (`hanzo-website.md`)
Corporate website (hanzo.ai).

**Hanzo Store** (`hanzo-store.md`)
AI app store / marketplace.

**Hanzo Analytics** (`hanzo-analytics.md`)
Analytics and tracking service.

### Core AI Infrastructure (12 skills)

**Hanzo Engine** (`hanzo-engine.md`)
Rust inference and embedding engine (mistral.rs fork).

**Hanzo vLLM** (`hanzo-vllm.md`)
Rust-based LLM inference engine (candle-vllm fork).

**Hanzo Node** (`hanzo-node.md`)
Rust AI agent node with P2P/MCP/WASM.

**Hanzo Network** (`hanzo-network.md`)
Distributed AI agent topology.

**Hanzo LLM Gateway** (`hanzo-llm-gateway.md`)
Unified LLM proxy (100+ providers). Package: `hanzo-llm` on PyPI. Docker: `ghcr.io/hanzoai/llm`.

**Hanzo MCP** (`hanzo-mcp.md`)
Model Context Protocol. 260+ tools, 13 HIP-0300 unified tools. ZAP-MCP bridge.

**Hanzo Agent** (`hanzo-agent.md`)
Multi-agent SDK (`hanzoai` v0.0.4, OpenAI agents fork).

**Hanzo ACI** (`hanzo-aci.md`)
Agent Computer Interface (tool/action abstraction).

**Hanzo Tools** (`hanzo-tools.md`)
Unified AI agent tool registry (88 tools, Rust).

**Hanzo Computer** (`hanzo-computer.md`)
Desktop automation and computer use.

**Hanzo Operative** (`hanzo-operative.md`)
Computer use agent (Anthropic fork).

**Hanzo Memory** (`hanzo-memory.md`)
Memory API server for AI agents.

### Protocol & Bridge (2 skills)

**Hanzo ZAP** (`hanzo-zap.md`)
ZAP protocol bridge sidecar. Native TCP binary protocol on port 9651. SQL/KV/Datastore/DocumentDB modes. ZAP-MCP 1:1 mapping.

**Hanzo Zrok** (`hanzo-zrok.md`)
Zero-trust sharing platform (zrok fork). Public/private shares through ZT fabric.

### Platform & Deployment (8 skills)

**Hanzo Gateway** (`hanzo-gateway.md`)
API gateway (KrakenD/Lura, 147+ endpoints, Go).

**Hanzo Tunnel** (`hanzo-tunnel.md`)
WebSocket tunnel and bot agent bridge (Rust + Python).

**Hanzo Operator** (`hanzo-operator.md`)
K8s operator for Hanzo services.

**Hanzo Charts** (`hanzo-charts.md`)
Helm charts for all Hanzo services.

**Hanzo HKE** (`hanzo-hke.md`)
Managed Kubernetes Engine.

**Hanzo VM** (`hanzo-vm.md`)
Cloud OS and VM management.

**Hanzo Functions** (`hanzo-functions.md`)
Serverless functions platform.

**Hanzo Stack** (`hanzo-stack.md`)
Full local dev environment.

### SDKs & Libraries (15 skills)

**Python SDK** (`python-sdk.md`)
Python SDK (`hanzoai` v2.2.0, Stainless, 9 packages).

**JS SDK** (`js-sdk.md`)
TypeScript SDK (`hanzoai` v0.1.0-alpha.2, Stainless).

**Go SDK** (`go-sdk.md`)
Go SDK (`github.com/hanzoai/go-sdk` v0.1.0-alpha.4).

**Rust SDK** (`rust-sdk.md`)
Rust SDK (v1.1.12, 14 crates: agents, MCP, PQC, guard).

**Hanzo SDK** (`hanzo-sdk.md`)
Unified multi-language CLI and client library.

**Hanzo OpenAPI** (`hanzo-openapi.md`)
Unified API spec for 26 cloud services.

**Hanzo tRPC OpenAPI** (`hanzo-trpc-openapi.md`)
tRPC-to-OpenAPI adapter.

**Hanzo Base** (`hanzo-base.md`)
Shared base library and utilities.

**Hanzo ORM** (`hanzo-orm.md`)
Go generics ORM with auto-serialization + KV cache.

**Hanzo Log** (`hanzo-log.md`)
Structured logging library (Go).

**Hanzo KV Go** (`hanzo-kv-go.md`) | **Hanzo PubSub Go** (`hanzo-pubsub-go.md`) | **Hanzo Search Go** (`hanzo-search-go.md`) | **Hanzo Vector Go** (`hanzo-vector-go.md`)
Go client libraries for KV, PubSub, Search, Vector stores.

**Hanzo Edge** (`hanzo-edge.md`)
Edge computing and CDN.

### Frontend & UI (5 skills)

**Hanzo UI** (`hanzo-ui.md`)
React component library (shadcn/ui fork, 161+ components, React 19, Tailwind 4).

**Hanzo Brand** (`hanzo-brand.md`)
Brand guidelines, colors, logos.

**Hanzo Desktop** (`hanzo-desktop.md`)
AI desktop app (Tauri + React Native).

**Hanzo Docs** (`hanzo-docs.md`)
Documentation site (fumadocs fork, Next.js/MDX).

**Hanzo Extension** (`hanzo-extension.md`)
Browser and IDE extensions.

### Data & Observability (15 skills)

**Hanzo O11y** (`hanzo-o11y.md`)
Full-stack observability (SigNoz fork). ClickHouse + OTEL. Logs, metrics, traces. APM for infrastructure.

**Hanzo Database** (`hanzo-database.md`) | **Hanzo SQL** (`hanzo-sql.md`)
PostgreSQL with AI extensions (`ghcr.io/hanzoai/sql`).

**Hanzo KV** (`hanzo-kv.md`)
Valkey/Redis (`ghcr.io/hanzoai/kv`).

**Hanzo DocumentDB** (`hanzo-documentdb.md`)
MongoDB-compatible (FerretDB over PostgreSQL).

**Hanzo Datastore** (`hanzo-datastore.md`) | **Hanzo Vector** (`hanzo-vector.md`)
ClickHouse analytics and Qdrant vector search.

**Hanzo S3** (`hanzo-s3.md`) | **Hanzo Storage** (`hanzo-storage.md`)
S3-compatible object storage (MinIO).

**Hanzo PubSub** (`hanzo-pubsub.md`) | **Hanzo Stream** (`hanzo-stream.md`)
NATS pub/sub and Kafka event streaming.

**Hanzo Logs** (`hanzo-logs.md`) | **Hanzo Metrics** (`hanzo-metrics.md`)
Loki log aggregation and VictoriaMetrics time-series.

**Hanzo Sentry** (`hanzo-sentry.md`) | **Hanzo Insights** (`hanzo-insights.md`)
Error tracking (Sentry fork) and product analytics (PostHog fork).

### AI Models & Training (8 skills)

**Zen LM** (`zenlm.md`)
Zen frontier models (MoDE architecture, 600M-480B params, 14 models).

**Hanzo Gym** (`hanzo-gym.md`) | **Hanzo Jin** (`hanzo-jin.md`) | **Hanzo Kensho** (`hanzo-kensho.md`)
Training platform, multimodal LLM, image generation.

**Hanzo Koe** (`hanzo-koe.md`) | **Hanzo Mugen** (`hanzo-mugen.md`) | **Hanzo Sho** (`hanzo-sho.md`)
Text-to-dialogue, audio generation, text diffusion.

**Hanzo ML** (`hanzo-ml.md`)
ML utilities and shared libraries.

### ML Infrastructure (3 skills)

**Hanzo Candle** (`hanzo-candle.md`)
Rust ML framework (HF candle fork).

**Hanzo ANE** (`hanzo-ane.md`)
Apple Neural Engine training (M1+).

**Hanzo EVM** (`hanzo-evm.md`)
Rust EVM (reth fork).

### Commerce & Finance (14 skills)

**Hanzo Commerce** (`hanzo-commerce.md`) | **Hanzo Commerce API** (`hanzo-commerce-api.md`)
Multi-tenant e-commerce (Go, 100+ models).

**Hanzo Checkout** (`hanzo-checkout.md`) | **Hanzo Form** (`hanzo-form.md`)
Payment checkout and form handling.

**Hanzo Ledger** (`hanzo-ledger.md`) | **Hanzo Treasury** (`hanzo-treasury.md`)
Double-entry financial ledger and programmable treasury.

**Hanzo Numscript** (`hanzo-numscript.md`) | **Hanzo Reconciliation** (`hanzo-reconciliation.md`)
Financial transaction DSL and auto-matching.

**Hanzo Payments** (`hanzo-payments.md`) | **Hanzo Sign** (`hanzo-sign.md`)
Payment processing and document signing.

**Hanzo AuthorizeNet** (`hanzo-authorizenet.md`) | **Hanzo GoChimp** (`hanzo-gochimp.md`)
AuthorizeNet and MailChimp Go clients.

**Hanzo Registry** (`hanzo-registry.md`) | **Hanzo Explorer** (`hanzo-explorer.md`) | **Hanzo Faucet** (`hanzo-faucet.md`)
Container registry, blockchain explorer, token faucet.

### Developer Tools (10 skills)

**Hanzo Dev** (`hanzo-dev.md`) | **Hanzo CLI** (`hanzo-cli.md`)
Terminal AI agent and CLI.

**Hanzo Code** (`hanzo-code.md`) | **Hanzo Editor** (`hanzo-editor.md`)
Web-based code editor and rich text editor.

**Hanzo Flow** (`hanzo-flow.md`) | **Hanzo Live** (`hanzo-live.md`)
Visual workflow builder and real-time AI pipelines.

**Hanzo REPL** (`hanzo-repl.md`) | **Hanzo Relay** (`hanzo-relay.md`)
Interactive REPL and client relay.

**Hanzo Skill** (`hanzo-skill.md`) | **Hanzo Studio** (`hanzo-studio.md`)
Skill installer CLI and visual AI engine (ComfyUI fork).

### Blockchain (5 skills)

**Hanzo Web3** (`hanzo-web3.md`) | **Hanzo Web3 Gateway** (`hanzo-web3-gateway.md`)
Blockchain API (100+ chains) and x402 micropayments.

**Hanzo Contracts** (`hanzo-contracts.md`)
Foundry contracts (AIToken, AIFaucet, HanzoRegistry).

**Hanzo NChain** (`hanzo-nchain.md`)
Blockchain node operator.

### Infrastructure Utilities (5 skills)

**Hanzo Runtime** (`hanzo-runtime.md`)
Runtime environment for AI workloads.

**Hanzo Extract** (`hanzo-extract.md`)
Content extraction and sanitization.

**Hanzo Suite** (`hanzo-suite.md`)
Test/development suite.

**Hanzo Dataroom** (`hanzo-dataroom.md`)
Document sharing (Papermark fork).

**Hanzo HIPs** (`hanzo-hips.md`)
Hanzo Improvement Proposals.

### Guides (1 skill)

**AI Development Guide** (`ai-development-guide.md`)
How to use AI with Hanzo + Lux.

## Decision Tree

```
What do you need?
+-- Deploy a service              -> hanzo-deploy.md + hanzo-k8s.md
+-- K8s infrastructure            -> hanzo-k8s.md + hanzo-ingress.md
+-- PaaS (deploy apps)            -> hanzo-platform.md
+-- Authentication                -> hanzo-id.md (IAM) + hanzo-iam.md (server)
+-- Secrets                       -> hanzo-kms.md
+-- Call LLMs                     -> hanzo-chat.md (UI) or python-sdk.md (SDK)
+-- LLM cost tracking             -> hanzo-console.md
+-- Infrastructure monitoring     -> hanzo-o11y.md
+-- Build agents                  -> hanzo-agent.md + hanzo-mcp.md + hanzo-tools.md
+-- Bot on Discord/Telegram/Slack -> hanzo-bot.md
+-- Orchestrate bots              -> hanzo-playground.md
+-- Team collaboration            -> hanzo-team.md
+-- Billing/payments              -> hanzo-billing.md
+-- React components              -> hanzo-ui.md
+-- API routing                   -> hanzo-gateway.md (behind ingress)
+-- Zero-trust networking         -> hanzo-zt.md + hanzo-zrok.md
+-- DNS                           -> hanzo-dns.md
+-- Database access via MCP       -> hanzo-zap.md
+-- Static site serving           -> hanzo-static.md
+-- AI provider management        -> hanzo-cloud.md
+-- Run locally                   -> hanzo-engine.md + hanzo-stack.md
+-- Prod infra manifests          -> hanzo-universe.md (private)
```

## Related Ecosystems

- **Lux Network** (`github.com/luxfi/skills`) -- Blockchain infrastructure (79 skills)
- **Zoo Foundation** -- Decentralized AI research

---

**Last Updated**: 2026-03-23
**Total Skills**: 131
**Gateway**: `discover-hanzo/SKILL.md`
