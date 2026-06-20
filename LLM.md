# LLM.md - Hanzo Skills

## Overview

Context-efficient development knowledge via progressive skill discovery. 131 skill files covering the entire Hanzo AI ecosystem.

## Structure

```
skills/
  skills/
    hanzo/              # 131 Hanzo ecosystem skills (flat markdown)
      INDEX.md          # Catalog and decision tree
      hanzo-k8s.md      # K8s infrastructure
      hanzo-deploy.md   # Deployment guide
      hanzo-platform.md # PaaS (Dokploy fork)
      hanzo-id.md       # Identity/IAM (Casdoor)
      hanzo-kms.md      # Secrets (Infisical fork)
      hanzo-ingress.md  # Ingress (Traefik fork)
      hanzo-cloud.md    # AI provider mgmt (Casibase)
      hanzo-console.md  # LLM observability (Langfuse)
      hanzo-chat.md     # Chat UI (LibreChat fork)
      hanzo-team.md     # Collaboration (Huly fork)
      hanzo-bot.md      # Multi-channel bot gateway
      hanzo-mcp.md      # Model Context Protocol
      hanzo-zap.md      # ZAP protocol bridge
      hanzo-o11y.md     # Observability (SigNoz fork)
      ... (131 total)
    # 80+ other skill directories (api, cloud, ml, etc.)
  commands/             # Custom commands
  scripts/              # Helper scripts
  tests/                # Test suite
```

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
- `hanzo-cloud.md` -- API gateway at api.hanzo.ai, 100+ providers
- `hanzo-console.md` -- LLM traces, costs, prompts (Langfuse)
- `hanzo-chat.md` -- Multi-model chat UI (14 Zen + 100+ third-party)
- `hanzo-mcp.md` -- 260+ tools, 13 HIP-0300 unified tools
- `hanzo-bot.md` -- 50+ messaging adapters, ACP gateway

### Data
- `hanzo-zap.md` -- ZAP sidecar bridges SQL/KV/Datastore/DocumentDB to MCP
- `hanzo-o11y.md` -- Infrastructure APM (SigNoz, ClickHouse, OTEL)

## Hard Rules

1. KMS for all secrets -- never plaintext in manifests
2. GHCR for images -- `ghcr.io/hanzoai/*`, `--platform linux/amd64`
3. Kustomize, not Helm
4. No nginx, no caddy -- Hanzo Ingress only
5. Self-hosted runners only -- no GHA billing
6. IAM for all auth -- RFC 6749 endpoints, implicit flow preferred
7. Universe repo is source of truth for K8s manifests
8. Zen models: qwen3+ only, never reference upstream names
9. luxfi packages only, never go-ethereum or ava-labs
10. compose.yml not docker-compose.yml

## Production Topology

```
hanzo-k8s (do-sfo3-hanzo-k8s, 22 nodes, LB: 165.232.146.176)
  +-- IAM (hanzo.id, lux.id, zoo.id, pars.id)
  +-- KMS (kms.hanzo.ai)
  +-- Platform (platform.hanzo.ai)
  +-- Cloud (cloud.hanzo.ai)
  +-- Console (console.hanzo.ai)
  +-- Chat (chat.hanzo.ai)
  +-- Team (hanzo.team)
  +-- Gateway (api.hanzo.ai, llm.hanzo.ai)
  +-- Billing (billing.hanzo.ai)
  +-- O11y (o11y.hanzo.ai)
  +-- Bot, Playground, DNS, ZT, ...
  +-- PostgreSQL (postgres.hanzo.svc)
  +-- Redis/Valkey (redis.hanzo.svc)
  +-- MongoDB (mongodb.hanzo.svc)
  +-- MinIO (minio.hanzo.svc)

lux-k8s (do-sfo3-lux-k8s, LB: 24.144.69.101)
  +-- 15 validators
  +-- Gateway (api.lux.network)
  +-- Markets, Cloud
```
