# Hanzo LLM Gateway - Unified AI Proxy for 100+ Providers

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-chat.md`, `hanzo/python-sdk.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo LLM Gateway is a **unified proxy** that routes requests to 100+ LLM providers through a single OpenAI-compatible API. Fork of LiteLLM with Hanzo routing, cost tracking, and provider fallback. Powers `api.hanzo.ai/v1`.

### Why Hanzo LLM Gateway?

- **One endpoint, all models**: OpenAI, Anthropic, Google, Meta, Mistral, Zen + 100 more
- **Smart routing**: Load balancing, fallback chains, cost optimization
- **Cost tracking**: Per-request attribution via Hanzo Console
- **Self-hostable**: Run your own gateway with custom provider keys
- **Rate limit handling**: Automatic retry with provider rotation

### OSS Base

Fork of **LiteLLM** proxy. Repo: `github.com/hanzoai/llm`.

## When to use

- Running a centralized LLM proxy for your team
- Routing between multiple AI providers
- Cost tracking and budget enforcement
- Self-hosting LLM access behind your firewall
- Adding custom models or providers

## Hard requirements

1. **At least one provider API key** (OpenAI, Anthropic, etc.)
2. **Port 4000** available (default)
3. **PostgreSQL** for logging (optional)

## Quick reference

| Item | Value |
|------|-------|
| Public endpoint | `https://api.hanzo.ai/v1` |
| Internal endpoint | `http://llm.hanzo.svc.cluster.local:4000/v1` |
| Port | 4000 |
| Config | `config.yaml` or env vars |
| Dashboard | `https://llm.hanzo.ai` (Cloud UI, NOT LLM endpoint) |
| Repo | `github.com/hanzoai/llm` |

## One-file quickstart

### Docker

```bash
docker run -d --name hanzo-llm \
  -p 4000:4000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  ghcr.io/hanzoai/llm:latest
```

### Config file

```yaml
# config.yaml
model_list:
  - model_name: zen-70b
    llm_params:
      model: together_ai/Qwen/Qwen3-235B-A22B
      api_key: os.environ/TOGETHER_API_KEY

  - model_name: gpt-4o
    llm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-sonnet
    llm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: least-busy
  num_retries: 3
  fallbacks:
    - zen-70b: [gpt-4o, claude-sonnet]

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

### Make commands

```bash
# Clone from github.com/hanzoai first
cd <project>
make dev              # Start dev server (port 4000)
make up               # Docker compose up
docker compose up -d  # Alternative
```

### Test request

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-70b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Core Concepts

### Provider Routing

```
Client Request
    │
    ▼
┌──────────────────┐
│  LLM Gateway     │
│  (port 4000)     │
├──────────────────┤
│ Router:          │
│ ├─ least-busy    │──▶ OpenAI
│ ├─ fallback      │──▶ Anthropic
│ └─ cost-based    │──▶ Together AI
├──────────────────┤    ▲
│ Logging:         │    │
│ └─ PostgreSQL    │    │
│ └─ Console       │────┘ (cost tracking)
└──────────────────┘
```

### Fallback Chains

When primary provider fails (429, 500, timeout), gateway automatically routes to fallback:

```yaml
fallbacks:
  - zen-70b: [gpt-4o, claude-sonnet]  # Try zen → GPT-4o → Claude
  - gpt-4o: [claude-sonnet]            # Try GPT-4o → Claude
```

### Budget & Rate Limits

```yaml
general_settings:
  max_budget: 100.00           # USD per month
  budget_duration: 1m          # Reset monthly
  max_parallel_requests: 100   # Concurrent limit
```

### Zen Model Mapping

Zen models map to upstream providers (private config in ``github.com/hanzoai/zen`gateway/config.yaml`):

**BRAND POLICY**: Never reference upstream model names in public-facing contexts. Zen models are presented as Hanzo's own architecture: **Zen MoDE (Mixture of Diverse Experts)**.

## Production Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-llm
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm
        image: ghcr.io/hanzoai/llm:latest
        ports:
        - containerPort: 4000
        env:
        - name: LITELLM_MASTER_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: master-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: database-url
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 on requests | Wrong master key | Check LITELLM_MASTER_KEY |
| Provider timeout | Upstream provider slow | Increase timeout or add fallback |
| Cost not tracking | No DATABASE_URL | Add PostgreSQL connection |
| Model not found | Not in config | Add to config.yaml |

## Related Skills

- `hanzo/hanzo-chat.md` - Chat API (uses this gateway)
- `hanzo/hanzo-console.md` - Observability (receives cost data)
- `hanzo/python-sdk.md` - Client library
- `hanzo/zenlm.md` - Zen model family

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: llm, proxy, litellm, gateway, routing
**Prerequisites**: LLM API concepts, Docker
