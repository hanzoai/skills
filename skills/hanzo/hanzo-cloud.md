# Hanzo Cloud - AI Provider Management Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-id.md`

## Overview

Hanzo Cloud is the **AI provider management platform** -- Go 1.26 Beego MVC backend with React frontend, PostgreSQL database, and integrations for 30+ AI providers. Fork of Casibase. Available at `cloud.hanzo.ai`. Provides the OpenAI-compatible API gateway at `api.hanzo.ai`, project management, API key generation, usage metrics, billing, and team management.

## When to use

- Managing AI provider configurations and API keys
- Monitoring LLM usage and costs across 100+ providers
- Creating projects with per-environment API keys
- Managing team access to AI resources with RBAC
- Setting up billing and subscriptions
- Accessing the OpenAI-compatible API at `api.hanzo.ai/v1`

## Hard requirements

1. **IAM configured** at hanzo.id for authentication
2. **iamEndpoint MUST be public URL** (`https://hanzo.id`), not internal cluster address
3. **PostgreSQL** for project/user/key storage
4. **LLM Gateway** (`hanzoai/llm`) for model routing

## Quick reference

| Item | Value |
|------|-------|
| Dashboard | `https://cloud.hanzo.ai` |
| API Gateway | `https://api.hanzo.ai/v1` (OpenAI-compatible) |
| LLM Endpoint | `https://llm.hanzo.ai` |
| Auth | IAM via hanzo.id |
| Backend | Go 1.26 + Beego MVC |
| Frontend | React (CRA via CRACO, .js + .less) |
| Database | PostgreSQL |
| Upstream | Casibase |
| Repo | `github.com/hanzoai/cloud` |
| K8s manifests | `universe/infra/k8s/cloud/` |
| Image | `ghcr.io/hanzoai/cloud:latest` |
| K8s service | `cloud-api.hanzo.svc:8000` |
| Port | 8000 |

## Architecture

```
cloud.hanzo.ai (dashboard)
        |
   Cloud Backend (Go/Beego, port 8000)
        |
   +----+----+----+
   |    |    |    |
  IAM  PostgreSQL  LLM Gateway   Console
  (hanzo.id)      (llm.hanzo.svc) (console.hanzo.ai)
```

### Integration with LLM Gateway

Cloud's AI bot uses `http://cloud-api.hanzo.svc:8000/api` as `OPENAI_BASE_URL`. The LLM Gateway (`hanzoai/llm`, aka hanzo-llm on PyPI) routes to 100+ providers including all Zen models.

### Integration with Console

Cloud API POSTs traces to `console.hanzo.ai/api/public/ingestion` for centralized observability of all LLM calls.

## Multi-org support

Cloud supports multiple organizations: hanzo, lux, zoo, pars.

### Organization hierarchy

```
Organization (hanzo)
  +-- Project (my-ai-app)
  |     +-- API Key (sk-hanzo-...)
  |     +-- Environment (production)
  |     +-- Environment (staging)
  +-- Project (chatbot)
  +-- Members
        +-- admin@hanzo.ai (OWNER)
        +-- dev@hanzo.ai (DEVELOPER)
```

### Multi-org bootstrap

```bash
HANZO_INIT_ORG_IDS=hanzo,lux,zoo,pars
HANZO_INIT_ORG_NAMES="Hanzo AI,Lux Network,Zoo Foundation,Pars"
HANZO_INIT_USER_EMAIL=z@hanzo.ai
HANZO_INIT_PROJECT_ORG_ID=hanzo
```

Membership bootstrap supports existing users by email -- password only required for brand-new users.

## API key usage

```bash
# Create key via dashboard, then use:
export HANZO_API_KEY=sk-hanzo-...

# OpenAI-compatible chat completions
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-70b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# List available models
curl https://api.hanzo.ai/v1/models \
  -H "Authorization: Bearer ${HANZO_API_KEY}"
```

## Supported providers (100+)

Zen models (14 frontier models), OpenAI, Anthropic, Google (Gemini), Meta (Llama), Mistral, Together AI, Groq, Fireworks, Ollama, Azure OpenAI, AWS Bedrock, Cohere, Replicate, DeepSeek, and many more via the LLM Gateway.

## K8s deployment

The Cloud API runs as `cloud-api` service in the hanzo namespace:

```yaml
# universe/infra/k8s/cloud/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloud-api
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: cloud
          image: ghcr.io/hanzoai/cloud:latest
          ports:
            - containerPort: 8000
          env:
            - name: iamEndpoint
              value: "https://hanzo.id"  # MUST be public URL
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login fails | `iamEndpoint` is internal URL | Set to `https://hanzo.id` (public) |
| API key not working | Key not associated with project | Create key via dashboard under correct project |
| Wrong org after login | Hardcoded org in config | Use `/api/get-app-login` for dynamic resolution |
| 502 on api.hanzo.ai | LLM Gateway down | Check `llm` pod in hanzo namespace |
| Traces not appearing | Console ingestion URL wrong | Verify `CONSOLE_URL` env var |

## Related Skills

- `hanzo/hanzo-console.md` -- Observability and tracing
- `hanzo/hanzo-id.md` -- Authentication
- `hanzo/hanzo-chat.md` -- Chat UI
- `hanzo/hanzo-llm-gateway.md` -- LLM proxy (100+ providers)
- `hanzo/hanzo-billing.md` -- Billing portal

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: cloud, dashboard, api-gateway, billing, providers, casibase
**Prerequisites**: Web browser, Hanzo account
