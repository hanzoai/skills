# Hanzo Cloud - AI Cloud Dashboard & Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-id.md`

## Overview

Hanzo Cloud is the **AI provider management platform** — Go 1.26 Beego MVC backend with React frontend (CRA via CRACO, .js + .less), PostgreSQL database, and integrations for 30+ AI providers. Fork of **Casibase** (upstream: casibase/casibase). Available at `cloud.hanzo.ai`.

### Features

- **30+ AI provider integrations**: OpenAI, Anthropic, Google, Together, Ollama, and more
- **Project management**: Create/manage AI projects and environments
- **API key generation**: Create and rotate API keys per project
- **Usage metrics**: Token usage, cost tracking, request volume per provider
- **Billing**: Subscription management, invoices, payment methods
- **Team management**: Invite members, RBAC roles
- **Multi-org**: Support for hanzo, lux, zoo, pars organizations

### Tech Stack

- **Backend**: Go 1.26, Beego MVC framework
- **Frontend**: React (CRA via CRACO, .js + .less — older setup)
- **Database**: PostgreSQL
- **Auth**: IAM via hanzo.id

## When to use

- Managing AI provider configurations and API keys
- Monitoring LLM usage and costs across providers
- Managing team access to AI resources
- Setting up billing and subscriptions
- Viewing request logs and traces

## Quick reference

| Item | Value |
|------|-------|
| Cloud | `https://cloud.hanzo.ai` |
| Console | `https://console.hanzo.ai` |
| Auth | IAM via hanzo.id |
| API | `https://api.hanzo.ai` |
| Backend | Go 1.26 + Beego MVC |
| Frontend | React |
| Database | PostgreSQL |
| Repo | `github.com/hanzoai/cloud` |

## Multi-Org Bootstrap

Console supports multi-org provisioning on startup:

```bash
# Environment variables for bootstrap
HANZO_INIT_ORG_IDS=hanzo,lux,zoo,pars
HANZO_INIT_ORG_NAMES="Hanzo AI,Lux Network,Zoo Foundation,Pars"
HANZO_INIT_USER_EMAIL=admin@example.com
HANZO_INIT_PROJECT_ORG_ID=hanzo  # Disambiguate for API key bootstrap
```

Membership bootstrap supports existing users by email — password only required for brand-new users.

## Core Concepts

### Organization Hierarchy

```
Organization (hanzo)
├── Project (my-ai-app)
│   ├── API Key (sk-hanzo-...)
│   ├── Environment (production)
│   └── Environment (staging)
├── Project (chatbot)
│   └── ...
└── Members
    ├── admin@example.com (OWNER)
    └── dev@hanzo.ai (DEVELOPER)
```

### API Key Usage

```bash
# Create via dashboard, then use:
export HANZO_API_KEY=sk-hanzo-...

curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{"model":"zen-70b","messages":[{"role":"user","content":"Hello"}]}'
```

## Related Skills

- `hanzo/hanzo-console.md` - Observability and tracing
- `hanzo/hanzo-id.md` - Authentication
- `hanzo/hanzo-chat.md` - LLM API
- `hanzo/hanzo-platform.md` - Deployment platform

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: dashboard, cloud, management, billing
**Prerequisites**: Web browser, Hanzo account
