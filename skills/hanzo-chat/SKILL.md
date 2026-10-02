---
name: hanzo-chat
description: "Hanzo Chat — AI chat platform with 14 Zen models, 100+ third-party models, MCP tools, code execution, and OpenID Connect auth."
metadata: { "bot": { "requires": { "bins": ["curl"] } } }
---

# Hanzo Chat — AI Chat Platform

AI chat application with 14 Zen models, 100+ third-party models via Hanzo LLM Gateway, MCP tool integration, and code execution.

## Quick Start

```bash
# Docker
docker compose up -d

# Access at http://localhost:3081
```

## API Usage

```bash
CHAT_URL=http://localhost:3081

# Send chat message
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }' "$CHAT_URL/api/chat" | jq

# List available models
curl -s "$CHAT_URL/api/models" | jq

# Get conversation history
curl -s -H "Authorization: Bearer $TOKEN" \
  "$CHAT_URL/api/messages?conversationId=..." | jq
```

## Models

### Zen Models (14 proprietary)

- Zen 1 through Zen 14 — Hanzo's own model family

### Third-Party (100+ via Gateway)

- OpenAI: GPT-4, GPT-4o, o1, o3
- Anthropic: Claude Opus, Sonnet, Haiku
- Google: Gemini 2.0
- Meta: Llama 3
- And 90+ more providers

## Features

- **MCP Tools**: Full Model Context Protocol integration
- **Code Execution**: Run code directly in chat
- **Chat History**: Persistent via MongoDB
- **Search**: Full-text search via Meilisearch
- **Auth**: OpenID Connect (Hanzo IAM)
- **Streaming**: Real-time token streaming

## Configuration

```bash
# Domain
DOMAIN_CLIENT=http://localhost:3081
DOMAIN_SERVER=http://localhost:3081

# Database
MONGO_URI=mongodb://user:pass@mongodb:27017/HanzoChat
MEILI_HOST=http://meilisearch:7700
MEILI_MASTER_KEY=...

# Auth
JWT_SECRET=...
OPENID_CLIENT_ID=...
OPENID_CLIENT_SECRET=...
OPENID_ISSUER=https://hanzo.id

# AI (via Hanzo Gateway)
OPENAI_API_KEY=sk-hanzo-...
OPENAI_BASE_URL=https://api.hanzo.ai/v1

# Features
MCP_ENABLED=true
```

## Port

- Web UI / API: `3081`
