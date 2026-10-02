---
name: hanzo-llm
description: "Hanzo LLM Gateway — unified proxy for 100+ AI providers. Route requests, track costs, manage virtual keys, rate limiting, and model fallbacks."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzoai",
              "label": "Install Hanzo AI SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo LLM Gateway — AI Provider Proxy

`pip install hanzoai`

Unified proxy for 100+ LLM providers with cost tracking, rate limiting, virtual keys, model fallbacks, and load balancing. OpenAI-compatible API.

## Quick Start

```python
from hanzoai import Hanzo

client = Hanzo(api_key="sk-hanzo-...")

# Route to any provider via unified API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Provider Routing

```python
# OpenAI
client.chat.completions.create(model="gpt-4o", ...)

# Anthropic Claude
client.chat.completions.create(model="claude-sonnet-4-20250514", ...)

# Google Gemini
client.chat.completions.create(model="gemini/gemini-2.0-flash", ...)

# AWS Bedrock
client.chat.completions.create(model="bedrock/anthropic.claude-3-sonnet", ...)

# Azure OpenAI
client.chat.completions.create(model="azure/gpt-4", ...)

# Ollama (local)
client.chat.completions.create(model="ollama/llama3", ...)

# Together AI
client.chat.completions.create(model="together_ai/meta-llama/Llama-3-70b", ...)
```

## Model Fallbacks

```python
# Configure fallback chain via gateway config
# If primary model fails, automatically try next
# gpt-4 → claude-sonnet → gemini-pro
```

## Cost Tracking

```python
# Get global spend
spend = client.global_spend.retrieve()

# Per-team spend
team_spend = client.global_spend.list_spend_logs(api_key="sk-...")

# Per-key spend tracking
key_info = client.key.retrieve(key="sk-...")
print(f"Spend: ${key_info.spend}")
```

## Virtual Keys

```python
# Create virtual key with budget
key = client.key.generate(
    team_id="team-123",
    max_budget=100.0,  # $100 budget
    models=["gpt-4", "claude-sonnet-4-20250514"],  # Allowed models
)

# List keys
keys = client.key.list()
```

## Rate Limiting

```python
# Per-key rate limits configured at gateway level
# Supports: requests/min, tokens/min, requests/day
```

## Team Management

```python
# Create team
client.team.create(team_alias="engineering")

# List teams
teams = client.team.list()

# Organization management
orgs = client.organization.list()
```

## Streaming

```python
for chunk in client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The food was delicious"
)
vector = response.data[0].embedding
```

## Supported Providers (100+)

OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Together AI, Cohere, Mistral, Groq, DeepSeek, Ollama, VLLM, HuggingFace, Replicate, Perplexity, Fireworks AI, and more.

## Port

- Gateway: `4000`

## Environment Variables

```bash
HANZO_API_KEY=sk-hanzo-...            # API key
HANZO_BASE_URL=https://api.hanzo.ai   # Gateway URL (or http://localhost:4000)
```
