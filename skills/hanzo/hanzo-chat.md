# Hanzo Chat - Unified LLM API for 86+ Models

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/python-sdk.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Chat provides a **single OpenAI-compatible API** for accessing 86+ models across multiple providers. One API key, one base URL, one SDK -- every model from OpenAI, Anthropic, Google, Meta, Mistral, and Zen available through `api.hanzo.ai/v1`. Streaming, function calling, vision, and tool use all work out of the box.

### Why Hanzo Chat?

- **One API for everything**: 86+ models, 5+ providers, single endpoint
- **OpenAI SDK compatible**: Drop-in replacement -- change base URL, done
- **Zen models included**: 14 Zen frontier models (600M - 480B params)
- **Smart routing**: Automatic fallback and load balancing across providers
- **Cost tracking**: Per-request cost attribution via Hanzo Console
- **Streaming**: SSE streaming for all models, all providers

## When to use

Use this skill when:
- The user wants to call LLMs programmatically
- The user needs multi-model access through a single API
- The user wants to compare models or implement fallback chains
- The user is building an AI application that needs chat completions
- The user wants to use Zen models

## Hard requirements

1. **API Key required.** If `HANZO_API_KEY` is not set, tell the user to get one at https://hanzo.ai/dashboard.
2. **Never expose the API key** in user-visible output, logs, or screenshots.
3. **Respect rate limits.** Back off on 429 responses.

## Preflight checks

Before making any request, silently verify:
- `HANZO_API_KEY` environment variable is set and non-empty
- If unset, suggest: `export HANZO_API_KEY=<your-key>`

## Quick reference

| Item | Value |
|------|-------|
| Base URL | `https://api.hanzo.ai/v1` |
| Auth header | `Authorization: Bearer ${HANZO_API_KEY}` |
| Chat endpoint | `POST /v1/chat/completions` |
| Models endpoint | `GET /v1/models` |
| Embeddings endpoint | `POST /v1/embeddings` |
| Docs | https://hanzo.ai/docs/api |
| Dashboard | https://hanzo.ai/dashboard |
| Chat UI | https://chat.hanzo.ai |

## Available models

### Zen Models (Hanzo frontier models)

| Model | Params | Context | Best for |
|-------|--------|---------|----------|
| `zen-480b` | 480B | 128K | Flagship reasoning, code, analysis |
| `zen-70b` | 70B | 128K | General purpose, fast |
| `zen-32b` | 32B | 128K | Balanced performance/cost |
| `zen-14b` | 14B | 128K | Efficient, edge-capable |
| `zen-7b` | 7B | 128K | Lightweight, fast inference |
| `zen-4b` | 4B | 32K | Ultra-lightweight, mobile |
| `zen-coder-32b` | 32B | 128K | Code generation, review |
| `zen-coder-14b` | 14B | 128K | Code, cost-efficient |
| `zen-embedding` | 600M | 32K | Embeddings (#1 MTEB) |

### Third-party models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o1-mini, o3-mini |
| Anthropic | claude-opus-4-6, claude-sonnet-4, claude-3.5-haiku |
| Google | gemini-2.0-flash, gemini-2.0-pro, gemini-1.5-pro |
| Meta | llama-3.3-70b, llama-3.1-405b |
| Mistral | mistral-large, mistral-medium, codestral |

## One-file quickstart

### curl

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{
    "model": "zen-70b",
    "messages": [{"role": "user", "content": "Explain x402 micropayments in one paragraph."}],
    "temperature": 0.7
  }'
```

### curl (streaming)

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -d '{
    "model": "zen-70b",
    "messages": [{"role": "user", "content": "Write a haiku about blockchains."}],
    "stream": true
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.hanzo.ai/v1",
    api_key=os.environ["HANZO_API_KEY"],
)

response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello, Hanzo!"}],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Python (streaming)

```python
stream = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Explain consensus algorithms."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Python (Hanzo SDK)

```python
from hanzo import Hanzo

client = Hanzo()  # uses HANZO_API_KEY from env

response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### TypeScript

```typescript
import OpenAI from "openai"

const client = new OpenAI({
  baseURL: "https://api.hanzo.ai/v1",
  apiKey: process.env.HANZO_API_KEY,
})

const response = await client.chat.completions.create({
  model: "zen-70b",
  messages: [{ role: "user", content: "Hello, Hanzo!" }],
})

console.log(response.choices[0].message.content)
```

### Go

```go
import "github.com/hanzoai/go-sdk"

client := hanzo.NewClient(os.Getenv("HANZO_API_KEY"))

resp, err := client.Chat.Completions.Create(ctx, hanzo.ChatCompletionParams{
    Model:    "zen-70b",
    Messages: []hanzo.Message{{Role: "user", Content: "Hello!"}},
})
```

## Endpoint selector

| Task | Endpoint | Method |
|------|----------|--------|
| Chat completion | `POST /v1/chat/completions` | POST |
| List models | `GET /v1/models` | GET |
| Get model info | `GET /v1/models/{id}` | GET |
| Embeddings | `POST /v1/embeddings` | POST |
| Moderations | `POST /v1/moderations` | POST |

## Function calling / Tool use

```python
response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }],
)

tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name == "get_weather"
# tool_call.function.arguments == '{"location": "Tokyo"}'
```

## Vision (multimodal)

```python
response = client.chat.completions.create(
    model="zen-70b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
)
```

## MCP Integration

Expose chat completions as MCP tools:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'

const chatTool: Tool = {
  name: 'hanzo_chat',
  description: 'Generate text using Hanzo Chat API (86+ models)',
  parameters: {
    model: { type: 'string', default: 'zen-70b' },
    prompt: { type: 'string', required: true },
    temperature: { type: 'number', default: 0.7 },
    max_tokens: { type: 'number', default: 1024 }
  },
  async execute({ model, prompt, temperature, max_tokens }) {
    const res = await fetch('https://api.hanzo.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HANZO_API_KEY}`
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        max_tokens
      })
    })
    const data = await res.json()
    return data.choices[0].message.content
  }
}
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check model name, message format |
| 401 | Unauthorized | Check API key |
| 404 | Model not found | Check model ID against /v1/models |
| 429 | Rate limited | Exponential backoff (1s, 2s, 4s, 8s) |
| 500 | Server error | Retry up to 3 times |
| 503 | Provider down | Hanzo auto-retries with fallback provider |

## Official links

- Chat UI: https://chat.hanzo.ai
- API Docs: https://hanzo.ai/docs/api
- Dashboard: https://hanzo.ai/dashboard
- Python SDK: https://github.com/hanzoai/python-sdk
- Go SDK: https://github.com/hanzoai/go-sdk
- JS SDK: https://github.com/hanzoai/js-sdk
- Hanzo AI: https://hanzo.ai

---

**Last Updated**: 2026-02-26
**Category**: Hanzo Ecosystem
**Related**: llm, ai, chat, inference
**Prerequisites**: HTTP/curl, any OpenAI-compatible SDK
