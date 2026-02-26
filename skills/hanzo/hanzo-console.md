# Hanzo Console - AI Observability and Prompt Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-chat.md`, `hanzo/hanzo-mcp.md`, `hanzo/python-sdk.md`

## Overview

Hanzo Console is the **observability and prompt management layer** for AI applications. It captures traces, scores, datasets, and prompts from any LLM application and provides a dashboard for debugging, evaluation, and cost analysis. Compatible with the Langfuse SDK and OpenTelemetry, it integrates natively with every Hanzo service.

### Why Hanzo Console?

- **Full trace capture**: Every LLM call, tool use, and agent step recorded
- **Cost attribution**: Per-request, per-user, per-model cost breakdown
- **Prompt management**: Version, test, and deploy prompts from a central registry
- **Datasets & evals**: Build evaluation sets and run automated scoring
- **Langfuse compatible**: Drop-in replacement for Langfuse SDK
- **Part of Hanzo ecosystem**: Auto-instrumented for Hanzo Chat, Web3, Commerce

## When to use

Use this skill when:
- The user wants to trace and debug LLM calls
- The user needs cost analysis across models and providers
- The user wants to manage prompt versions and deployments
- The user needs to build evaluation datasets and run scoring
- The user wants to monitor AI application performance

## Hard requirements

1. **API Key required.** Use `HANZO_API_KEY` or the Langfuse-compatible pair `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`. Get keys at https://console.hanzo.ai.
2. **Never expose keys** in user-visible output, logs, or screenshots.
3. **Traces are async.** The SDK batches and sends traces in the background. Call `flush()` before process exit.

## Preflight checks

Before making any request, silently verify:
- `HANZO_API_KEY` is set, OR both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
- If using Langfuse SDK, set `LANGFUSE_HOST=https://console.hanzo.ai`

## Quick reference

| Item | Value |
|------|-------|
| Dashboard | `https://console.hanzo.ai` |
| API Base URL | `https://console.hanzo.ai/api` |
| Langfuse host | `https://console.hanzo.ai` |
| Auth (Hanzo) | `Authorization: Bearer ${HANZO_API_KEY}` |
| Auth (Langfuse) | Basic auth with public/secret key pair |
| Docs | https://hanzo.ai/docs/console |

## One-file quickstart

### Python (Langfuse SDK)

```python
from langfuse import Langfuse

lf = Langfuse(
    host="https://console.hanzo.ai",
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
)

# Create a trace
trace = lf.trace(name="chat-request", user_id="user_123")

# Log a generation (LLM call)
generation = trace.generation(
    name="chat-completion",
    model="zen-70b",
    input=[{"role": "user", "content": "Hello!"}],
    output="Hi there! How can I help?",
    usage={"input": 5, "output": 8, "unit": "TOKENS"},
    metadata={"temperature": 0.7},
)

# Score the trace
trace.score(name="quality", value=0.95, comment="Accurate and concise")

# Flush before exit
lf.flush()
```

### Python (Hanzo SDK decorator)

```python
from hanzo import Hanzo
from hanzo.console import observe

client = Hanzo()  # uses HANZO_API_KEY from env

@observe()
def chat(message: str) -> str:
    """Automatically traced: input, output, latency, cost."""
    response = client.chat.completions.create(
        model="zen-70b",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content

result = chat("What is Hanzo?")
```

### Python (OpenAI integration)

```python
from langfuse.openai import openai

# Drop-in replacement: all OpenAI calls auto-traced
openai.base_url = "https://api.hanzo.ai/v1"
openai.api_key = os.environ["HANZO_API_KEY"]

response = openai.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    langfuse_trace_id="trace_abc",  # optional: link to existing trace
)
# Trace automatically sent to console.hanzo.ai
```

### curl (API direct)

```bash
# Create trace
curl -X POST https://console.hanzo.ai/api/public/traces \
  -H "Content-Type: application/json" \
  -u "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" \
  -d '{
    "name": "api-request",
    "userId": "user_123",
    "input": {"message": "Hello"},
    "output": {"response": "Hi there!"}
  }'

# Get trace
curl https://console.hanzo.ai/api/public/traces/{trace_id} \
  -u "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}"
```

## Endpoint selector

### Traces

| Task | Endpoint | Method |
|------|----------|--------|
| Create trace | `POST /api/public/traces` | POST |
| Get trace | `GET /api/public/traces/{id}` | GET |
| List traces | `GET /api/public/traces` | GET |

### Generations (LLM calls)

| Task | Endpoint | Method |
|------|----------|--------|
| Create generation | `POST /api/public/generations` | POST |
| Update generation | `PATCH /api/public/generations/{id}` | PATCH |
| List generations | `GET /api/public/generations` | GET |

### Scores

| Task | Endpoint | Method |
|------|----------|--------|
| Create score | `POST /api/public/scores` | POST |
| List scores | `GET /api/public/scores` | GET |

### Prompts

| Task | Endpoint | Method |
|------|----------|--------|
| Create prompt | `POST /api/public/prompts` | POST |
| Get prompt | `GET /api/public/prompts/{name}` | GET |
| List prompts | `GET /api/public/prompts` | GET |
| Get prompt version | `GET /api/public/prompts/{name}/{version}` | GET |

### Datasets

| Task | Endpoint | Method |
|------|----------|--------|
| Create dataset | `POST /api/public/datasets` | POST |
| List datasets | `GET /api/public/datasets` | GET |
| Create dataset item | `POST /api/public/dataset-items` | POST |
| Create dataset run | `POST /api/public/dataset-run-items` | POST |

### Sessions

| Task | Endpoint | Method |
|------|----------|--------|
| Get session | `GET /api/public/sessions/{id}` | GET |
| List sessions | `GET /api/public/sessions` | GET |

## Prompt management

Version, test, and deploy prompts from a central registry:

```python
from langfuse import Langfuse

lf = Langfuse(host="https://console.hanzo.ai")

# Get latest production prompt
prompt = lf.get_prompt("chat-system-prompt")
compiled = prompt.compile(user_name="Alice")

# Use in chat completion
response = client.chat.completions.create(
    model="zen-70b",
    messages=[
        {"role": "system", "content": compiled},
        {"role": "user", "content": "Hello!"},
    ],
    langfuse_prompt=prompt,  # links trace to prompt version
)
```

### Prompt lifecycle

1. **Create**: Author prompt in Console dashboard or via API
2. **Version**: Each edit creates a new version (immutable)
3. **Label**: Mark versions as `production`, `staging`, `latest`
4. **Deploy**: SDK fetches by label; no code change needed to update prompts
5. **Evaluate**: A/B test prompt versions against datasets

## Evaluation and datasets

Build evaluation pipelines:

```python
# Create evaluation dataset
dataset = lf.create_dataset(name="chat-eval-v1")

# Add items
lf.create_dataset_item(
    dataset_name="chat-eval-v1",
    input={"messages": [{"role": "user", "content": "What is Hanzo?"}]},
    expected_output="Hanzo is an AI infrastructure company...",
)

# Run evaluation
for item in lf.get_dataset("chat-eval-v1").items:
    response = client.chat.completions.create(
        model="zen-70b",
        messages=item.input["messages"],
    )

    # Score the result
    item.link(
        trace_id=response.langfuse_trace_id,
        run_name="zen-70b-eval-v1",
    )
```

## MCP Integration

Expose observability as MCP tools:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'

const consoleTools: Tool[] = [
  {
    name: 'console_get_trace',
    description: 'Get trace details for debugging an LLM call',
    parameters: {
      trace_id: { type: 'string', required: true }
    },
    async execute({ trace_id }) {
      const res = await fetch(
        `https://console.hanzo.ai/api/public/traces/${trace_id}`,
        { headers: { 'Authorization': `Bearer ${process.env.HANZO_API_KEY}` } }
      )
      return await res.json()
    }
  },
  {
    name: 'console_get_prompt',
    description: 'Fetch a managed prompt by name and label',
    parameters: {
      name: { type: 'string', required: true },
      label: { type: 'string', default: 'production' }
    },
    async execute({ name, label }) {
      const res = await fetch(
        `https://console.hanzo.ai/api/public/prompts/${name}?label=${label}`,
        { headers: { 'Authorization': `Bearer ${process.env.HANZO_API_KEY}` } }
      )
      return await res.json()
    }
  },
  {
    name: 'console_cost_summary',
    description: 'Get cost breakdown by model and time period',
    parameters: {
      period: { type: 'string', enum: ['day', 'week', 'month'], default: 'week' }
    },
    async execute({ period }) {
      const res = await fetch(
        `https://console.hanzo.ai/api/public/metrics/cost?period=${period}`,
        { headers: { 'Authorization': `Bearer ${process.env.HANZO_API_KEY}` } }
      )
      return await res.json()
    }
  }
]
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check trace/generation format |
| 401 | Unauthorized | Check API keys or Langfuse credentials |
| 404 | Not found | Check trace/prompt/dataset ID |
| 429 | Rate limited | SDK auto-retries; manual calls should backoff |
| 500 | Server error | Retry up to 3 times |

## Official links

- Dashboard: https://console.hanzo.ai
- Documentation: https://hanzo.ai/docs/console
- Langfuse SDK: https://github.com/langfuse/langfuse-python
- Hanzo Python SDK: https://github.com/hanzoai/python-sdk
- Hanzo AI: https://hanzo.ai

---

**Last Updated**: 2026-02-26
**Category**: Hanzo Ecosystem
**Related**: observability, tracing, prompts, evaluation
**Prerequisites**: Python, Langfuse SDK or Hanzo SDK
