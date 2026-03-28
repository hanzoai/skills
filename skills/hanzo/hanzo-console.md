# Hanzo Console - AI Observability and Prompt Management

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-chat.md`, `hanzo/hanzo-cloud.md`, `hanzo/python-sdk.md`

## Overview

Hanzo Console is the **observability and prompt management layer** for AI applications. Fork of Langfuse. Captures traces, scores, datasets, and prompts from any LLM application. Provides a dashboard for debugging, evaluation, and cost analysis. Compatible with the Langfuse SDK and OpenTelemetry. Live at `console.hanzo.ai`.

## When to use

- Tracing and debugging LLM calls, tool use, and agent steps
- Cost attribution per-request, per-user, per-model
- Managing prompt versions and deployments
- Building evaluation datasets and running automated scoring
- Monitoring AI application performance and latency

## Hard requirements

1. **API Key required**: Use `HANZO_API_KEY` or Langfuse pair `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`
2. **Never expose keys** in user-visible output, logs, or screenshots
3. **Traces are async**: SDK batches in background. Call `flush()` before process exit
4. **PostgreSQL backend**: Console uses `console` database on `postgres.hanzo.svc`

## Quick reference

| Item | Value |
|------|-------|
| Dashboard | `https://console.hanzo.ai` |
| API Base | `https://console.hanzo.ai/api` |
| Langfuse host | `https://console.hanzo.ai` |
| Ingestion API | `https://console.hanzo.ai/api/public/ingestion` |
| Auth (Hanzo) | `Authorization: Bearer ${HANZO_API_KEY}` |
| Auth (Langfuse) | Basic auth with public/secret key pair |
| Upstream | Langfuse |
| Repo | `github.com/hanzoai/console` |
| K8s manifests | `universe/infra/k8s/console/` |
| Image | `ghcr.io/hanzoai/console:main` |
| Worker Image | `ghcr.io/hanzoai/console-worker:main` |
| Port | 3000 (web), 3030 (worker) |
| IAM Client ID | hanzo-console |
| Worker health | `/api/health` (NOT `/api/public/health`) |

### Recent Changes (2026-03-28)
- Tracking embed + product keys section added to console settings
- Worker readiness/liveness probes fixed: path `/api/public/health` → `/api/health`
- IAM OAuth provider registered (env: `IAM_CLIENT_ID`, `IAM_CLIENT_SECRET`, `IAM_SERVER_URL`)

## Architecture

```
Your Application
      |
  Langfuse SDK / OTEL
      |
console.hanzo.ai/api/public/ingestion
      |
Console Backend (Node.js)
      |
  +---+---+
  |       |
PostgreSQL  ClickHouse
(metadata)  (traces, optional)
```

Cloud API (`cloud.hanzo.ai`) POSTs traces to Console's ingestion endpoint for centralized observability.

## Python quickstart (Langfuse SDK)

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

## Python quickstart (Hanzo SDK decorator)

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
```

## Python quickstart (OpenAI drop-in)

```python
from langfuse.openai import openai

# Drop-in replacement: all OpenAI calls auto-traced
openai.base_url = "https://api.hanzo.ai/v1"
openai.api_key = os.environ["HANZO_API_KEY"]

response = openai.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Hello!"}],
)
# Trace automatically sent to console.hanzo.ai
```

## API reference

### Traces

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/public/traces` | Create trace |
| GET | `/api/public/traces/{id}` | Get trace |
| GET | `/api/public/traces` | List traces |

### Generations (LLM calls)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/public/generations` | Create generation |
| PATCH | `/api/public/generations/{id}` | Update generation |
| GET | `/api/public/generations` | List generations |

### Scores

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/public/scores` | Create score |
| GET | `/api/public/scores` | List scores |

### Prompts

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/public/prompts` | Create prompt |
| GET | `/api/public/prompts/{name}` | Get prompt (latest) |
| GET | `/api/public/prompts/{name}/{version}` | Get prompt version |
| GET | `/api/public/prompts` | List prompts |

### Datasets

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/public/datasets` | Create dataset |
| GET | `/api/public/datasets` | List datasets |
| POST | `/api/public/dataset-items` | Create dataset item |
| POST | `/api/public/dataset-run-items` | Create dataset run |

## Prompt management

Version, test, and deploy prompts from a central registry:

```python
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
2. **Version**: Each edit creates a new immutable version
3. **Label**: Mark versions as `production`, `staging`, `latest`
4. **Deploy**: SDK fetches by label; no code change needed
5. **Evaluate**: A/B test prompt versions against datasets

## Multi-org bootstrap

Console supports multi-org provisioning on startup:

```bash
HANZO_INIT_ORG_IDS=hanzo,lux,zoo,pars
HANZO_INIT_ORG_NAMES="Hanzo AI,Lux Network,Zoo Foundation,Pars"
HANZO_INIT_USER_EMAIL=z@hanzo.ai
HANZO_INIT_PROJECT_ORG_ID=hanzo
```

## K8s deployment

```yaml
# universe/infra/k8s/console/
apiVersion: apps/v1
kind: Deployment
metadata:
  name: console
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: console
          image: ghcr.io/hanzoai/console:latest
          ports:
            - containerPort: 3000
          envFrom:
            - secretRef:
                name: console-secrets  # KMS-synced
          env:
            - name: DATABASE_URL
              value: postgresql://console:pass@postgres.hanzo.svc:5432/console
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check trace/generation format |
| 401 | Unauthorized | Check API keys or Langfuse credentials |
| 404 | Not found | Check trace/prompt/dataset ID |
| 429 | Rate limited | SDK auto-retries; manual calls should back off |
| 500 | Server error | Retry up to 3 times |

## Related Skills

- `hanzo/hanzo-cloud.md` -- Cloud API (sends traces to Console)
- `hanzo/hanzo-chat.md` -- Chat app (auto-instrumented)
- `hanzo/hanzo-o11y.md` -- Infrastructure observability (SigNoz)
- `hanzo/python-sdk.md` -- Python SDK with `@observe()` decorator

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: observability, tracing, prompts, evaluation, langfuse, cost-tracking
**Prerequisites**: Python or Node.js, Langfuse SDK or Hanzo SDK
