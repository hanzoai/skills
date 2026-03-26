<!-- Updated: 2026-03-26T15:03:36Z -->
# Hanzo O11y - Full-Stack Observability Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-k8s.md`, `hanzo/hanzo-deploy.md`

## Overview

Hanzo O11y is a **full-stack observability platform** for logs, metrics, and traces. Fork of SigNoz. Go 1.25 backend (Gin/gorilla-mux) with React 18 + Vite frontend. Uses **ClickHouse** as the telemetry datastore and **OpenTelemetry** for ingestion. Live at `o11y.hanzo.ai`.

**Hanzo O11y is for infrastructure observability (APM, logs, metrics). For LLM-specific observability (traces, costs, prompts), use Hanzo Console.**

## When to use

- Application performance monitoring (APM, errors, latency)
- Centralized log management with full-text search
- Distributed tracing across microservices
- Custom metrics dashboards and alerting
- Infrastructure monitoring (CPU, memory, disk, network)
- Replacing Datadog, New Relic, or Elastic with self-hosted

## Hard requirements

1. **ClickHouse** (`ghcr.io/hanzoai/datastore`) for telemetry storage
2. **ZooKeeper** for ClickHouse coordination
3. **OpenTelemetry Collector** (`ghcr.io/hanzoai/otel-collector`) for data ingestion
4. **Docker or Kubernetes** for deployment

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://o11y.hanzo.ai` |
| Backend | Go 1.25 (Gin, gorilla/mux) |
| Frontend | React 18, TypeScript, Vite, Ant Design |
| Telemetry store | ClickHouse (`ghcr.io/hanzoai/datastore`) |
| Ingestion | OTEL Collector (`ghcr.io/hanzoai/otel-collector`) |
| Metadata | SQLite (community) or PostgreSQL (enterprise) |
| Auth | JWT (built-in), OIDC + SAML (enterprise) |
| Upstream | SigNoz |
| Repo | `github.com/hanzoai/o11y` |
| K8s manifests | `universe/infra/k8s/o11y/` |
| Port (frontend) | 3301 |
| Port (query) | 8080 |
| Port (OTEL gRPC) | 4317 |
| Port (OTEL HTTP) | 4318 |

## Architecture

```
Applications (OTEL SDK instrumented)
        |
   OTEL Collector (ghcr.io/hanzoai/otel-collector)
   port 4317 (gRPC) / 4318 (HTTP)
        |
   +----+----+
   |         |
ClickHouse  O11y Backend
(datastore) (Go, port 8080)
   |              |
   +---------+----+
             |
        O11y Frontend
        (React, port 3301)
```

## Components

| Component | Image | Port | Purpose |
|-----------|-------|------|---------|
| Frontend | `ghcr.io/hanzoai/o11y-frontend` | 3301 | React dashboard |
| Query Service | `ghcr.io/hanzoai/o11y-query` | 8080 | Backend API |
| OTEL Collector | `ghcr.io/hanzoai/otel-collector` | 4317, 4318 | Telemetry ingestion |
| ClickHouse | `ghcr.io/hanzoai/datastore` | 9000, 8123 | Columnar telemetry store |
| ZooKeeper | `bitnami/zookeeper` | 2181 | ClickHouse coordination |

## Instrument your application

### Python (OpenTelemetry)

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure exporter
exporter = OTLPSpanExporter(endpoint="http://otel-collector.hanzo.svc:4317")
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Use tracer
tracer = trace.get_tracer("my-service")
with tracer.start_as_current_span("my-operation"):
    # your code
    pass
```

### Go (OpenTelemetry)

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/sdk/trace"
)

exporter, _ := otlptracegrpc.New(ctx,
    otlptracegrpc.WithEndpoint("otel-collector.hanzo.svc:4317"),
    otlptracegrpc.WithInsecure(),
)
tp := trace.NewTracerProvider(trace.WithBatcher(exporter))
otel.SetTracerProvider(tp)
```

### Node.js (OpenTelemetry)

```typescript
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc'

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://otel-collector.hanzo.svc:4317'
  })
})
sdk.start()
```

## Features

### Traces
- Distributed tracing with Jaeger-compatible UI
- Flame graphs, Gantt charts, service maps
- Trace-to-logs correlation
- Root cause analysis

### Metrics
- Custom dashboards with PromQL-compatible queries
- System metrics (CPU, memory, disk, network)
- Application metrics (request rate, error rate, latency)
- Alerting on any metric condition

### Logs
- Full-text search with attribute filtering
- Log-to-trace correlation
- Live tail
- Log volume analytics

### LLM Observability
- Track LLM calls, token usage, costs
- Prompt/response analysis
- Model performance comparison

## K8s deployment

```yaml
# universe/infra/k8s/o11y/
# Uses kustomize with base configs for all components
cd ~/work/hanzo/universe/infra/k8s/o11y
kubectl --context do-sfo3-hanzo-k8s kustomize . | kubectl apply -f -
```

## Alerting

```yaml
# Example alert rule
- alert: HighErrorRate
  expr: |
    sum(rate(signoz_calls_total{status_code="STATUS_CODE_ERROR"}[5m]))
    / sum(rate(signoz_calls_total[5m])) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Error rate above 5%"
```

## O11y vs Console

| Feature | O11y (SigNoz) | Console (Langfuse) |
|---------|---------------|---------------------|
| Purpose | Infrastructure APM | LLM observability |
| Traces | Distributed service traces | LLM call traces |
| Metrics | System/app metrics | Token usage, costs |
| Logs | Application logs | Prompt/response logs |
| Storage | ClickHouse | PostgreSQL |
| Protocol | OpenTelemetry | Langfuse SDK |

Use O11y for infrastructure. Use Console for LLM-specific observability. Both complement each other.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No traces appearing | OTEL Collector unreachable | Check collector pod and port 4317 |
| ClickHouse OOM | Insufficient memory | Increase ClickHouse memory limits |
| Slow queries | No retention policy | Configure TTL on ClickHouse tables |
| Frontend 502 | Query service down | Check o11y-query pod logs |

## Related Skills

- `hanzo/hanzo-console.md` -- LLM observability (complementary)
- `hanzo/hanzo-k8s.md` -- K8s infrastructure
- `hanzo/hanzo-deploy.md` -- Deployment workflow

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: observability, signoz, opentelemetry, traces, metrics, logs, clickhouse
**Prerequisites**: OTEL concepts, ClickHouse basics
