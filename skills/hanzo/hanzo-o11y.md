# Hanzo Observability - Monitoring, Tracing & Error Tracking

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo Observability covers the full monitoring stack: **OpenTelemetry** for distributed tracing, **Prometheus + Grafana** for metrics and dashboards, and **Sentry** for error tracking. All components run in-cluster on K8s.

### Three Pillars

| Pillar | Tool | Purpose |
|--------|------|---------|
| **Metrics** | Prometheus + Grafana | Time-series metrics, dashboards, alerting |
| **Traces** | OpenTelemetry | Distributed request tracing across services |
| **Errors** | Sentry | Error tracking, stack traces, user context |

## When to use

- Setting up monitoring for Hanzo services
- Configuring distributed tracing across microservices
- Error tracking and alerting
- Performance analysis and dashboards
- Debugging latency and throughput issues

## Quick reference

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics collection + queries |
| Grafana | 3000 | Dashboards + visualization |
| OTel Collector | 4317 (gRPC), 4318 (HTTP) | Trace/metric ingestion |
| Alertmanager | 9093 | Alert routing |

| Item | Value |
|------|-------|
| O11y config | `github.com/hanzoai/o11y` |
| OTel Collector | `github.com/hanzoai/otel-collector` |
| Sentry | `github.com/hanzoai/sentry` |

## OpenTelemetry Instrumentation

### Python (FastAPI)

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
trace.set_tracer_provider(tracer_provider)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Manual spans
tracer = trace.get_tracer("hanzo-service")

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    with tracer.start_as_current_span("chat_completion") as span:
        span.set_attribute("model", request.model)
        span.set_attribute("message_count", len(request.messages))

        result = await inference(request)

        span.set_attribute("tokens", result.usage.total_tokens)
        span.set_attribute("latency_ms", result.latency_ms)
        return result
```

### Go

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func initTracer() func() {
    exporter, _ := otlptracegrpc.New(ctx,
        otlptracegrpc.WithEndpoint("otel-collector:4317"),
        otlptracegrpc.WithInsecure(),
    )
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("hanzo-api"),
        )),
    )
    otel.SetTracerProvider(tp)
    return func() { tp.Shutdown(ctx) }
}

// Usage
tracer := otel.Tracer("hanzo-api")
ctx, span := tracer.Start(ctx, "process_request")
defer span.End()
span.SetAttributes(attribute.String("model", "zen-70b"))
```

### TypeScript (Node.js)

```typescript
import { NodeSDK } from "@opentelemetry/sdk-node"
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc"
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node"

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: "http://otel-collector:4317",
  }),
  instrumentations: [getNodeAutoInstrumentations()],
})
sdk.start()
```

## Prometheus Metrics

### Instrument Service

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
REQUEST_COUNT = Counter(
    "hanzo_requests_total",
    "Total requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "hanzo_request_duration_seconds",
    "Request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
TOKENS_PROCESSED = Counter(
    "hanzo_tokens_total",
    "Tokens processed",
    ["model", "type"],  # type: input/output
)

# Expose metrics endpoint
start_http_server(8000)

# Use in handlers
@app.post("/v1/chat/completions")
async def chat(request):
    with REQUEST_LATENCY.labels("POST", "/v1/chat/completions").time():
        result = await process(request)
        REQUEST_COUNT.labels("POST", "/v1/chat/completions", 200).inc()
        TOKENS_PROCESSED.labels(request.model, "input").inc(result.usage.prompt_tokens)
        TOKENS_PROCESSED.labels(request.model, "output").inc(result.usage.completion_tokens)
        return result
```

### Prometheus Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "hanzo-services"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: ${1}:${2}
```

## Grafana Dashboards

### Key Dashboards

| Dashboard | Metrics | Use |
|-----------|---------|-----|
| LLM Gateway | Request rate, latency, tokens/sec, errors | Monitor AI traffic |
| Service Health | CPU, memory, restarts, ready replicas | Infrastructure |
| Database | Query latency, connections, cache hit rate | PostgreSQL/Redis |
| Cost Tracking | Tokens × model price, by customer | Billing |

### Example PromQL

```promql
# Request rate (5 min window)
rate(hanzo_requests_total[5m])

# P99 latency
histogram_quantile(0.99, rate(hanzo_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(hanzo_requests_total{status=~"5.."}[5m]))
/ sum(rate(hanzo_requests_total[5m]))

# Tokens per second by model
rate(hanzo_tokens_total[1m])
```

## Local Setup (compose.yml)

```yaml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"   # gRPC
      - "4318:4318"   # HTTP
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml
    command: ["--config=/etc/otel/config.yaml"]

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "${GRAFANA_PASSWORD}"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana_data:
```

## Sentry Integration

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,  # 10% of requests
    environment="production",
)
```

## Related Skills

- `hanzo/hanzo-console.md` - LLM-specific observability (traces, costs, scores)
- `hanzo/hanzo-universe.md` - Production infrastructure
- `hanzo/hanzo-stack.md` - Local dev stack (Prometheus included)
- `hanzo/hanzo-insights.md` - Product analytics (different from infra o11y)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: observability, monitoring, tracing, prometheus, grafana, opentelemetry
**Prerequisites**: Docker, metrics concepts, distributed systems
