# Hanzo Observability - Monitoring, Tracing & Error Tracking

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-console.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo Observability covers the full monitoring stack: **OpenTelemetry** for distributed tracing, **Prometheus + Grafana** for metrics, and **Sentry** for error tracking.

## When to use

- Setting up monitoring for Hanzo services
- Configuring distributed tracing
- Error tracking and alerting
- Performance analysis and dashboards

## Components

| Component | Path | Purpose |
|-----------|------|---------|
| O11y Stack | ``github.com/hanzoai/o11y`` | Observability config |
| OTel Collector | ``github.com/hanzoai/otel-collector`` | Trace/metric collection |
| Sentry | ``github.com/hanzoai/sentry`` | Error tracking |

## Quick reference

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |
| OTel Collector | 4317 (gRPC), 4318 (HTTP) | Trace ingestion |

## One-file quickstart

```yaml
# compose.yml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"  # gRPC
      - "4318:4318"  # HTTP
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml

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
```

### Instrument Python service

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer("hanzo-service")

with tracer.start_as_current_span("process_request"):
    # Your code here
    pass
```

## Related Skills

- `hanzo/hanzo-console.md` - LLM-specific observability
- `hanzo/hanzo-universe.md` - Production infrastructure
- `hanzo/hanzo-stack.md` - Local dev stack

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: observability, monitoring, tracing, prometheus, grafana
**Prerequisites**: Docker, metrics concepts
