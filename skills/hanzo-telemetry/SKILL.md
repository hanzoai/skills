---
name: hanzo-telemetry
description: "OpenTelemetry-based observability with distributed tracing, metrics, and logs. Instrument applications with OTLP, Prometheus, and Zipkin exporters."
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
              "package": "opentelemetry-api",
              "label": "Install OpenTelemetry (pip)",
            },
          ],
      },
  }
---

# Hanzo Telemetry — Observability

`pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`

OpenTelemetry-based distributed tracing, metrics, and logs for Hanzo services.

## Quick Start

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Set up tracing
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-service")

# Create spans
with tracer.start_as_current_span("process_request") as span:
    span.set_attribute("user.id", "123")
    result = do_work()
    span.set_attribute("result.status", "success")
```

## Metrics

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

provider = MeterProvider()
metrics.set_meter_provider(provider)

meter = metrics.get_meter("my-service")

# Counter
request_counter = meter.create_counter("requests_total")
request_counter.add(1, {"method": "GET", "path": "/api/chat"})

# Histogram
latency = meter.create_histogram("request_duration_ms")
latency.record(42.5, {"method": "POST"})

# Gauge
active_connections = meter.create_up_down_counter("active_connections")
active_connections.add(1)
```

## Automatic Instrumentation

```python
# Auto-instrument popular frameworks
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-instrumentation-requests
pip install opentelemetry-instrumentation-redis

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

## Logging

```python
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

provider = LoggerProvider()
provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
set_logger_provider(provider)
```

## Exporters

| Exporter   | Protocol      | Port |
| ---------- | ------------- | ---- |
| OTLP gRPC  | gRPC          | 4317 |
| OTLP HTTP  | HTTP/protobuf | 4318 |
| Prometheus | HTTP scrape   | 8888 |
| Zipkin     | HTTP          | 9411 |

## Environment Variables

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=my-service
OTEL_RESOURCE_ATTRIBUTES=environment=production
```
