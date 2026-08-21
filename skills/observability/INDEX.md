# Observability Skills

## Category Overview

**Total Skills**: 8
**Category**: observability

## Hanzo Reality (read first)

In the Hanzo stack, observability is NOT a separate deployment: o11y is an
embedded subsystem of the unified `cloud` binary (HIP-0106), with a **native
datastore (ClickHouse) metrics driver** and **OTLP-over-ZAP** — services hold
ZAP-typed telemetry end-to-end and the pb↔zap boundary lives only at the
OTel interop edge (`zap2pb`/`pb2zap`). Third-party OTLP SDKs ingest via the
collector on standard 4317/4318. No ZooKeeper anywhere; datastore runs
keeper-free.

Load: `cat skills/hanzo/hanzo-o11y.md` and
`cat skills/hanzo-cloud-architecture/SKILL.md`

## Skills in This Category

### alerting-strategy.md
**Description**: Setting up production monitoring and on-call rotation

**Load this skill**:
```bash
cat skills/observability/alerting-strategy.md
```

---

### dashboard-design.md
**Description**: Building Grafana dashboards for service monitoring

**Load this skill**:
```bash
cat skills/observability/dashboard-design.md
```

---

### distributed-tracing.md
**Description**: Debugging latency issues in microservices

**Load this skill**:
```bash
cat skills/observability/distributed-tracing.md
```

---

### metrics-instrumentation.md
**Description**: Instrumenting applications for observability

**Load this skill**:
```bash
cat skills/observability/metrics-instrumentation.md
```

---

### observability-cost-optimization.md
**Description**: Optimizing observability costs through sampling, cardinality management, and retention strategies

**Load this skill**:
```bash
cat skills/observability/observability-cost-optimization.md
```

---

### opentelemetry-integration.md
**Description**: Integrating OpenTelemetry for unified observability (traces, metrics, logs)

**Load this skill**:
```bash
cat skills/observability/opentelemetry-integration.md
```

---

### production-incident-debugging.md
**Description**: Debugging production incidents using observability data (logs, metrics, traces)

**Load this skill**:
```bash
cat skills/observability/production-incident-debugging.md
```

---

### structured-logging.md
**Description**: Setting up application logging infrastructure

**Load this skill**:
```bash
cat skills/observability/structured-logging.md
```

---

## Loading All Skills

```bash
# List all skills in this category
ls skills/observability/*.md

# Load specific skills
cat skills/observability/alerting-strategy.md
cat skills/observability/dashboard-design.md
cat skills/observability/distributed-tracing.md
# ... and 5 more
```

## Related Categories

See `skills/README.md` for the complete catalog of all categories and gateway skills.

---

**Browse**: This index provides a quick reference. Load the `observability` gateway skill for common workflows and integration patterns.

```bash
cat skills/observability/SKILL.md
```
