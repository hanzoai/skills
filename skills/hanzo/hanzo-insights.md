# Hanzo Insights - Product Analytics Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-o11y.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Insights is a **full product analytics platform** -- a PostHog fork with product analytics, feature flags, session recording, A/B testing, heatmaps, LLM analytics, error tracking, surveys, web analytics, and a custom query language (InsightsQL). Polyglot monorepo: Django/Python backend, React/TypeScript frontend, Rust high-performance services, Go livestream server. ClickHouse for event storage, PostgreSQL for metadata, Kafka for event streaming, Redis for caching. Self-hostable with Docker Compose or K8s.

### Why Hanzo Insights?

- **PostHog fork**: Full product analytics suite, rebranded under `@hanzo/` namespace
- **40+ product modules**: analytics, feature flags, experiments, session replay, error tracking, LLM analytics, surveys, web analytics, notebooks, workflows, data warehouse, CDP
- **InsightsQL**: Custom SQL-like query language (ANTLR grammar, Python + C++ parsers)
- **Multi-language SDKs**: JavaScript (`@hanzo/insights`), Node.js (`@hanzo/insights-node`), Python (`hanzo_insights`), Go (`insights-go`), Rust (`insights-rs`)
- **Rust services**: High-performance capture, feature-flag evaluation, webhook delivery, error symbolication (cymbal), Kafka dedup, person resolution
- **MCP server**: Model Context Protocol integration for AI agent access to analytics data
- **CLI**: Rust CLI (`insights-cli`) for queries, sourcemap uploads, endpoint management

### Tech Stack

- **Backend**: Django 4.2 + DRF on Python 3.12 (Granian ASGI server)
- **Frontend**: React 18 + TypeScript + Kea state management + Vite + Tailwind
- **Rust services**: axum, rdkafka, sqlx, clickhouse-rs (workspace with 30+ crates)
- **Livestream**: Go service for real-time event streaming
- **Query engine**: InsightsQL (ANTLR4 grammar -> Python3 + C++ targets)
- **Databases**: ClickHouse (events), PostgreSQL (metadata), Kafka (streaming), Redis (cache)
- **Task queue**: Celery + Temporal (batch exports, async workflows)
- **Data pipelines**: Dagster (orchestration), dlt (data loading)
- **Build**: pnpm workspace + Turborepo (JS), uv (Python), Cargo workspace (Rust)
- **Package manager**: pnpm 10.x, Node 24, Python 3.12.12, uv 0.10.x

### OSS Base

Repo: `hanzoai/insights`. License: MIT.

## When to use

- Product analytics and event tracking for any application
- Feature flag management and gradual rollouts
- A/B testing / experimentation with statistical analysis
- Session recording and user behavior replay
- LLM/AI application observability and analytics
- Error tracking with sourcemap support
- Web analytics (privacy-first alternative to GA)
- User surveys and feedback collection
- Data warehouse querying (Snowflake, BigQuery, Postgres, S3)
- Self-hosted analytics with full data ownership

## Hard requirements

1. **ClickHouse** for event storage (columnar analytics)
2. **PostgreSQL** for metadata and Django ORM
3. **Kafka** (Redpanda) for event streaming pipeline
4. **Redis** for caching and task queues
5. **Object storage** (MinIO/S3/SeaweedFS) for session recordings and batch exports

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/insights` |
| Upstream | PostHog fork |
| License | MIT |
| Dashboard | `https://insights.hanzo.ai` |
| API | `https://insights.hanzo.ai/api/` |
| CLI host | `https://insights.hanzo.ai` |
| Python | 3.12.12 |
| Node | >=24 <25 |
| pnpm | 10.29.3 |
| uv | ~0.10.2 |

### SDKs

| Language | Package | Repo |
|----------|---------|------|
| JavaScript (browser) | `@hanzo/insights` v6.0.0 | in-repo `common/insights-js` |
| JavaScript (lite) | `@hanzo/insights-lite` | in-repo `common/insights-js-lite` |
| Node.js | `@hanzo/insights-node` v6.0.0 | in-repo `common/insights-node` |
| Python | `hanzo_insights` | `hanzoai/insights-python` |
| Go | `github.com/hanzoai/insights-go` | `hanzoai/insights-go` |
| Rust | `insights-rs` | `hanzoai/insights-rs` |

## Monorepo structure

```
insights/
  insights/           # Django app (Python backend)
    api/              # REST API views (DRF)
    insightsql/       # Query language (ANTLR grammar + interpreter)
    models/           # Django models
    clickhouse/       # ClickHouse query builders
    session_recordings/
    heatmaps/
    batch_exports/
    cdp/              # Customer Data Platform
    llm/              # LLM integration helpers
    warehouse/        # Data warehouse connectors
    tasks/            # Celery tasks
    temporal/         # Temporal workflow definitions
  frontend/           # React + TypeScript (Vite, Kea, Tailwind)
    src/
    @hanzo/           # Shared frontend packages
  products/           # Product modules (each has backend/ + frontend/)
    product_analytics/
    feature_flags/
    experiments/
    replay/             # Session recording
    error_tracking/
    llm_analytics/      # LLM observability (Dockerfile.llm-analytics)
    web_analytics/
    surveys/
    cohorts/
    dashboards/
    notebooks/
    workflows/
    data_warehouse/
    marketing_analytics/
    revenue_analytics/
    customer_analytics/
    cdp/
    ...40+ total
  rust/               # Rust workspace (30+ crates)
    capture/          # High-performance event capture (axum)
    feature-flags/    # Rust feature flag evaluator
    hook-worker/      # Webhook delivery
    hook-api/         # Webhook API
    cymbal/           # Error symbolication
    cyclotron-core/   # Job scheduler
    embedding-worker/ # Embedding generation
    kafka-deduplicator/
    personinsights-*/  # Person resolution services
    property-defs-rs/
    common/           # Shared Rust libs (kafka, redis, metrics, health, etc.)
  livestream/         # Go service (real-time event streaming)
  cli/                # Rust CLI (insights-cli)
  funnel-udf/         # Rust ClickHouse UDF for funnels
  common/             # Shared packages
    insights-js/      # Browser SDK wrapper (@hanzo/insights)
    insights-js-lite/ # Lightweight browser SDK
    insights-node/    # Node.js SDK wrapper (@hanzo/insights-node)
    insightsql_parser/  # InsightsQL parser (Python package)
    insightscli/      # Internal dev CLI tooling
    design-system/    # Shared UI components
    tailwind/         # Tailwind config
    storybook/
    siphash/          # SipHash implementation
    scriptvm/         # Script VM (TypeScript + Rust)
    ingestion/        # Shared ingestion code
  services/
    mcp/              # MCP server (Cloudflare Worker, TypeScript)
    llm-gateway/      # LLM gateway integration
  nodejs/             # Node.js plugin server
  docs/               # Internal documentation
  docker/             # Docker configs (ClickHouse, Caddy, Temporal, etc.)
  proto/              # Protobuf definitions
  terraform/          # Infrastructure as code
  playwright/         # E2E tests
```

## Development commands

```bash
# Python backend
uv sync --all-extras          # Install Python deps
pytest                        # Run all backend tests
pytest path/to/test.py::TestClass::test_method  # Single test
ruff check . --fix && ruff format .   # Lint + format Python
python manage.py migrate      # Run Django migrations

# Frontend
pnpm install                  # Install JS deps
pnpm --filter=@hanzo/frontend build   # Build frontend
pnpm --filter=@hanzo/frontend test    # Run frontend tests
pnpm --filter=@hanzo/frontend format  # Format frontend

# Full stack
./bin/start                   # Start dev server (backend + frontend)

# InsightsQL grammar
pnpm grammar:build:python     # Rebuild ANTLR Python parser
pnpm grammar:build:cpp        # Rebuild ANTLR C++ parser

# Schema / OpenAPI
pnpm schema:build             # Build TypeScript schema from Django serializers
pnpm openapi:build            # Generate OpenAPI spec + TS types via Orval

# Rust services
cd rust && cargo build        # Build all Rust services
cd rust && cargo test         # Test all Rust services

# CLI
cd cli && cargo build         # Build insights-cli

# Docker (self-host)
docker compose -f docker-compose.hobby.yml up  # Full self-hosted stack
```

## Self-hosting

The `docker-compose.hobby.yml` runs the complete stack:

- `web`: Django app (Granian ASGI)
- `worker`: Celery worker
- `plugins`: Node.js plugin server
- `db`: PostgreSQL (`ghcr.io/hanzoai/sql`)
- `kv`: Redis
- `datastore`: ClickHouse
- `kafka`: Redpanda
- `capture`: Rust event capture
- `feature-flags`: Rust flag evaluator
- `property-defs-rs`: Rust property definitions
- `livestream`: Go real-time events
- `cymbal`: Rust error symbolication
- `objectstorage`: MinIO
- `seaweedfs`: Session recording storage
- `temporal`: Workflow orchestration
- `proxy`: Caddy reverse proxy

```bash
# Minimal self-host
export DOMAIN=insights.example.com
export INSIGHTS_SECRET=$(openssl rand -hex 32)
docker compose -f docker-compose.hobby.yml up -d
```

## InsightsQL

Custom query language with ANTLR4 grammar. Two parser targets: Python3 (backend queries) and C++ (ClickHouse UDFs). Supports:

- Event queries with property filters
- Funnel analysis
- Retention cohorts
- Path analysis
- Aggregations, breakdowns, sampling

Security: never interpolate user data into InsightsQL f-strings. Use `ast.Constant()` placeholders or pass entire expressions through the parser.

## CLI

```bash
insights-cli login                    # Authenticate interactively
insights-cli query "SELECT count() FROM events"  # Run InsightsQL query
insights-cli sourcemap upload ./dist  # Upload sourcemaps for error tracking
insights-cli exp endpoints list       # List data endpoints
```

Environment variables: `INSIGHTS_CLI_HOST`, `INSIGHTS_CLI_API_KEY`, `INSIGHTS_CLI_PROJECT_ID`.

## Architecture guidelines

- API views declare request/response schemas via `@validated_request` or `@extend_schema`
- Django serializers are source of truth for frontend types (auto-generated via drf-spectacular + Orval)
- New features go in `products/` directory (each has `backend/`, `frontend/`, `manifest.tsx`)
- Always filter querysets by `team_id`
- Do not add domain-specific fields to `Team` model -- use Team Extension pattern
- Frontend state management: Kea (not React hooks for business logic)
- Conventional commits: `feat(scope):`, `fix(scope):`, `chore(scope):`

## Related Skills

- `hanzo/hanzo-o11y.md` - Technical observability (metrics, traces -- different from product analytics)
- `hanzo/hanzo-console.md` - LLM-specific observability
- `hanzo/hanzo-cloud.md` - Dashboard with analytics views

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: analytics, posthog, feature-flags, ab-testing, session-recording, clickhouse, insightsql, llm-analytics, error-tracking
**Prerequisites**: Python, TypeScript, Docker, analytics concepts
