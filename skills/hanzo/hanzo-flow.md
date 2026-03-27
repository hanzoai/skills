# Hanzo Flow

Visual workflow builder for AI applications. Fork of Langflow.

## URLs
- **Landing**: flow.hanzo.ai
- **App**: app.flow.hanzo.ai
- **IAM Client ID**: hanzo-flow
- **Docker Image**: ghcr.io/hanzoai/flow:main

## Tech Stack
- **Runtime**: Python 3.12 (bookworm-slim builder, trixie-slim runtime)
- **Framework**: FastAPI + Typer CLI
- **Package Manager**: uv (workspace with pyproject.toml)
- **Database**: PostgreSQL (via FLOW_DATABASE_URL, requires `--extra postgresql` for psycopg driver)

## Workspace Structure
- `src/backend/base/` — core flow package (langflow-base on PyPI)
- `src/backend/flow/` — root hanzoflow package
- `src/lfx/` — LFX extensions
- `pyproject.toml` — root workspace config
- `uv.lock` — locked dependencies

## Key Config
- `FLOW_SSO_ENABLED=true`, `FLOW_SSO_PROVIDER=oidc`, `FLOW_OIDC_ISSUER_URL=https://hanzo.id`
- `LANGFLOW_CONFIG_DIR=/app/data` (writable dir for settings/SQLite)
- `LANGFLOW_BACKEND_ONLY=true` (API-only mode, no embedded frontend)

## Build & Deploy
```bash
uv sync --frozen --no-dev --extra postgresql
python -m flow run --host 0.0.0.0 --port 7860 --backend-only
```
- Dockerfile CMD: `python -m flow run --host 0.0.0.0 --port 7860 --backend-only`
- Docker build uses `--extra postgresql` (psycopg driver required for production PG)
- K8s: deployment/flow, service/flow, port 7860
- Health check: `GET /health` on port 7860
- Both flow.hanzo.ai and app.flow.hanzo.ai route to the flow service

## Package Rename Status (2026-03-27)
- Internal package renamed from `langflow` → `flow` (979 files)
- PyPI package: `hanzo-flow` (was `hanzoflow`)
- CLI binary: `flow` (via `[project.scripts] flow = "flow.launcher:main"`)
- Dockerfile now uses `python -m flow run` (working as of 2026-03-27)
- Telemetry renamed: `_get_langflow_desktop` → `_get_flow_desktop`
- Env vars still use `LANGFLOW_*` prefix (backward compat)
- LANGFLOW_SSO_ENABLED, LANGFLOW_OIDC_* used for SSO config

## Upstream
- Fork of: langflow-ai/langflow
- Behind upstream: varies
- openai version conflict: hanzo-llm needs >=2.8.0, other deps cap <2.0.0 (overridden)
