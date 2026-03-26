# Hanzo Flow

Visual workflow builder for AI applications. Fork of Langflow.

## URLs
- **Landing**: flow.hanzo.ai
- **App**: app.flow.hanzo.ai
- **IAM Client ID**: hanzo-flow
- **Docker Image**: ghcr.io/hanzoai/flow:main

## Tech Stack
- **Runtime**: Python 3.12 (bookworm-slim)
- **Framework**: FastAPI + Typer CLI
- **Package Manager**: uv (workspace with pyproject.toml)
- **Database**: PostgreSQL (via FLOW_DATABASE_URL)

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
uv sync --frozen --no-dev
python -m flow run --host 0.0.0.0 --port 3006
```
- Dockerfile: `python -m flow run` (uv doesn't generate console scripts)
- K8s: deployment/flow, port 7860 (health check on /health)

## CRITICAL: Package Rename Status (2026-03-26)
- Internal package renamed from `langflow` → `flow` (979 files)
- PyPI package: `hanzo-flow` (was `hanzoflow`)
- CLI binary: `flow` (via `[project.scripts] flow = "flow.launcher:main"`)
- **BROKEN**: The rename has cascading import failures:
  - Missing v1/v2 router exports (fixed)
  - Missing frontend static files at flow/frontend
  - Telemetry attribute renames (_get_langflow_desktop → _get_flow_desktop)
  - Permission errors on /app/flow directory
- **Current state**: Scaled to 0 in K8s. Needs proper incremental revert with tests.
- Env vars still use `LANGFLOW_*` prefix (backward compat)

## Upstream
- Fork of: langflow-ai/langflow
- Behind upstream: varies
- openai version conflict: hanzo-llm needs >=2.8.0, other deps cap <2.0.0 (overridden)
