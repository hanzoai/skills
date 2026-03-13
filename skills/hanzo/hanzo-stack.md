# Hanzo Stack - Full Integrated Development Environment

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-database.md`

## Overview

Hanzo Stack provides the **complete integrated development environment** for running all Hanzo services locally. One command to start everything: LLM Gateway, Chat, Console, Cloud, IAM, KMS, PostgreSQL, Redis, MongoDB, MinIO.

### Why Hanzo Stack?

- **One-command setup**: `make dev` starts everything
- **Full ecosystem**: All Hanzo services running locally
- **Hot-reload**: Development mode with file watching
- **Consistent**: Same compose config for all developers
- **Offline-capable**: Local models via LLM Gateway + Ollama

## When to use

- Local development across multiple Hanzo services
- Testing integrations between Hanzo components
- Full-stack AI application development
- Onboarding new team members
- Reproducing production issues locally

## Hard requirements

1. **Docker** with compose v2
2. **16GB+ RAM** (recommended for full stack)
3. **make** (standard on macOS/Linux)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai` (experiments/stack directory) |
| Setup | `make setup` |
| Start prod | `make up` |
| Start dev | `make dev` |
| Status | `make status` |
| Stop | `make down` |
| Logs | `make logs` |

## Service Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Search UI | 3000 | `http://localhost:3000` | AI-powered search |
| Chat UI | 3081 | `http://localhost:3081` | Hanzo Chat (LibreChat) |
| LLM Gateway | 4000 | `http://localhost:4000` | Unified LLM proxy |
| Payment API | 4242 | `http://localhost:4242` | Commerce API |
| Admin UI | 5173 | `http://localhost:5173` | Admin dashboard |
| Core API | 8000 | `http://localhost:8000` | Main API server |
| PostgreSQL | 5432 | `postgresql://localhost:5432` | Primary database |
| Redis | 6379 | `redis://localhost:6379` | Cache/queues |
| MongoDB | 27017 | `mongodb://localhost:27017` | Document storage |
| MinIO | 9000 | `http://localhost:9000` | S3-compatible storage |
| Prometheus | 9090 | `http://localhost:9090` | Metrics collection |

## One-file quickstart

```bash
# Clone and setup
git clone https://github.com/hanzoai/experiments.git
cd experiments/stack

# Configure environment
make setup    # Creates .env from .env.example, prompts for API keys

# Start all services
make dev      # Development mode (hot-reload)

# Verify
make status   # Check all services are healthy
curl http://localhost:4000/v1/models  # List available models
curl http://localhost:3081            # Open Chat UI
```

## Makefile Commands

```bash
make setup        # Initial setup (env, pull images)
make up           # Start all services (production mode)
make dev          # Start with hot-reload (development)
make down         # Stop all services
make restart      # Restart all services
make status       # Show service status and health
make logs         # Stream all logs
make logs-llm     # Stream LLM Gateway logs only
make logs-chat    # Stream Chat logs only
make clean        # Remove volumes and data
make pull         # Pull latest images
make build        # Build local images
make test         # Run integration tests
make reset        # Full reset (clean + setup)
```

## Environment Configuration

```bash
# .env (created by make setup)

# Required: At least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HANZO_API_KEY=hanzo-...

# Optional: Local models
OLLAMA_URL=http://host.docker.internal:11434

# Database
DATABASE_URL=postgresql://hanzo:hanzo@postgres:5432/hanzo
REDIS_URL=redis://redis:6379

# IAM
HANZO_IAM_URL=https://hanzo.id
HANZO_IAM_CLIENT_ID=app-hanzo
HANZO_IAM_CLIENT_SECRET=secret

# KMS
KMS_ENDPOINT=https://kms.hanzo.ai
KMS_CLIENT_ID=...
KMS_CLIENT_SECRET=...
```

## Compose Architecture

```
┌─────────────────────────────────────────┐
│              compose.yml                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │ Chat UI  │  │ Search   │  │ Admin │ │
│  │  :3081   │  │  :3000   │  │ :5173 │ │
│  └────┬─────┘  └────┬─────┘  └───┬───┘ │
│       │              │            │     │
│  ┌────┴──────────────┴────────────┴───┐ │
│  │        LLM Gateway :4000          │ │
│  └────────────────┬───────────────────┘ │
│                   │                     │
│  ┌────────────────┴───────────────────┐ │
│  │          Core API :8000            │ │
│  └──┬─────────┬──────────┬───────────┘ │
│     │         │          │             │
│  ┌──┴───┐ ┌──┴───┐ ┌───┴────┐ ┌─────┐│
│  │Postgres│ │Redis │ │MongoDB │ │MinIO││
│  │ :5432 │ │:6379 │ │ :27017 │ │:9000││
│  └───────┘ └──────┘ └────────┘ └─────┘│
└─────────────────────────────────────────┘
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port conflict | Service already running | `lsof -i :PORT && kill PID` |
| OOM | Too many services | Start subset: `make up SERVICES="llm chat postgres redis"` |
| Slow start | Image pull | `make pull` beforehand |
| DB connection fail | Postgres not ready | Wait or `make restart` |
| Chat not loading | Missing OPENAI_API_KEY | Add at least one LLM key to .env |

```bash
# Debug specific service
docker compose logs -f llm-gateway
docker compose exec postgres psql -U hanzo

# Reset everything
make clean && make setup && make dev

# Port conflicts
lsof -i :4000   # Find what's using the port
```

## Related Skills

- `hanzo/hanzo-llm-gateway.md` - LLM proxy (port 4000)
- `hanzo/hanzo-chat.md` - Chat UI (port 3081)
- `hanzo/hanzo-database.md` - PostgreSQL/Redis setup
- `hanzo/hanzo-o11y.md` - Monitoring (Prometheus port 9090)
- `hanzo/hanzo-universe.md` - Production K8s (vs local stack)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: stack, development, local, docker, compose
**Prerequisites**: Docker, make, 16GB+ RAM
