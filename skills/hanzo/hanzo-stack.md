# Hanzo Stack - Full Integrated Development Environment

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-database.md`

## Overview

Hanzo Stack provides the **complete integrated development environment** for running all Hanzo services locally. One command to start everything: LLM Gateway, Chat, Console, IAM, KMS, PostgreSQL, Redis, MinIO.

## When to use

- Local development across multiple Hanzo services
- Testing integrations between Hanzo components
- Full-stack AI application development
- Onboarding new team members

## Quick reference

| Item | Value |
|------|-------|
| Path | ``github.com/hanzoai/experiments`stack` |
| Setup | `make setup` |
| Start prod | `make up` |
| Start dev | `make dev` |
| Status | `make status` |
| Stop | `make down` |

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| Search UI | 3000 | `http://localhost:3000` |
| Chat UI | 3081 | `http://localhost:3081` |
| LLM Gateway | 4000 | `http://localhost:4000` |
| Payment | 4242 | `http://localhost:4242` |
| Az2 Admin | 5173 | `http://localhost:5173` |
| Core API | 8000 | `http://localhost:8000` |
| PostgreSQL | 5432 | `postgresql://localhost:5432` |
| Redis | 6379 | `redis://localhost:6379` |
| MinIO | 9000 | `http://localhost:9000` |
| Prometheus | 9090 | `http://localhost:9090` |

## One-file quickstart

```bash
cd `github.com/hanzoai/experiments`stack
make setup    # Configure environment (.env)
make dev      # Start all services with hot-reload

# Verify
make status   # Check service health
curl http://localhost:4000/v1/models  # List available models
```

## Related Skills

- `hanzo/hanzo-llm-gateway.md` - LLM proxy (port 4000)
- `hanzo/hanzo-database.md` - PostgreSQL/Redis setup
- `hanzo/hanzo-universe.md` - Production K8s (vs local stack)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: stack, development, local, docker-compose
**Prerequisites**: Docker, make
