# Hanzo SQL - PostgreSQL with AI Extensions

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kv.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-orm.md`

## Overview

Hanzo SQL is a **PostgreSQL fork** with pre-built extensions for AI workloads: pgvector (vector similarity), pg_cron (scheduled jobs), pg_documentdb (MongoDB wire protocol), and PostGIS (geospatial). Ships as `ghcr.io/hanzoai/sql`. Built on PostgreSQL 18 (Debian bookworm). Default branch: `master`. Port 5432.

### Why Hanzo SQL?

- **pgvector**: Vector similarity search for AI embeddings (IVFFlat, HNSW indexes)
- **pg_documentdb**: MongoDB wire protocol compatibility over PostgreSQL (replaces standalone MongoDB)
- **pg_cron**: Scheduled jobs inside the database
- **PostGIS**: Geospatial data and queries
- **Multi-tenant**: Organization-scoped schemas (`hanzo`, `console`, `iam`)
- **AI-tuned config**: `conf/postgresql.conf` optimized for vector operations and parallel queries

### Tech Stack

- **Language**: C (PostgreSQL core)
- **Build**: Meson (primary) / Make (GNUmakefile.in) / `configure` (autoconf)
- **Image**: `ghcr.io/hanzoai/sql` (multi-stage Dockerfile, Debian bookworm)
- **Base**: PostgreSQL 18

### OSS Base

Repo: `hanzoai/sql` (PostgreSQL fork). The `hanzo/` directory contains Hanzo-specific config, compose file, and init SQL.

## When to use

- Primary OLTP database for any Hanzo service
- Vector embedding storage and similarity search
- MongoDB wire protocol access (via pg_documentdb, paired with FerretDB)
- Multi-tenant application data with schema isolation
- Scheduled database jobs (pg_cron)

## Hard requirements

1. **Port 5432** available
2. **Docker** for container deployment (or full PostgreSQL build toolchain)
3. **Shared preload libraries**: `vector,pg_cron,pg_documentdb_core,pg_documentdb`

## Quick reference

| Item | Value |
|------|-------|
| Port | 5432 |
| Image | `ghcr.io/hanzoai/sql` |
| PG Version | 18 (bookworm) |
| Default DB | `hanzo` |
| Default User | `hanzo` |
| Config | `conf/postgresql.conf` |
| License | PostgreSQL License |
| Repo | `github.com/hanzoai/sql` |
| pgAdmin | port 5050 (via compose) |

## One-file quickstart

### Docker (Hanzo image)

```bash
docker run -d --name hanzo-sql \
  -p 5432:5432 \
  -e POSTGRES_DB=hanzo \
  -e POSTGRES_USER=hanzo \
  -e POSTGRES_PASSWORD=secret \
  ghcr.io/hanzoai/sql
```

### Docker Compose (local dev)

```bash
cd hanzo/
docker compose up -d

# Connect
docker exec -it hanzo-postgres psql -U hanzo -d hanzo
```

The `hanzo/compose.yml` starts PostgreSQL 17 with pgvector and pgAdmin:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: hanzo-postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: hanzo
      POSTGRES_USER: hanzo
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-hanzo_dev}

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: hanzo-pgadmin
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL:-admin@hanzo.ai}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
```

## Core Concepts

### Pre-Installed Extensions

The Docker image builds and installs these extensions:

| Extension | Source | Purpose |
|-----------|--------|---------|
| **pgvector** | `hanzoai/sql-vector` | Vector similarity search |
| **pg_cron** | citusdata/pg_cron v1.6.7 | Scheduled jobs |
| **pg_documentdb** | microsoft/documentdb | MongoDB wire protocol |
| **PostGIS 3** | apt package | Geospatial data |

Init script (`docker-entrypoint-initdb.d/01-extensions.sql`) enables them on startup. DocumentDB extensions are wrapped in exception handlers so PostgreSQL starts even if they fail to load.

### Shared Preload Libraries

```
shared_preload_libraries = 'vector,pg_cron,pg_documentdb_core,pg_documentdb'
```

DocumentDB RUM extension uses graceful loading:
```
documentdb_core.rum_library_load_option = 'try_documentdb_extended_rum'
```

### Multi-Tenant Schema

The `hanzo/init.sql` creates:

- **Schemas**: `hanzo`, `console`, `iam`
- **Roles**: `hanzo_console`, `hanzo_iam`, `hanzo_readonly`
- **Core tables**: `hanzo.organizations`, `hanzo.projects`, `hanzo.embeddings`
- **Vector index**: IVFFlat on `hanzo.embeddings` (1536-dim, cosine similarity)
- **Trigram indexes**: For fuzzy text search on org/project names

### PostgreSQL Configuration

`conf/postgresql.conf` is tuned for AI workloads:

```ini
# Memory - tuned for vector operations
shared_buffers = 512MB
work_mem = 64MB
effective_cache_size = 1536MB

# Parallel queries - leverage multi-core for vector ops
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 16

# Connections
max_connections = 200

# WAL
wal_level = replica
max_wal_size = 2GB
```

### Integration Points

```
hanzo/console  -- AI observability metadata
hanzo/iam      -- User/org data (Casdoor)
hanzo/platform -- PaaS orchestration metadata
hanzo/kms      -- Secret management data
```

Connection string pattern:
```
DATABASE_URL=postgresql://hanzo:password@localhost:5432/hanzo
```

### Directory Structure

```
sql/
  README.md              # Upstream PostgreSQL README
  Dockerfile             # Multi-stage: build extensions + runtime
  Makefile               # PostgreSQL build
  meson.build            # Meson build system
  configure / configure.ac
  conf/
    postgresql.conf      # Hanzo-tuned PostgreSQL config
  docker-entrypoint-initdb.d/
    01-extensions.sql    # Auto-enable extensions on startup
  hanzo/
    LLM.md               # Hanzo-specific documentation
    compose.yml          # Local dev compose (PG + pgAdmin)
    init.sql             # Schema, roles, tables, indexes
  src/                   # PostgreSQL source code
  contrib/               # PostgreSQL contrib modules
  doc/                   # PostgreSQL documentation
  config/                # Build configuration
```

### Syncing with Upstream

```bash
git fetch upstream
git merge upstream/master
git checkout --ours hanzo/   # Preserve Hanzo customizations
git push origin master
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| documentdb extension not loaded | Missing shared_preload_libraries | Ensure `pg_documentdb_core,pg_documentdb` in config |
| Vector index slow | IVFFlat needs more lists | Increase `lists` parameter or switch to HNSW |
| pg_cron not working | Wrong database_name | Set `cron.database_name` to match your DB |
| Connection refused | PG not ready | Wait for healthcheck: `pg_isready -U hanzo -d hanzo` |
| Init SQL failed | Extension .so missing | Check Dockerfile build stage completed |

## Related Skills

- `hanzo/hanzo-kv.md` - Redis-compatible cache
- `hanzo/hanzo-orm.md` - Go ORM with PostgreSQL backend
- `hanzo/hanzo-datastore.md` - ClickHouse (OLAP)
- `hanzo/hanzo-platform.md` - PaaS deployment
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: sql, postgresql, pgvector, documentdb, database, multi-tenant
**Prerequisites**: Docker or PostgreSQL build toolchain
