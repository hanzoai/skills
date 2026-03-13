# Hanzo Database - PostgreSQL, Redis & Vector Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-orm.md`, `hanzo/hanzo-datastore.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo Database covers PostgreSQL (with pgvector), Redis, and MongoDB configurations used across the Hanzo ecosystem. All databases run in-cluster on K8s (no managed DBs).

## When to use

- Configuring PostgreSQL for Hanzo services
- Setting up pgvector for AI embeddings
- Redis caching and session storage
- Database troubleshooting

## Quick reference

| Database | Port | Use Case |
|----------|------|----------|
| PostgreSQL | 5432 | Primary RDBMS, pgvector |
| Redis | 6379 | Cache, sessions, queues |
| MongoDB | 27017 | Document storage |
| MinIO | 9000 | Object storage (S3-compatible) |

## Production Databases

| Cluster | Host | Databases |
|---------|------|-----------|
| hanzo-k8s | `postgres.hanzo.svc` | iam, cloud, console, hanzo_cloud, kms, platform |
| lux-k8s | `postgres.hanzo.svc` | cloud, commerce, console, gateway, hanzo, kms |

## One-file quickstart

```bash
# Connection (from within K8s)
DATABASE_URL=postgresql://user:pass@postgres.hanzo.svc:5432/mydb
REDIS_URL=redis://redis.hanzo.svc:6379

# Enable pgvector
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create embeddings table
psql $DATABASE_URL -c "
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB
);
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"
```

### Python pgvector usage

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)

# Insert embedding
cur = conn.cursor()
cur.execute("INSERT INTO embeddings (content, embedding) VALUES (%s, %s)",
            ("Hello world", embedding_vector))

# Similarity search
cur.execute("SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT 5",
            (query_vector,))
```

## Troubleshooting

```bash
# Port conflicts
lsof -i :5432 && kill -9 PID

# Reset (destructive)
docker compose down -v && docker system prune -a
```

## Related Skills

- `hanzo/hanzo-orm.md` - Go ORM for database access
- `hanzo/hanzo-datastore.md` - Vector database
- `hanzo/hanzo-stack.md` - Local stack with all DBs

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: postgresql, redis, pgvector, database
**Prerequisites**: SQL, Redis basics
