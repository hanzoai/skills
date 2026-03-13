# Hanzo Database - PostgreSQL, Redis & Vector Storage

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-orm.md`, `hanzo/hanzo-datastore.md`, `hanzo/hanzo-stack.md`

## Overview

Hanzo Database covers PostgreSQL (with pgvector), Redis, MongoDB, and MinIO configurations used across the Hanzo ecosystem. **All databases run in-cluster on K8s** — no managed database services (DO Managed DB decommissioned Feb 2026).

### Why In-Cluster?

- **Cost**: 5-10x cheaper than managed DBs at scale
- **Latency**: Same-cluster networking, no external hops
- **Control**: Full PostgreSQL config, extensions, versions
- **Consistency**: Same setup across all environments

## When to use

- Configuring PostgreSQL for Hanzo services
- Setting up pgvector for AI embeddings
- Redis caching, sessions, and job queues
- MongoDB for document storage (Chat)
- Database troubleshooting and maintenance

## Quick reference

| Database | Port | Image | Use Case |
|----------|------|-------|----------|
| PostgreSQL | 5432 | `postgres:16` | Primary RDBMS, pgvector |
| Redis | 6379 | `redis:7-alpine` | Cache, sessions, BullMQ queues |
| MongoDB | 27017 | `mongo:7` | Document storage (Chat) |
| MinIO | 9000 | `minio/minio` | S3-compatible object storage |

## Production Layout

### hanzo-k8s Cluster

| Service | Database | Host |
|---------|----------|------|
| `postgres.hanzo.svc` | — | Shared PostgreSQL instance |
| ↳ | `iam` | Casdoor (hanzo.id) |
| ↳ | `cloud` | Cloud dashboard |
| ↳ | `console` | Observability |
| ↳ | `hanzo_cloud` | Cloud API |
| ↳ | `kms` | KMS/Infisical |
| ↳ | `platform` | PaaS platform |

### lux-k8s Cluster

| Service | Database | Host |
|---------|----------|------|
| `postgres.hanzo.svc` | — | Shared PostgreSQL instance |
| ↳ | `cloud` | Lux Cloud |
| ↳ | `commerce` | Commerce API |
| ↳ | `console` | Console |
| ↳ | `gateway` | API Gateway |
| ↳ | `hanzo` | Core Hanzo |
| ↳ | `kms` | KMS |

## PostgreSQL + pgvector

### Setup

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),       -- OpenAI ada-002 dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index (fast approximate search)
CREATE INDEX ON embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- HNSW index (better recall, more memory)
CREATE INDEX ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Similarity Search

```sql
-- Cosine similarity (most common for embeddings)
SELECT content, 1 - (embedding <=> $1::vector) AS similarity
FROM embeddings
ORDER BY embedding <=> $1::vector
LIMIT 5;

-- L2 distance
SELECT content, embedding <-> $1::vector AS distance
FROM embeddings
ORDER BY embedding <-> $1::vector
LIMIT 5;

-- Inner product
SELECT content, embedding <#> $1::vector AS score
FROM embeddings
ORDER BY embedding <#> $1::vector
LIMIT 5;

-- With metadata filter
SELECT content, 1 - (embedding <=> $1::vector) AS similarity
FROM embeddings
WHERE metadata->>'source' = 'docs'
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

### Python pgvector Usage

```python
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)
cur = conn.cursor()

# Insert embedding
embedding = np.random.rand(1536).astype(np.float32)  # Your actual embedding
cur.execute(
    "INSERT INTO embeddings (content, embedding, metadata) VALUES (%s, %s, %s)",
    ("Hello world", embedding, '{"source": "docs"}')
)

# Similarity search
query_vec = np.random.rand(1536).astype(np.float32)  # Your query embedding
cur.execute(
    "SELECT content, 1 - (embedding <=> %s) AS similarity "
    "FROM embeddings ORDER BY embedding <=> %s LIMIT 5",
    (query_vec, query_vec)
)
for row in cur.fetchall():
    print(f"{row[0]}: {row[1]:.3f}")

conn.commit()
```

### Go pgvector Usage

```go
import (
    "github.com/jackc/pgx/v5"
    "github.com/pgvector/pgvector-go"
)

conn, _ := pgx.Connect(ctx, os.Getenv("DATABASE_URL"))

// Insert
embedding := pgvector.NewVector(floats)
conn.Exec(ctx,
    "INSERT INTO embeddings (content, embedding) VALUES ($1, $2)",
    "Hello world", embedding,
)

// Search
rows, _ := conn.Query(ctx,
    "SELECT content, 1 - (embedding <=> $1) AS similarity "+
    "FROM embeddings ORDER BY embedding <=> $1 LIMIT 5",
    pgvector.NewVector(queryVec),
)
```

## Redis Patterns

```bash
# Connection
REDIS_URL=redis://redis.hanzo.svc:6379

# From K8s pod
redis-cli -h redis.hanzo.svc
```

### Common Patterns

```python
import redis

r = redis.from_url(os.environ["REDIS_URL"])

# Cache
r.setex("key", 3600, "value")  # 1 hour TTL
value = r.get("key")

# Session
r.hset(f"session:{session_id}", mapping={"user_id": "123", "role": "admin"})

# Rate limiting
key = f"ratelimit:{user_id}:{minute}"
count = r.incr(key)
r.expire(key, 60)
if count > 100:
    raise RateLimitExceeded()

# Job queue (BullMQ pattern)
r.xadd("jobs:inference", {"model": "zen-70b", "prompt": "Hello"})
```

## Local Development (compose.yml)

```yaml
services:
  postgres:
    image: postgres:16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: hanzo
      POSTGRES_USER: hanzo
      POSTGRES_PASSWORD: "${DB_PASSWORD}"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: hanzo
      MONGO_INITDB_ROOT_PASSWORD: "${MONGO_PASSWORD}"
    volumes:
      - mongo_data:/data/db

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: hanzo
      MINIO_ROOT_PASSWORD: "${MINIO_PASSWORD}"
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  redis_data:
  mongo_data:
  minio_data:
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port conflict | Service already running | `lsof -i :5432 && kill PID` |
| Connection refused | DB not started | `docker compose up -d postgres` |
| pgvector not found | Extension not installed | `CREATE EXTENSION vector;` |
| Slow queries | Missing index | Add IVFFlat or HNSW index |
| OOM | Too many connections | Configure `max_connections` |

```bash
# Port conflicts
lsof -i :5432

# Full reset (destructive — loses all data)
docker compose down -v && docker system prune -a

# Connect to prod DB (from K8s pod)
kubectl exec -it postgres-0 -n hanzo -- psql -U hanzo
```

## Related Skills

- `hanzo/hanzo-orm.md` - Go ORM for database access
- `hanzo/hanzo-datastore.md` - Vector database abstraction
- `hanzo/hanzo-stack.md` - Local stack with all DBs
- `hanzo/hanzo-kms.md` - Secret management (DB credentials)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: postgresql, redis, pgvector, mongodb, minio, database
**Prerequisites**: SQL, Redis basics, Docker
