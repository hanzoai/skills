# Hanzo Vector - Vector Database for AI Embeddings

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/hanzo-storage.md`, `hanzo/hanzo-llm-gateway.md`

## Overview

Hanzo Vector is a **placeholder repository** for a planned vector database service optimized for AI workloads. The repo currently contains only a README and MIT license -- no implementation code exists yet.

The README describes a production-ready vector database with HNSW indexing, similarity search, and REST/gRPC APIs, but these features are **not yet implemented** in the repository.

### Planned Features (per README, not yet built)

- **HNSW Indexing**: Approximate nearest neighbor search
- **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product
- **Payload Filtering**: Combine vector search with metadata filters
- **Horizontal Scaling**: Distributed architecture
- **REST & gRPC APIs**: Language-agnostic integration
- **Cloud Native**: Kubernetes-ready with Helm charts

### Current State

The repository at `github.com/hanzoai/vector` contains exactly two files:
- `README.md` -- Feature description and planned Docker quickstart
- `LICENSE` -- MIT License (Copyright 2026 Hanzo AI)

No source code, no Dockerfile, no go.mod, no tests. This is a stub repo.

## When to use

- Do **not** depend on this repo for production use -- it has no implementation
- For vector search today, use PostgreSQL with pgvector (already in the Hanzo stack)
- Watch this repo for future development of a dedicated vector database

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/vector` |
| Status | **Stub / Placeholder** |
| License | MIT |
| Code | None |
| Planned Docker | `docker run -p 6333:6333 -p 6334:6334 hanzo/vector` |
| Planned Ports | 6333 (REST), 6334 (gRPC) |

## Current Alternatives in Hanzo Stack

For vector search needs today, use:

```sql
-- PostgreSQL with pgvector extension
CREATE EXTENSION vector;
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Similarity search
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM embeddings
ORDER BY embedding <=> $1
LIMIT 10;
```

## Related Skills

- `hanzo/hanzo-database.md` - PostgreSQL (pgvector for current vector search)
- `hanzo/hanzo-storage.md` - S3-compatible object storage
- `hanzo/hanzo-llm-gateway.md` - LLM Gateway (generates embeddings)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vector, embeddings, similarity-search, ai
**Prerequisites**: None (repo is a placeholder)
