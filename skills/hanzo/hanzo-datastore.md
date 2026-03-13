# Hanzo Datastore - Vector Database Integration

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/hanzo-search.md`, `hanzo/hanzo-engine.md`

## Overview

Hanzo Datastore provides **vector database integration** for AI applications — embedding storage, similarity search, and RAG retrieval. Available in Go (`datastore-go`) and general (`datastore`).

## When to use

- Storing and querying vector embeddings
- Building RAG (Retrieval-Augmented Generation) pipelines
- Semantic search over documents
- Nearest-neighbor queries

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/hanzoai/datastore-go` |
| Local | ``github.com/hanzoai/datastore``, ``github.com/hanzoai/datastore-go`` |

## One-file quickstart

```go
import ds "github.com/hanzoai/datastore-go"

store, _ := ds.New(ds.Config{
    Backend: "pgvector",
    DSN:     os.Getenv("DATABASE_URL"),
})

// Store embedding
store.Upsert(ctx, ds.Document{
    ID:        "doc-1",
    Content:   "Hanzo AI is a frontier AI company",
    Embedding: embeddingVector, // []float32
    Metadata:  map[string]any{"source": "docs"},
})

// Similarity search
results, _ := store.Query(ctx, ds.QueryRequest{
    Embedding: queryVector,
    TopK:      5,
    Filter:    map[string]any{"source": "docs"},
})
```

## Related Skills

- `hanzo/hanzo-database.md` - PostgreSQL with pgvector
- `hanzo/hanzo-search.md` - AI-powered search
- `hanzo/hanzo-engine.md` - Generate embeddings

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
