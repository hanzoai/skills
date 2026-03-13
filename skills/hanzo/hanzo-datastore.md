# Hanzo Datastore - Vector Database Integration

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/hanzo-search.md`, `hanzo/hanzo-engine.md`

## Overview

Hanzo Datastore provides a **unified vector database abstraction** for AI applications — embedding storage, similarity search, and RAG retrieval. Available in Go (`datastore-go`) and general (`datastore`). Supports multiple backends (pgvector, Pinecone, Weaviate, Qdrant, Milvus) behind a single interface.

### Why Hanzo Datastore?

- **Backend-agnostic**: Swap pgvector for Pinecone without code changes
- **Go-native**: High-performance, type-safe Go API
- **RAG-ready**: Built for Retrieval-Augmented Generation pipelines
- **Metadata filtering**: Rich query filters on document metadata
- **Batch operations**: Efficient bulk upsert and query

## When to use

- Building RAG (Retrieval-Augmented Generation) pipelines
- Storing and querying vector embeddings
- Semantic search over documents
- Nearest-neighbor queries with metadata filtering
- Abstracting away vector DB vendor specifics

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/hanzoai/datastore-go` |
| General | `github.com/hanzoai/datastore` |
| Backends | pgvector, Pinecone, Weaviate, Qdrant, Milvus |

## Go SDK

```bash
go get github.com/hanzoai/datastore-go
```

### Basic Usage

```go
import ds "github.com/hanzoai/datastore-go"

// Create store with pgvector backend
store, err := ds.New(ds.Config{
    Backend: "pgvector",
    DSN:     os.Getenv("DATABASE_URL"),
})
defer store.Close()

// Upsert a document
err = store.Upsert(ctx, ds.Document{
    ID:        "doc-1",
    Content:   "Hanzo AI is a frontier AI company",
    Embedding: embeddingVector, // []float32 from your embedding model
    Metadata:  map[string]any{"source": "docs", "section": "about"},
})

// Similarity search
results, err := store.Query(ctx, ds.QueryRequest{
    Embedding: queryVector,
    TopK:      5,
    Filter:    map[string]any{"source": "docs"},
})
for _, r := range results {
    fmt.Printf("%.3f: %s\n", r.Score, r.Content)
}
```

### Batch Operations

```go
// Batch upsert
docs := []ds.Document{
    {ID: "doc-1", Content: "First document", Embedding: vec1},
    {ID: "doc-2", Content: "Second document", Embedding: vec2},
    {ID: "doc-3", Content: "Third document", Embedding: vec3},
}
err = store.BatchUpsert(ctx, docs)

// Batch query
queries := []ds.QueryRequest{
    {Embedding: q1, TopK: 5},
    {Embedding: q2, TopK: 3},
}
results, err := store.BatchQuery(ctx, queries)
```

### Backend Configuration

```go
// pgvector (default, uses existing PostgreSQL)
store, _ := ds.New(ds.Config{
    Backend: "pgvector",
    DSN:     "postgresql://user:pass@postgres:5432/vectors",
    Options: map[string]any{
        "table":      "embeddings",
        "dimensions": 1536,
        "index_type": "hnsw",  // or "ivfflat"
    },
})

// Pinecone
store, _ := ds.New(ds.Config{
    Backend: "pinecone",
    Options: map[string]any{
        "api_key":     os.Getenv("PINECONE_API_KEY"),
        "environment": "us-east-1-aws",
        "index":       "my-index",
    },
})

// Qdrant
store, _ := ds.New(ds.Config{
    Backend: "qdrant",
    Options: map[string]any{
        "url":        "http://qdrant:6333",
        "collection": "documents",
    },
})

// Weaviate
store, _ := ds.New(ds.Config{
    Backend: "weaviate",
    Options: map[string]any{
        "url":   "http://weaviate:8080",
        "class": "Document",
    },
})
```

### RAG Pipeline

```go
import (
    ds "github.com/hanzoai/datastore-go"
    "github.com/hanzoai/go-sdk"
)

// Initialize
client := hanzo.NewClient()
store, _ := ds.New(ds.Config{Backend: "pgvector", DSN: dbURL})

// Index documents
for _, doc := range documents {
    // Generate embedding
    resp, _ := client.Embeddings.Create(ctx, hanzo.EmbeddingRequest{
        Model: "zen-embedding",
        Input: doc.Content,
    })

    // Store
    store.Upsert(ctx, ds.Document{
        ID:        doc.ID,
        Content:   doc.Content,
        Embedding: resp.Data[0].Embedding,
        Metadata:  map[string]any{"source": doc.Source},
    })
}

// Query (RAG retrieval step)
queryEmb, _ := client.Embeddings.Create(ctx, hanzo.EmbeddingRequest{
    Model: "zen-embedding",
    Input: userQuestion,
})

results, _ := store.Query(ctx, ds.QueryRequest{
    Embedding: queryEmb.Data[0].Embedding,
    TopK:      5,
})

// Build context for LLM
context := ""
for _, r := range results {
    context += r.Content + "\n\n"
}

// Generate answer with context
answer, _ := client.Chat.Completions.Create(ctx, hanzo.ChatRequest{
    Model: "zen-70b",
    Messages: []hanzo.Message{
        {Role: "system", Content: "Answer based on this context:\n" + context},
        {Role: "user", Content: userQuestion},
    },
})
```

### Document Interface

```go
type Document struct {
    ID        string             // Unique identifier
    Content   string             // Original text content
    Embedding []float32          // Vector embedding
    Metadata  map[string]any     // Filterable metadata
}

type QueryRequest struct {
    Embedding   []float32          // Query vector
    TopK        int                // Number of results
    Filter      map[string]any     // Metadata filters
    MinScore    float32            // Minimum similarity threshold
    IncludeContent bool            // Include original content in results
}

type QueryResult struct {
    ID        string
    Content   string
    Score     float32            // Similarity score (0-1)
    Metadata  map[string]any
}
```

## Related Skills

- `hanzo/hanzo-database.md` - PostgreSQL with pgvector (primary backend)
- `hanzo/hanzo-search.md` - Full-text + vector search (Meilisearch)
- `hanzo/hanzo-engine.md` - Generate embeddings for documents
- `hanzo/hanzo-orm.md` - Go ORM (for non-vector data)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vector, embeddings, rag, similarity-search, go
**Prerequisites**: Go, embedding concepts, vector search basics
