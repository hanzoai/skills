# Hanzo Search - High-Performance Search Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/hanzo-datastore.md`

## Overview

Hanzo Search is a **high-performance search engine** built as a Rust workspace with 26 crates. Fork of Meilisearch v1.37.0 with AI-powered ranking, vector search, and faceting. Designed for sub-50ms search across millions of documents.

### Features

- **Full-text search**: Typo-tolerant, prefix matching, phrase search
- **Vector search**: Hybrid keyword + semantic search with embeddings
- **AI ranking**: ML-based relevance scoring and reranking
- **Faceted search**: Filterable, sortable, with distribution counts
- **Multi-index**: Multiple independent search indices
- **RESTful API**: Simple HTTP/JSON interface
- **Self-hostable**: Single binary, no external dependencies

### OSS Base

Fork of **Meilisearch** v1.37.0. Repo: `hanzoai/search`.

## When to use

- Adding search to applications or documentation
- Building product catalogs with faceted filtering
- Implementing hybrid keyword + semantic search
- Replacing Elasticsearch/Algolia with a lighter alternative

## Quick reference

| Item | Value |
|------|-------|
| Tech | Rust workspace (26 crates) |
| Binary | `hanzo-search` |
| Port | 7700 (default) |
| Repo | `github.com/hanzoai/search` |

## One-file quickstart

```bash
cd search

# Build
cargo build --release

# Run
./target/release/hanzo-search --http-addr 0.0.0.0:7700 --master-key YOUR_KEY

# Index documents
curl -X POST http://localhost:7700/indexes/movies/documents \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @movies.json

# Search
curl http://localhost:7700/indexes/movies/search \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"q": "batman", "limit": 10}'
```

### Configuration

```bash
# Environment variables
MEILI_MASTER_KEY=your-master-key    # API authentication
MEILI_DB_PATH=./data.ms             # Data directory
MEILI_HTTP_ADDR=0.0.0.0:7700       # Listen address
MEILI_ENV=production                # production or development
MEILI_MAX_INDEXING_MEMORY=2Gi      # Indexing memory limit
```

### Vector Search

```bash
# Create index with embedder
curl -X PATCH http://localhost:7700/indexes/docs/settings \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "embedders": {
      "default": {
        "source": "openAi",
        "apiKey": "sk-...",
        "model": "text-embedding-3-small",
        "dimensions": 1536
      }
    }
  }'

# Hybrid search (keyword + vector)
curl http://localhost:7700/indexes/docs/search \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"q": "how to deploy", "hybrid": {"semanticRatio": 0.5}}'
```

## Workspace Crates

Key crates in the Rust workspace:

| Crate | Purpose |
|-------|---------|
| `meilisearch` | Main binary and HTTP API |
| `milli` | Core indexing engine |
| `index-scheduler` | Async task scheduling |
| `meilisearch-auth` | API key management |
| `meilitool` | CLI maintenance tool |
| `filter-parser` | Query filter DSL |
| `json-depth-checker` | Document validation |
| `flatten-serde-json` | Nested document flattening |

## Docker

```bash
docker run -p 7700:7700 \
  -v $(pwd)/meili_data:/meili_data \
  -e MEILI_MASTER_KEY=YOUR_KEY \
  ghcr.io/hanzoai/search:latest
```

## Related Skills

- `hanzo/hanzo-database.md` - PostgreSQL backend storage
- `hanzo/hanzo-datastore.md` - Vector database integration

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: search, rust, meilisearch, full-text, vector
**Prerequisites**: Rust toolchain (for building from source)
