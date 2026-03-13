# Hanzo Memory - AI Memory Service with LanceDB

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Memory (`@hanzo/memory`) is a **TypeScript AI memory service** that stores, searches, and retrieves contextual memories using vector embeddings. It provides a REST API server (Fastify) and a TypeScript client library. The primary backend is LanceDB for vector storage, with 6 embedding provider options including local (ONNX, Candle, llama.cpp) and API-based (OpenAI). Port of the Python memory service with full feature parity.

### What it actually is

- npm package `@hanzo/memory` v0.1.0
- Fastify REST API server (`src/server.ts`) on port 8000
- TypeScript client (`src/client.ts`) for programmatic access
- LanceDB vector database backend (default) + in-memory backend (testing)
- 6 embedding providers: Mock, OpenAI, ONNX, Transformers.js, Candle (Rust/Metal), llama.cpp
- 2 LLM providers: OpenAI, Mock (for PII stripping and result filtering)
- Zod for runtime type validation
- 110 tests passing (92 unit + 18 integration)
- Dockerfile with 4 build stages: production, test, benchmark, development
- pnpm monorepo with `packages/embedding` and `packages/inference` sub-packages

## When to use

- Adding persistent memory to AI agents or chatbots
- Storing user preferences and conversation context
- Building knowledge bases with semantic search
- Chat session history with vector search
- Any service needing user-scoped memory with embeddings

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/memory` |
| Package | `@hanzo/memory` |
| Version | 0.1.0 |
| Branch | `main` |
| Language | TypeScript |
| Runtime | Node.js >= 18 |
| Server | Fastify 4 on port 8000 |
| DB | LanceDB (default), in-memory (testing) |
| Validation | Zod |
| Tests | 110 passing (92 unit + 18 integration) |
| Test runner | Vitest |
| Build | tsup |
| Package manager | pnpm |
| Docker | Multi-stage (production, test, benchmark, dev) |
| CI | GitHub Actions (Linux + macOS matrix) |
| License | BSD-3-Clause |

## Project structure

```
hanzoai/memory/
  package.json              # @hanzo/memory 0.1.0
  tsconfig.json             # TypeScript config
  tsup.config.ts            # Build config (tsup)
  vitest.config.ts          # Test config
  Dockerfile                # 4-stage build
  docker-compose.yml        # Server, test, benchmark, dev targets
  .env.example              # Configuration template
  src/
    index.ts                # Package exports
    server.ts               # Fastify REST API server
    client.ts               # MemoryClient class
    config.ts               # Configuration loading from env
    db/                     # Database backends
    models/                 # Zod schemas and types
    services/
      memory.ts             # Core MemoryService
      embeddings.ts         # Embedding provider factory
      embeddings/           # Provider implementations
      llm.ts                # LLM provider (OpenAI, Mock)
      index.ts              # Service exports
    types/                  # TypeScript type definitions
  packages/
    embedding/              # Standalone embedding package
    inference/              # Standalone inference package
  tests/                    # Unit and integration tests
  benchmarks/               # Performance benchmarks
```

## Dependencies

From `package.json`:
- `@lancedb/lancedb` ^0.4.0 -- vector database
- `fastify` ^4.25.0 -- HTTP server
- `fastify-zod` ^1.4.0 -- Zod integration for Fastify
- `openai` ^4.0.0 -- OpenAI API client
- `zod` ^3.22.0 -- runtime type validation
- `uuid` ^9.0.0 -- unique identifiers
- `dotenv` ^16.0.0 -- env config
- `onnxruntime-node` 1.22.0-rev -- local ONNX embeddings (optional)
- `sharp` ^0.34.3 -- image processing (optional, for Transformers.js)

## API endpoints

### Health
- `GET /health` -- service health status

### Memory operations
- `POST /v1/remember` -- store a memory
- `POST /v1/memories/add` -- add memory (alias)
- `POST /v1/memories/get` -- get specific memory
- `POST /v1/memories/search` -- semantic search
- `DELETE /v1/memories` -- delete memory
- `POST /v1/memories/delete` -- delete (RPC-style)
- `POST /v1/user/delete` -- delete all user data

### Projects
- `POST /v1/project/create` -- create project
- `GET /v1/projects` -- list user projects

### Knowledge base
- `POST /v1/kb/create` -- create knowledge base
- `GET /v1/kb/list` -- list knowledge bases
- `POST /v1/kb/facts/add` -- add fact
- `POST /v1/kb/facts/get` -- search facts
- `POST /v1/kb/facts/delete` -- delete facts

### Chat sessions
- `POST /v1/chat/sessions/create` -- create session
- `POST /v1/chat/messages/add` -- add message
- `GET /v1/chat/sessions/:session_id/messages` -- get messages
- `POST /v1/chat/search` -- search messages

## Quickstart

### As a library

```typescript
import { MemoryClient } from '@hanzo/memory'

const client = new MemoryClient()

// Store a memory
const memory = await client.remember({
  userid: 'user-123',
  content: 'User prefers dark mode',
  importance: 8
})

// Semantic search
const results = await client.search({
  userid: 'user-123',
  query: 'What are the user preferences?',
  limit: 5
})
```

### As a server

```bash
git clone https://github.com/hanzoai/memory.git
cd memory
pnpm install
pnpm run server:dev   # Dev mode with auto-reload on port 8000
```

### Docker

```bash
docker-compose up memory-server      # Production
docker-compose up memory-dev         # Development with hot-reload
docker-compose up memory-test        # Run tests
docker-compose up memory-benchmark   # Run benchmarks
```

## Embedding providers

| Provider | Key | Local | GPU | Dependencies |
|----------|-----|-------|-----|-------------|
| Mock | `mock` | Yes | No | None (default) |
| OpenAI | `openai` | No | N/A | `OPENAI_API_KEY` |
| ONNX | `onnx` | Yes | No | `onnxruntime-node` |
| Transformers.js | `transformers` | Yes | No | `sharp` module |
| Candle | `candle` | Yes | Metal (macOS) | Rust + `candle-embeddings` |
| llama.cpp | `llama` | Yes | CUDA/Metal | `llama.cpp` binary + model |

Configure via `EMBEDDING_PROVIDER` env var.

## Environment variables

```bash
# Database
DB_BACKEND=lancedb              # or 'memory'
LANCEDB_URI=./lancedb_data

# Embeddings
EMBEDDING_PROVIDER=mock         # mock|openai|onnx|transformers|candle|llama
EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2

# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Server
HANZO_HOST=0.0.0.0
HANZO_PORT=8000

# Features
STRIP_PII_DEFAULT=false
FILTER_WITH_LLM_DEFAULT=false
```

## Troubleshooting

- **sharp install fails on macOS**: Use `EMBEDDING_PROVIDER=mock` or `EMBEDDING_PROVIDER=openai` to avoid Transformers.js dependency
- **LanceDB write errors**: Check `LANCEDB_URI` directory exists and is writable
- **ONNX returns mock embeddings**: Full ONNX model support planned, currently falls back to mock
- **Candle not found**: Install via `cargo install candle-embeddings`
- **Vite CJS deprecation warning**: Comes from Vitest, does not affect functionality

## Related Skills

- `hanzo/hanzo-agent.md` -- Agent framework that uses memory
- `hanzo/hanzo-mcp.md` -- MCP integration for memory tools
- `hanzo/hanzo-chat.md` -- Chat application with memory support
- `hanzo/hanzo-candle.md` -- Rust ML framework (Candle embeddings)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: memory, vector-db, lancedb, embeddings, ai, agent
**Prerequisites**: Node.js >= 18, pnpm
