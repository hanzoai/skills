---
name: hanzo-memory
description: "AI memory and knowledge management with Hanzo Memory. Store memories, manage knowledge bases, semantic search, chat history, and fact management via the hanzo-memory Python SDK."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-memory",
              "label": "Install Hanzo Memory SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo Memory — AI Memory & Knowledge

`pip install hanzo-memory`

Vector-backed memory service with semantic search, knowledge bases, and chat history.

## Quick Start

```python
from hanzo_memory import MemoryService

memory = MemoryService()

# Store a memory
await memory.store(key="user:preferences", value="Prefers Python over JavaScript")

# Search memories
results = await memory.search("programming language preferences", limit=5)
for r in results:
    print(r.content, r.score)
```

## Environment Variables

```bash
HANZO_API_KEY=your-key                          # API key
HANZO_LLM_MODEL=gpt-4o-mini                    # LLM for processing
OPENAI_API_KEY=...                               # For embeddings
HANZO_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # Embedding model
HANZO_DB_BACKEND=lancedb                         # Backend: lancedb or infinity
HANZO_LANCEDB_PATH=data/lancedb                 # Storage path
HANZO_DISABLE_AUTH=false                         # Disable auth (dev only)
```

## Memory Operations

```python
# Store memory
await memory.store(
    key="context:meeting-notes",
    value="Q4 planning: focus on AI agents and MCP tools",
    metadata={"project": "hanzo-bot", "date": "2026-02-26"}
)

# Retrieve by key
result = await memory.retrieve(key="context:meeting-notes")

# Semantic search
results = await memory.search(
    query="what are the Q4 priorities?",
    limit=10
)

# Delete memory
await memory.delete(key="context:meeting-notes")
```

## Knowledge Base Management

```python
# Create knowledge base
await memory.create_kb(name="product-docs", description="Product documentation")

# Add facts to knowledge base
await memory.add_fact(
    kb_name="product-docs",
    fact="Hanzo Bot supports 100+ LLM providers via the unified gateway"
)

await memory.add_fact(
    kb_name="product-docs",
    fact="IAM supports OAuth2 with PKCE for browser-based apps"
)

# Query facts
facts = await memory.query_facts(
    kb_name="product-docs",
    query="How many LLM providers are supported?",
    limit=5
)

# List knowledge bases
kbs = await memory.list_kbs()
```

## Chat History

```python
# Store chat message
await memory.add_message(
    session_id="session-123",
    role="user",
    content="How do I deploy to Hanzo PaaS?"
)

await memory.add_message(
    session_id="session-123",
    role="assistant",
    content="You can deploy using the Platform API..."
)

# Retrieve chat history
messages = await memory.get_messages(session_id="session-123")

# Search across chat history
results = await memory.search_messages(query="deployment", limit=10)
```

## REST API (FastAPI Service)

Run the memory service:

```bash
uvicorn hanzo_memory.server:app --port 8100
```

### Endpoints

```
POST   /v1/remember              # Store memory
GET    /v1/memories               # List memories
GET    /v1/memories/{key}         # Get memory by key
DELETE /v1/memories/{key}         # Delete memory
POST   /search                    # Semantic search

POST   /v1/kb                     # Create knowledge base
GET    /v1/kb                     # List knowledge bases
POST   /v1/kb/facts               # Add fact
GET    /v1/kb/facts/{kb}          # Query facts

POST   /v1/chat/messages          # Add chat message
GET    /v1/chat/sessions/{id}     # Get session messages
POST   /v1/chat/search            # Search messages
```

## Database Backends

### LanceDB (default)

- Cross-platform (Linux, macOS, Windows, ARM, WASM)
- Embedded, no server needed
- Good for development and small deployments

### InfinityDB (high-performance)

- Linux/Windows only
- Better for large-scale production
- Requires separate server process

Configure via `HANZO_DB_BACKEND=lancedb` or `HANZO_DB_BACKEND=infinity`.

## Multi-Tenancy

```python
# Isolate memories by user/project
await memory.store(
    key="user:alice:notes",
    value="...",
    metadata={"user_id": "alice", "project": "my-project"}
)

# Search within tenant scope
results = await memory.search(
    query="notes",
    filters={"user_id": "alice"}
)
```
