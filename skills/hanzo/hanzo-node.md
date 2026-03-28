# Hanzo Node - Rust AI Agent Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-desktop.md`, `hanzo/hanzo-mcp.md`, `hanzo/python-sdk.md`

## Overview

Hanzo Node is a **Rust-based AI agent infrastructure node** that lets you create AI agents without writing code. It provides local AI inference, peer-to-peer networking, MCP tool support, and an HTTP API for agent orchestration. It is the backend engine that powers the Hanzo Desktop application.

Workspace version **1.1.20**, Rust edition **2021**, requires Rust >= 1.85.

### OSS Info

Repo: `github.com/hanzoai/node`. Companion desktop frontend: `github.com/hanzoai/desktop`.

## When to use

- Running a local AI agent node for inference and tool execution
- Building codeless AI agents with task scheduling and automation
- Embedding AI capabilities into applications via the HTTP API
- Participating in the Hanzo peer-to-peer compute network
- Developing against the Hanzo agent protocol (MCP, libp2p, DID)

## Hard requirements

1. **Rust >= 1.85** (required for `std::fs::exists`)
2. Platform: Linux, macOS, or Windows
3. SQLite (bundled via rusqlite)

## Quick reference

| Item | Value |
|------|-------|
| Language | Rust |
| Edition | 2021 |
| Version | 1.1.20 |
| Repo | `github.com/hanzoai/node` |
| Binary | `hanzo-bin/hanzo-node` |
| CLI | `hanzo-bin/hanzoai` |
| Makefile binary | `hanzoai` (renamed from `hanzod` on 2026-03-28) |
| Build | `cargo build` |
| Test | `IS_TESTING=1 cargo test -- --test-threads=1` |
| API port | 3690 |
| P2P port | 3691 |
| Config | `hanzo.toml` |
| License | Apache-2.0 |

## Project structure

```
node/
├── Cargo.toml              # Workspace root
├── Cargo.lock
├── Makefile
├── hanzo.toml              # Node configuration
├── hanzo-bin/
│   ├── hanzo-node/         # Main node binary
│   ├── hanzoai/            # CLI binary
│   └── hanzo-migrate/      # Database migration tool
├── hanzo-libs/             # Library crates (~40 crates)
│   ├── hanzo-api/          # HTTP API layer
│   ├── hanzo-config/       # Configuration management
│   ├── hanzo-database/     # Database abstraction
│   ├── hanzo-db-sqlite/    # SQLite backend
│   ├── hanzo-did/          # Decentralized identity
│   ├── hanzo-embed/        # Embedding generation
│   ├── hanzo-fs/           # File system operations
│   ├── hanzo-http-api/     # HTTP API endpoints
│   ├── hanzo-identity/     # Identity management
│   ├── hanzo-jobs/         # Job execution
│   ├── hanzo-job-queue-manager/ # Job queue
│   ├── hanzo-libp2p/       # P2P networking (libp2p 0.55)
│   ├── hanzo-libp2p-relayer/ # P2P relay node
│   ├── hanzo-llm/          # LLM provider integrations
│   ├── hanzo-mcp/          # Model Context Protocol
│   ├── hanzo-messages/     # Message protocol definitions
│   ├── hanzo-mining/       # Compute mining
│   ├── hanzo-models/       # Model management
│   ├── hanzo-model-discovery/ # Model discovery
│   ├── hanzo-pqc/          # Post-quantum cryptography
│   ├── hanzo-vm/           # Virtual machine
│   ├── hanzo-compute/      # Compute orchestration
│   ├── hanzo-ai-format/    # AI data formats
│   ├── hanzo-runner/       # Task runner
│   ├── hanzo-runtime/      # Runtime environment
│   ├── hanzo-tools/        # Tool definitions
│   ├── hanzo-tools-runner/ # Tool execution engine
│   ├── hanzo-wasm/         # WebAssembly support
│   └── hanzo-wasm-runtime/ # WASM runtime
├── hanzo-test-framework/   # Testing framework
├── hanzo-test-macro/       # Test macros
├── cloud-node/             # Cloud node variant
├── contracts/              # Associated smart contracts
├── docker/                 # Docker configurations
├── docker-build/           # Docker build scripts
├── docs/                   # Documentation
├── examples/               # Usage examples
├── knowledge/              # Knowledge base files
├── scripts/                # Build and run scripts
└── HIPs/                   # Hanzo Improvement Proposals
```

## Workspace crates

The workspace contains approximately 40 crates in `hanzo-libs/`. Key groupings:

**Core infrastructure**: hanzo-api, hanzo-config, hanzo-database, hanzo-db-sqlite, hanzo-http-api, hanzo-fs, hanzo-messages, hanzo-runtime

**AI / ML**: hanzo-llm, hanzo-embed, hanzo-models, hanzo-model-discovery, hanzo-ai-format, hanzo-mcp

**Networking**: hanzo-libp2p, hanzo-libp2p-relayer, hanzo-compute, hanzo-mining

**Identity / Security**: hanzo-did, hanzo-identity, hanzo-pqc

**Execution**: hanzo-vm, hanzo-wasm, hanzo-wasm-runtime, hanzo-runner, hanzo-tools, hanzo-tools-runner, hanzo-jobs, hanzo-job-queue-manager

**Excluded from build** (noted in Cargo.toml comments): hanzo-db, hanzo-hmm, hanzo-kbs, hanzo-l2, hanzo-consensus, hanzo-sheet, hanzo-simulation, hanzo-tests -- these have external path dependencies or incomplete implementations.

## Configuration (hanzo.toml)

The node is configured via `hanzo.toml` at the repo root:

```toml
[node]
ip = "0.0.0.0"
port = 3691               # P2P port
api_ip = "0.0.0.0"
api_port = 3690            # HTTP API port

[embeddings]
use_native_embeddings = true
use_gpu = true
default_embedding_model = "qwen3-embedding-8b"

[database]
path = "./storage/db.sqlite"

[security]
pqc_enabled = false
privacy_tier = 0           # 0-4: Open to TEE-I/O

[tools]
mcp_enabled = true
javascript_runtime = "deno"
python_runtime = "uv"

[llm_providers]
ollama_base_url = "http://localhost:11434"
lm_studio_base_url = "http://localhost:1234"
```

See the full `hanzo.toml` in the repo for all options including logging, performance tuning, wallet configuration, and development settings.

## Important: Deno binary (2026-03-27)
- The 112MB deno binary has been removed from git and LFS tracking
- `hanzo-bin/hanzo-node/shinkai-tools-runner-resources/deno` is in `.gitignore`
- Deno must be installed separately on the host or in the container at build time
- Do NOT commit the deno binary back into the repo

## Building and running

```bash
git clone https://github.com/hanzoai/node.git
cd node
cargo build

# Quick start with localhost script
sh scripts/run_node_localhost.sh

# To reset, delete the storage/ folder and rebuild
```

## Testing

```bash
# Run all tests (must be from repo root)
IS_TESTING=1 cargo test -- --test-threads=1

# Run a specific test
IS_TESTING=1 cargo test tcp_node_test -- --nocapture --test-threads=1

# Dockerized tests
docker build -t testing_image -f .github/Dockerfile .
docker run --entrypoint /entrypoints/run-main-cargo-tests.sh testing_image
```

## API

The node exposes an HTTP API on port 3690 with OpenAPI documentation:

```bash
# Swagger UI (requires --features hanzo_node/swagger-ui)
http://{NODE_IP}:3690/v2/swagger-ui/

# Generate OpenAPI schemas
cargo run --example generate_openapi_docs
# Output goes to docs/openapi/
```

## Key dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| tokio | 1.36 | Async runtime |
| libp2p | 0.55.0 | P2P networking (noise, yamux, tcp, quic, relay) |
| rmcp | 0.8 | Model Context Protocol |
| warp | 0.3.7 | HTTP server |
| rusqlite | 0.32.1 | SQLite (bundled) |
| ed25519-dalek | 2.1.1 | Cryptographic signatures |
| serde / serde_json | 1.0 | Serialization |
| reqwest | 0.11.27 | HTTP client |

## Docker

Docker build files are in `docker/` and `docker-build/`. The node is designed to run as a single binary with minimal dependencies.

## Related Skills

- `hanzo/hanzo-desktop.md` - Desktop frontend (Tauri + React) that wraps this node
- `hanzo/hanzo-mcp.md` - MCP protocol and tools
- `hanzo/python-sdk.md` - Python client SDK
- `hanzo/hanzo-evm.md` - EVM execution engine (separate Rust project)

---

**Last Updated**: 2026-03-27
**Category**: Hanzo Ecosystem
**Related**: rust, ai, infrastructure, libp2p, mcp, agents
**Prerequisites**: Rust >= 1.85, basic understanding of AI agent systems
