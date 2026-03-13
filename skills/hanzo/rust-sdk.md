# Hanzo Rust SDK - Infrastructure, Agents, PQC & MCP

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-node.md`, `hanzo/hanzo-evm.md`

## Overview

Hanzo Rust SDK is a **full Rust infrastructure SDK** — not just an API client. Contains 14 active crates covering agent framework, MCP implementation, post-quantum cryptography, AI safety (LLM I/O sanitization), and multi-backend proxy.

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/rust-sdk` |
| Version | 1.1.12 |
| Rust edition | 2021 |
| MSRV | 1.85 |
| License | MIT OR Apache-2.0 |
| Active crates | 14 |
| Disabled crates | 30+ (in development) |

## Active Crates

### Core

| Crate | Purpose |
|-------|---------|
| `hanzo-config` | Configuration management |
| `hanzo-message-primitives` | Core message schemas |

### Cryptography

| Crate | Purpose |
|-------|---------|
| `hanzo-pqc` | Post-quantum crypto (ML-KEM, ML-DSA) |
| `hanzo-crypto` | General crypto primitives |
| `hanzo-did` | W3C Decentralized Identifiers |

### AI Safety

| Crate | Purpose |
|-------|---------|
| `hanzo-guard` | LLM I/O sanitization — PII redaction, injection detection |
| `hanzo-extract` | Content extraction with sanitization |

### Agent Framework

| Crate | Purpose |
|-------|---------|
| `hanzo-agent` | Agent framework core |
| `hanzo-agent-proxy` | Multi-backend OpenAI-compatible proxy |
| `hanzo-agents` | Specialized agents (architect, CTO, reviewer) |

### MCP Stack

| Crate | Purpose |
|-------|---------|
| `hanzo-mcp-core` | MCP protocol types and serialization |
| `hanzo-mcp-client` | MCP client implementation |
| `hanzo-mcp-server` | MCP server implementation |
| `hanzo-mcp` | Unified MCP interface |

## Disabled Crates (In Development)

30+ crates in various stages: `hanzo-kbs`, `hanzo-identity`, `hanzo-libp2p`, `hanzo-compute`, `hanzo-database`, `hanzo-embedding`, `hanzo-llm`, `hanzo-hmm`, `hanzo-wasm`, `hanzo-mining`, `hanzo-marketplace`, `hanzo-baml`, `hanzo-db`, `hanzo-simulation`, `hanzo-sqlite`.

## Usage Examples

### Post-Quantum Crypto

```rust
use hanzo_pqc::{ml_kem, ml_dsa};

// ML-KEM key encapsulation
let (ek, dk) = ml_kem::keygen();
let (ct, ss) = ml_kem::encapsulate(&ek);
let ss2 = ml_kem::decapsulate(&dk, &ct);
assert_eq!(ss, ss2);

// ML-DSA signatures
let (pk, sk) = ml_dsa::keygen();
let sig = ml_dsa::sign(&sk, message);
assert!(ml_dsa::verify(&pk, message, &sig));
```

### AI Safety (Guard)

```rust
use hanzo_guard::{Guard, Policy};

let guard = Guard::new(Policy::default());

// Sanitize LLM input (detect injection)
let safe_input = guard.sanitize_input(user_message)?;

// Sanitize LLM output (redact PII)
let safe_output = guard.sanitize_output(llm_response)?;
```

### Agent Proxy

```rust
use hanzo_agent_proxy::Proxy;

// Multi-backend proxy (OpenAI-compatible)
let proxy = Proxy::builder()
    .backend("openai", openai_config)
    .backend("anthropic", anthropic_config)
    .backend("local", ollama_config)
    .build();

proxy.serve("0.0.0.0:8080").await?;
```

### MCP Server

```rust
use hanzo_mcp::{Server, Tool};

let server = Server::builder()
    .name("my-tools")
    .tool(Tool::new("search", search_handler))
    .tool(Tool::new("compute", compute_handler))
    .build();

server.serve_stdio().await?;
```

## Related Skills

- `hanzo/hanzo-mcp.md` — MCP protocol (TypeScript counterpart)
- `hanzo/hanzo-node.md` — Rust AI agent node
- `hanzo/hanzo-evm.md` — Rust EVM (reth fork)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: rust, sdk, pqc, mcp, agents, safety
**Prerequisites**: Rust, async/tokio, cryptography basics
