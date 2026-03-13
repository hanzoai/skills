# Hanzo EVM - Rust Execution Engine for Post-Quantum Web3

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/rust-sdk.md`, `hanzo/hanzo-node.md`, `hanzo/hanzo-contracts.md`

## Overview

Hanzo EVM is a **Rust-based EVM execution engine** — a fork of paradigmxyz/reth, rebranded for the Hanzo ecosystem. Designed as a toolkit for building post-quantum Web3 infrastructure, L2/L3 app chains, and high-performance blockchain nodes.

### OSS Base

Fork of **reth** (paradigmxyz/reth). Branch: `rebrand/hanzo-evm`.

## When to use

- Building blockchain nodes with EVM execution
- L2/L3 app chain development
- Post-quantum Web3 infrastructure
- High-performance transaction processing

## Quick reference

| Item | Value |
|------|-------|
| Binary | `hanzo-evm` |
| Repo | `github.com/hanzoai/evm` |
| Build | `cargo build --release` |
| Test | `cargo test` |

## Naming Convention

| Layer | Convention | Example |
|-------|-----------|---------|
| Crate names | `hanzo-evm-*` | `hanzo-evm-execution` |
| Module imports | `hanzo_evm_*` | `hanzo_evm_execution` |
| Internal modules | clean/neutral | `mod evm;` |
| Types | clean `Evm*` | `EvmApi`, `EvmRpcModule` |
| Env vars | `EVM_*` | neutral, no hanzo prefix |
| Config/data | neutral | `evm.toml`, `evm/` |

### Key Renames from reth

| reth | Hanzo EVM |
|------|-----------|
| `reth-evm` | `hanzo-evm-execution` (NOT hanzo-evm-evm) |
| `reth-evm-ethereum` | `hanzo-evm-eth-execution` |
| `reth-ethereum` | `hanzo-evm-ethereum` (umbrella) |
| `reth-nippy-jar` | `hanzo-evm-nippy-jar` |

## One-file quickstart

```bash
# Build
# Clone from github.com/hanzoai first
cd <project>
cargo build --release

# Run node
./target/release/hanzo-evm node --chain mainnet

# With custom config
EVM_DATA_DIR=/data/evm ./target/release/hanzo-evm node \
  --chain mainnet \
  --http \
  --http.port 8545
```

## Design Principles

- Crate names: `hanzo-evm-*` (required for publishing)
- Internal code: FREE of hanzo branding unless necessary
- No compound words (`Evm` not `HanzoEvm`)
- Neutral env/config (`EVM_*` not `HANZO_EVM_*`)
- Clean, beautiful, simple to build hanzo node with

## Related Skills

- `hanzo/hanzo-node.md` - Distributed AI node (uses EVM plugin)
- `hanzo/rust-sdk.md` - Rust SDK
- `hanzo/hanzo-contracts.md` - Smart contracts

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: evm, blockchain, rust, reth
**Prerequisites**: Rust, EVM concepts, blockchain fundamentals
