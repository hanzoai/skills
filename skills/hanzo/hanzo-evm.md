# Hanzo EVM - Rust Execution Engine for Post-Quantum Web3

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/rust-sdk.md`, `hanzo/hanzo-node.md`, `hanzo/hanzo-contracts.md`

## Overview

Hanzo EVM is a **Rust-based EVM execution engine** — fork of paradigmxyz/reth v1.11.0, rebranded for the Hanzo ecosystem. Designed as a modular toolkit for building post-quantum Web3 infrastructure, L2/L3 app chains, and high-performance blockchain nodes.

### Why Hanzo EVM?

- **Modular architecture**: Plug-and-play components for custom chains
- **Post-quantum ready**: Integration with Hanzo PQC (ML-KEM, ML-DSA)
- **High performance**: Parallel EVM execution, optimized storage
- **Clean abstractions**: Neutral naming, beautiful API surface
- **Full reth compatibility**: All upstream features + Hanzo extensions

### OSS Base

Fork of **paradigmxyz/reth** v1.11.0. Rust edition 2024, MSRV 1.93.

**NOTE**: Crate rename is in progress. The binary is `hanzo-evm` and `bin/hanzo-evm/` exists, but most workspace dependency names still reference `reth-*` internally. The naming convention below describes the target state.

## When to use

- Building blockchain nodes with EVM execution
- L2/L3 app chain development on Lux or standalone
- Post-quantum Web3 infrastructure
- High-performance transaction processing
- Custom consensus + EVM combinations

## Hard requirements

1. **Rust 1.93+** (edition 2024)
2. **~1300+ source files** across workspace
3. Build: `cargo build --release` (~15-30 min first build)

## Quick reference

| Item | Value |
|------|-------|
| Binary | `hanzo-evm` |
| Repo | `github.com/hanzoai/evm` |
| Upstream | paradigmxyz/reth v1.11.0 |
| Rust edition | 2024 |
| MSRV | 1.93 |
| Build | `cargo build --release` |
| Test | `cargo test` |
| License | MIT OR Apache-2.0 |

## Naming Convention

| Layer | Convention | Example |
|-------|-----------|---------|
| Crate names | `hanzo-evm-*` | `hanzo-evm-execution` |
| Module imports | `hanzo_evm_*` | `hanzo_evm_execution` |
| Internal modules | clean/neutral | `mod evm;` |
| Types | clean `Evm*` | `EvmApi`, `EvmRpcModule` |
| Env vars | `EVM_*` | neutral, no hanzo prefix |
| Config/data | neutral | `evm.toml`, `evm/`, `evm.ipc` |
| Binary | `hanzo-evm` | single binary |

### Key Renames from reth

| reth | Hanzo EVM | Rationale |
|------|-----------|-----------|
| `reth` (binary) | `hanzo-evm` | Brand identity |
| `reth-evm` | `hanzo-evm-execution` | Avoid `hanzo-evm-evm` |
| `reth-evm-ethereum` | `hanzo-evm-eth-execution` | Clean naming |
| `reth-ethereum` | `hanzo-evm-ethereum` | Umbrella crate |
| `reth-nippy-jar` | `hanzo-evm-nippy-jar` | Lib name = `hanzo_evm_nippy_jar` |
| `reth-primitives` | `hanzo-evm-primitives` | Core types |
| `reth-db` | `hanzo-evm-db` | Storage layer |
| `reth-rpc` | `hanzo-evm-rpc` | JSON-RPC |
| `reth-network` | `hanzo-evm-network` | P2P networking |
| `reth-consensus` | `hanzo-evm-consensus` | Consensus interface |
| `reth-payload-builder` | `hanzo-evm-payload-builder` | Block building |
| `reth-transaction-pool` | `hanzo-evm-transaction-pool` | Mempool |

## Workspace Crates (Key)

### Core

| Crate | Purpose |
|-------|---------|
| `hanzo-evm` | Main binary + CLI |
| `hanzo-evm-primitives` | Core types (Block, Transaction, Receipt) |
| `hanzo-evm-execution` | EVM execution engine |
| `hanzo-evm-ethereum` | Ethereum umbrella (mainnet, sepolia, holesky) |
| `hanzo-evm-eth-execution` | Ethereum-specific EVM rules |

### Storage

| Crate | Purpose |
|-------|---------|
| `hanzo-evm-db` | Database abstraction (MDBX) |
| `hanzo-evm-db-api` | Database trait interfaces |
| `hanzo-evm-db-common` | Shared DB utilities |
| `hanzo-evm-nippy-jar` | Compressed static file storage |
| `hanzo-evm-stages` | Sync pipeline stages |
| `hanzo-evm-static-file-types` | Static file type definitions |

### Networking

| Crate | Purpose |
|-------|---------|
| `hanzo-evm-network` | P2P networking (devp2p) |
| `hanzo-evm-network-api` | Network trait interfaces |
| `hanzo-evm-discv4` | Node discovery v4 |
| `hanzo-evm-discv5` | Node discovery v5 |
| `hanzo-evm-dns-discovery` | DNS-based discovery |
| `hanzo-evm-eth-wire` | Ethereum wire protocol |

### RPC

| Crate | Purpose |
|-------|---------|
| `hanzo-evm-rpc` | JSON-RPC server |
| `hanzo-evm-rpc-api` | RPC trait interfaces |
| `hanzo-evm-rpc-eth-api` | eth_* namespace |
| `hanzo-evm-rpc-engine-api` | Engine API (consensus layer) |

### Consensus

| Crate | Purpose |
|-------|---------|
| `hanzo-evm-consensus` | Consensus trait + validators |
| `hanzo-evm-auto-seal` | Dev mode auto-sealing |
| `hanzo-evm-beacon-consensus` | Beacon chain consensus |

### Block Building

| Crate | Purpose |
|-------|---------|
| `hanzo-evm-payload-builder` | Block payload construction |
| `hanzo-evm-transaction-pool` | Transaction mempool |
| `hanzo-evm-basic-payload-builder` | Default payload builder |

## One-file quickstart

```bash
# Build from source
git clone https://github.com/hanzoai/evm.git
cd evm
cargo build --release

# Run mainnet node
./target/release/hanzo-evm node --chain mainnet

# Run with HTTP + WS RPC
./target/release/hanzo-evm node \
  --chain mainnet \
  --http \
  --http.port 8545 \
  --http.api eth,net,web3,debug,trace \
  --ws \
  --ws.port 8546

# Run dev mode (auto-sealing)
./target/release/hanzo-evm node --dev

# Custom data directory
EVM_DATA_DIR=/data/evm ./target/release/hanzo-evm node --chain mainnet
```

### CLI Commands

```bash
hanzo-evm node                    # Run full node
hanzo-evm node --dev              # Development mode
hanzo-evm init                    # Initialize database
hanzo-evm import <file>           # Import blocks from file
hanzo-evm db stats                # Database statistics
hanzo-evm db diff                 # Compare DB states
hanzo-evm p2p header <hash>       # Fetch header via P2P
hanzo-evm p2p body <hash>         # Fetch body via P2P
hanzo-evm stage run --stage <n>   # Run specific sync stage
hanzo-evm recover storage-tries   # Recover storage tries
```

### Configuration (evm.toml)

```toml
[node]
chain = "mainnet"
datadir = "/data/evm"
instance = 1

[rpc]
http = true
http_addr = "0.0.0.0"
http_port = 8545
ws = true
ws_port = 8546

[network]
port = 30303
max_peers = 100

[stages.sync]
pipeline_batch_size = 10000
```

## Design Principles

1. **Crate names**: `hanzo-evm-*` (required for publishing)
2. **Internal code**: FREE of hanzo branding unless necessary
3. **No compound words**: `Evm` not `HanzoEvm`
4. **Neutral env/config**: `EVM_*` not `HANZO_EVM_*`
5. **Clean, beautiful, simple** to build a Hanzo node with

## Architecture Vision

- **Hanzo EVM** = execution engine / toolkit for Post-Quantum Web3
- L2/L3 app chains on **Lux consensus** for verticals (commerce, AI compute)
- Eventual L1 capability; toolkit supports all layers
- Commerce chain: blockchain backend for Hanzo Commerce
- Fork alloy + revm for full post-quantum ZK EVM stack

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails | Rust too old | Install Rust 1.93+ |
| MDBX error | Max readers | Increase `EVM_DB_MAX_READERS` |
| Sync stalls | Network issues | Check peers with `hanzo-evm p2p info` |
| OOM on build | Low memory | Use `cargo build -j 2` to limit parallelism |

## Related Skills

- `hanzo/hanzo-node.md` - Distributed AI node (uses EVM plugin)
- `hanzo/rust-sdk.md` - Rust SDK (PQC crypto for EVM)
- `hanzo/hanzo-contracts.md` - Smart contracts deployed on EVM

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: evm, blockchain, rust, reth, execution-engine
**Prerequisites**: Rust 1.93+, EVM concepts, blockchain fundamentals
