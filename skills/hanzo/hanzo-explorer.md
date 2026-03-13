# Hanzo Explorer - EVM Blockchain Explorer for Lux Network

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-evm.md`, `hanzo/hanzo-network.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo Explorer is an **EVM blockchain explorer** for inspecting and analyzing transactions, accounts, smart contracts, and tokens on Lux Network and other EVM-compatible chains. It is a fork of Blockscout (v8.1.1), an open-source alternative to Etherscan. Built with Elixir/Phoenix (backend) and a separate frontend, it provides full blockchain indexing, smart contract verification, and a REST/GraphQL API. Includes Lux-specific Docker configurations for mainnet deployment.

### Why Hanzo Explorer?

- **Full EVM support**: Transactions, accounts, balances, tokens (ERC-20/721/1155), internal transactions
- **Smart contract verification**: Solidity source verification and interaction UI
- **GraphQL + REST API**: Programmatic access to all indexed blockchain data
- **NFT media handling**: Dedicated media handler for NFT assets
- **Lux-native**: Pre-built Docker configs for Lux mainnet/testnet
- **Microservices**: Stats, visualizer (Sol2UML), sig-provider, user-ops-indexer

### Tech Stack

- **Backend**: Elixir 1.17.3 + OTP 27 (Phoenix framework)
- **Database**: PostgreSQL 14+
- **Cache**: Redis
- **Frontend**: Node.js 20.17 (separate Blockscout frontend)
- **Indexer**: Custom blockchain indexer (Elixir)
- **JSON-RPC**: ethereum_jsonrpc client for EVM nodes
- **Deployment**: Docker Compose, Kubernetes, Ansible
- **Microservices**: Rust-based (stats, visualizer, sig-provider, user-ops-indexer)
- **CI**: CircleCI

### OSS Base

Repo: `hanzoai/explorer` (fork of `blockscout/blockscout`). GPL v3.0 License.

## When to use

- Deploying a block explorer for Lux Network or any EVM chain
- Indexing and querying blockchain data (transactions, tokens, contracts)
- Verifying and interacting with smart contracts
- Building analytics dashboards on top of blockchain data
- Running a self-hosted alternative to Etherscan

## Hard requirements

1. **Elixir 1.17+** with OTP 27
2. **PostgreSQL 14+** for blockchain data storage
3. **Redis** for caching
4. **Running EVM JSON-RPC node** (Lux, Geth, Erigon, etc.)
5. **Node.js 20+** for frontend build

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/explorer` |
| Branch | `master` |
| Version | 8.1.1 |
| Backend | Elixir 1.17 + Phoenix |
| Database | PostgreSQL 14+ |
| License | GPL v3.0 |
| Upstream | `blockscout/blockscout` |

## One-file quickstart

### Docker Compose (quickest)

```bash
# Standard EVM chain
cd docker-compose
docker-compose up --build

# Lux mainnet specific
cd docker-luxnet
docker-compose -f explorer.yml up -d
```

### Manual development

```bash
# Install Elixir dependencies
mix deps.get

# Create and migrate database
mix ecto.create && mix ecto.migrate

# Start the server
mix phx.server
```

## Core Concepts

### Architecture

```
hanzoai/explorer/
  apps/                         Umbrella Elixir apps
    block_scout_web/            Phoenix web app (API + UI)
    ethereum_jsonrpc/           JSON-RPC client for EVM nodes
    explorer/                   Core data models + indexing logic
    indexer/                    Blockchain indexer (blocks, txs, tokens)
    nft_media_handler/          NFT media processing
    utils/                      Shared utilities
  config/                       Elixir configuration
  docker-compose/               Generic Docker configs
    docker-compose.yml          Standard setup (Postgres, Redis, backend, frontend, nginx)
    erigon.yml                  Erigon JSON-RPC client config
    geth.yml                    Geth JSON-RPC client config
    hardhat-network.yml         HardHat dev network config
    microservices.yml           Stats, visualizer, sig-provider
    envs/                       Environment variable files
  docker-luxnet/                Lux Network specific configs
    explorer.yml                Lux mainnet explorer compose
    microservices.yml           Lux microservices compose
    services/                   Service definitions
  docker-testnet/               Testnet configurations
  docker/                       Dockerfile and build configs
  rel/                          Elixir release configuration
  bin/                          Helper scripts
  mix.exs                       Umbrella project definition
```

### Umbrella Apps

| App | Purpose |
|-----|---------|
| `block_scout_web` | Phoenix web application, REST API, GraphQL API |
| `ethereum_jsonrpc` | JSON-RPC client for communicating with EVM nodes |
| `explorer` | Core domain logic, Ecto schemas, chain data models |
| `indexer` | Real-time blockchain indexer (blocks, transactions, tokens, logs) |
| `nft_media_handler` | NFT image/media fetching and processing |
| `utils` | Shared utility functions |

### Microservices (Rust)

| Service | Purpose |
|---------|---------|
| Stats | Chain statistics and analytics |
| Sol2UML Visualizer | Solidity contract visualization |
| Sig-provider | Function signature decoding |
| User-ops-indexer | ERC-4337 account abstraction indexing |

### Docker Compose Configs

| Config | Use Case |
|--------|----------|
| `docker-compose.yml` | Standard full setup with all services |
| `erigon.yml` | Erigon JSON-RPC backend |
| `geth.yml` | Geth/Reth JSON-RPC backend |
| `geth-clique-consensus.yml` | Geth with Clique PoA |
| `hardhat-network.yml` | HardHat local dev network |
| `external-db.yml` | Explorer with external PostgreSQL |
| `external-backend.yml` | Frontend with external API |
| `external-frontend.yml` | Backend with external frontend |
| `microservices.yml` | All Rust microservices |
| `no-services.yml` | Explorer only, no microservices |

### Environment Variables

```bash
# Backend (in docker-compose/envs/common-blockscout.env)
DATABASE_URL=postgresql://user:pass@postgres:5432/blockscout
ETHEREUM_JSONRPC_VARIANT=geth
ETHEREUM_JSONRPC_HTTP_URL=http://host.docker.internal:8545
ETHEREUM_JSONRPC_WS_URL=ws://host.docker.internal:8546

# Frontend (in docker-compose/envs/common-frontend.env)
NEXT_PUBLIC_API_HOST=localhost
NEXT_PUBLIC_API_PORT=80
```

## Lux Network Integration

The `docker-luxnet/` directory contains Lux-specific configurations for connecting to Lux Network's EVM-compatible C-Chain. These configs are pre-tuned for Lux's block times, gas model, and RPC endpoints.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Indexer not syncing | Wrong JSON-RPC URL | Verify `ETHEREUM_JSONRPC_HTTP_URL` points to running node |
| DB connection error | PostgreSQL not ready | Wait for DB startup or check `DATABASE_URL` |
| Frontend blank | API URL mismatch | Check `NEXT_PUBLIC_API_HOST` matches backend |
| Linux localhost issue | Docker networking | Use `http://0.0.0.0/` instead of `http://127.0.0.1/` |

## Related Skills

- `hanzo/hanzo-evm.md` - Hanzo EVM execution engine (Rust, reth fork)
- `hanzo/hanzo-network.md` - Lux/Hanzo network infrastructure
- `hanzo/hanzo-web3.md` - Web3 services and gateway
- `hanzo/hanzo-contracts.md` - Smart contract development

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: blockchain, explorer, evm, elixir, blockscout, lux-network
**Prerequisites**: Elixir, PostgreSQL, Redis, EVM node
