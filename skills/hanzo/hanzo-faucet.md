# Hanzo Faucet - Hanzo Network Token Faucet

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-explorer.md`, `hanzo/hanzo-contracts.md`

## Overview

Hanzo Faucet is a **token distribution service** for Hanzo Network (the AI Compute L1 blockchain). It distributes test $AI tokens on both Testnet (Chain ID: 36962) and Mainnet (Chain ID: 36963). The backend is an Express/TypeScript server using the `luxfi` SDK for EVM transactions. The frontend is a Next.js 16 app with RainbowKit wallet integration. Includes Solidity smart contracts built with Foundry. Rate limiting and Google reCAPTCHA v3 protect against abuse.

### What it actually is

- Express backend (port 8000) that signs and sends $AI token transactions
- Next.js 16 frontend (port 3000) with RainbowKit wallet connect
- pnpm workspace monorepo: root backend + `app/` frontend
- Uses `luxfi` SDK (not ethers/web3 directly) for chain interaction
- Solidity contracts in `contracts/` built with Foundry
- Google reCAPTCHA v3 verification middleware
- Per-address and global rate limiting
- Supports ERC20 token drips in addition to native $AI
- Dockerized deployment

### Hanzo Network details

- **Mainnet**: Chain ID 36963, RPC `https://rpc.hanzo.network`, 1 AI per drip, 48h rate limit
- **Testnet**: Chain ID 36962, RPC `https://rpc-testnet.hanzo.network`, 10 AI per drip, 24h rate limit
- **Token**: $AI (18 decimals)
- **Explorer**: `https://explorer.hanzo.network` / `https://explorer-testnet.hanzo.network`

## When to use

- Getting test tokens for Hanzo Network development
- Deploying a faucet for a new Hanzo Network chain or subnet
- Adding new ERC20 token drips to the faucet

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/faucet` |
| Package | `@hanzo/faucet` |
| Version | 1.0.0 |
| Branch | `main` |
| Backend | Express + TypeScript (port 8000) |
| Frontend | Next.js 16 + React 19 (port 3000) |
| Web3 | luxfi SDK, RainbowKit, wagmi v2, viem v2 |
| Contracts | Solidity 0.8.28, Foundry |
| Package Manager | pnpm (workspace) |
| License | BSD-3-Clause |

## Quickstart

```bash
git clone https://github.com/hanzoai/faucet.git
cd faucet
pnpm install

# Configure
cp .env.example .env
# Set PK (private key), CAPTCHA_SECRET

# Generate a new wallet
pnpm generate

# Start backend (port 8000)
pnpm dev

# Start frontend (port 3000)
pnpm dev:app

# Start both
pnpm dev:all

# Build for production
pnpm build
pnpm start
```

## Architecture

```
  Browser (port 3000)          Express API (port 8000)
  ┌──────────────────┐         ┌───────────────────────┐
  │  Next.js 16 App  │────────>│  /api/sendToken       │
  │  RainbowKit      │         │  /api/faucetAddress    │
  │  wagmi/viem      │         │  /api/getBalance       │
  └──────────────────┘         │  /api/getChainConfigs  │
                               │  /health               │
                               └──────────┬────────────┘
                                          │
                               ┌──────────▼────────────┐
                               │  EVM Instance (luxfi)  │
                               │  Signs & sends tx      │
                               │  Per-chain private keys│
                               └───────────────────────┘
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sendToken` | Send tokens (requires captcha) |
| GET | `/api/faucetAddress?chain=C` | Get faucet wallet address |
| GET | `/api/getBalance?chain=C` | Get faucet balance |
| GET | `/api/getChainConfigs` | Get all chain configurations |
| GET | `/health` | Health check |
| GET | `/ip` | Client IP (Cloudflare aware) |

## Directory structure

```
hanzoai/faucet/
  package.json         # @hanzo/faucet v1.0.0 (root workspace)
  server.ts            # Express backend entry point
  config.json          # Chain configs (testnet/mainnet), rate limits
  types.ts             # TypeScript type definitions
  middlewares/         # Rate limiter, captcha, URI parser
  vms/
    evm.ts             # EVM chain instance (luxfi SDK)
  client/              # Legacy frontend (being replaced by app/)
  app/                 # Next.js 16 frontend (pnpm workspace)
    src/               # React components
    e2e/               # Playwright tests
    package.json       # @hanzo/faucet-app
  contracts/           # Solidity smart contracts
    src/               # Contract source
    test/              # Foundry tests
    script/            # Deploy scripts
    foundry.toml       # Foundry config
  scripts/
    generateKey.ts     # Wallet key generation
  Dockerfile           # Container build
```

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `PK` | Yes | Default faucet wallet private key |
| `HANZO_TESTNET` | No | Testnet-specific private key (overrides PK) |
| `HANZO_MAINNET` | No | Mainnet-specific private key (overrides PK) |
| `CAPTCHA_SECRET` | Yes | Google reCAPTCHA v3 secret |
| `V2_CAPTCHA_SECRET` | No | reCAPTCHA v2 fallback secret |
| `PORT` | No | Backend port (default: 8000) |

## Rate limiting

- **Global**: 40 requests per minute across all endpoints
- **Per-chain testnet**: 1 request per 24 hours per address
- **Per-chain mainnet**: 1 request per 48 hours per address
- Uses `cf-connecting-ip` header for Cloudflare deployments
- `REVERSE_PROXIES` config controls trusted proxy hop count

## Contracts (Foundry)

```bash
cd contracts
forge build           # Compile
forge test            # Run tests
forge test --gas-report
```

## Related Skills

- `hanzo/hanzo-explorer.md` -- Block explorer for Hanzo Network
- `hanzo/hanzo-contracts.md` -- Smart contract patterns

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: faucet, blockchain, hanzo-network, tokens, evm, web3
**Prerequisites**: Node.js 17+, pnpm 8+, funded wallet with AI tokens, reCAPTCHA keys
