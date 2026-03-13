# Hanzo Contracts - AI Infrastructure Smart Contracts

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-evm.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo Contracts is the **smart contract repository** for the Hanzo AI ecosystem. Built with **Foundry** and **Solidity 0.8.31**, it contains three core contracts for the AI token economy, faucet distribution, and on-chain identity registry.

The contracts use `@luxfi/` imports (from the Lux standard library) and OpenZeppelin for ERC20, access control, and utility patterns. The repo targets the Cancun EVM version with via-IR compilation and optimizer enabled.

**Note**: The `master` branch contains legacy Truffle-era contracts (Proxy, Versioned, Blacklist, Whitelist, Migrations with Solidity <0.6.0). The active `main` branch is the current Foundry-based codebase documented here.

## When to use

- Deploying or interacting with the $AI governance token
- Setting up token vesting schedules or staking
- Distributing tokens via the faucet mechanism
- Managing on-chain identity via HanzoRegistry
- Understanding the Hanzo AI token economics

## Hard requirements

1. **Foundry** (forge, cast, anvil)
2. **Solidity 0.8.31** compiler
3. Git submodules for dependencies (`@luxfi/standard`, `forge-std`)

## Quick reference

| Item | Value |
|------|-------|
| Framework | Foundry |
| Solidity | 0.8.31 |
| EVM target | Cancun |
| Repo | `github.com/hanzoai/contracts` |
| Branch | `main` |
| Build | `forge build` |
| Test | `forge test` |
| Format | `forge fmt` |
| Optimizer | 200 runs, via-IR |
| License | MIT |

## Contracts

| Contract | Path | Purpose |
|----------|------|---------|
| AIToken | `src/AIToken.sol` | ERC20 governance token with staking, vesting, and deflationary burn |
| AIFaucet | `src/AIFaucet.sol` | Token distribution / faucet mechanism |
| HanzoRegistry | `src/identity-registry/HanzoRegistry.sol` | On-chain identity registry |

## Project structure

```
contracts/          (main branch)
├── foundry.toml    # Foundry config (solc 0.8.31, Cancun, via-IR)
├── foundry.lock
├── package.json
├── src/
│   ├── AIToken.sol
│   ├── AIFaucet.sol
│   └── identity-registry/
│       └── HanzoRegistry.sol
├── scripts/        # Deployment scripts
├── broadcast/      # Deployment artifacts
├── lib/            # Git submodule dependencies
│   ├── standard/   # @luxfi/standard (Lux standard library)
│   └── forge-std/  # Foundry test utilities
└── .github/        # CI workflows
```

## AIToken.sol

ERC20 governance token ("AI Token", ticker "AI") with a 1 billion max supply:

- **Distribution**: 15% team, 35% ecosystem, 20% public sale, 10% liquidity, 20% staking rewards
- **Staking**: Lock periods of 0/30/90/180/365 days with 1-15% APR reward rates
- **Vesting**: Per-address vesting schedules with cliff and linear unlock
- **Burn**: Deflationary 0.1% burn rate on transfers (configurable by owner)
- **Pausable**: Owner can pause all transfers
- **Imports**: `@luxfi/tokens/LRC20/LRC20.sol`, `@luxfi/access/Access.sol`, `@luxfi/utils/Utils.sol`

## Foundry remappings

```toml
remappings = [
    "@luxfi/=lib/standard/contracts/",
    "@luxfi/standard/=lib/standard/contracts/",
    "@openzeppelin/contracts/=lib/standard/lib/openzeppelin-contracts/contracts/",
    "@openzeppelin/contracts-upgradeable/=lib/standard/lib/openzeppelin-contracts-upgradeable/contracts/",
    "forge-std/=lib/forge-std/src/",
]
```

## RPC endpoints (from foundry.toml)

```toml
hanzo_mainnet = "http://127.0.0.1:19630/ext/bc/.../rpc"
hanzo_testnet = "http://127.0.0.1:19640/ext/bc/.../rpc"
hanzo_devnet  = "http://127.0.0.1:19650/ext/bc/.../rpc"
```

These point to local Lux subnet RPC endpoints for development.

## Development

```bash
git clone https://github.com/hanzoai/contracts.git
cd contracts
git checkout main
git submodule update --init --recursive
forge build
forge test
forge fmt
```

## One-file quickstart

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.31;

import {AIToken} from "src/AIToken.sol";

// Deploy with treasury address
// AIToken token = new AIToken(treasuryAddress);
// token.stake(1000 * 10**18, 90);  // Stake 1000 AI for 90 days at 6% APR
```

## Legacy branch (master)

The `master` branch is a historical artifact containing the original Hanzo Solidity contracts:

| Contract | Solidity | Framework |
|----------|----------|-----------|
| Proxy.sol | >=0.4.25 <0.6.0 | Truffle |
| Versioned.sol | >=0.4.25 <0.6.0 | Truffle |
| Blacklist.sol | >=0.4.25 <0.6.0 | Truffle |
| Whitelist.sol | >=0.4.25 <0.6.0 | Truffle |
| Migrations.sol | >=0.4.25 <0.6.0 | Truffle |

Published as `hanzo-solidity` v0.2.1 on npm. Uses `openzeppelin-solidity` v2.1.2. Not used for new development.

## Related Skills

- `hanzo/hanzo-evm.md` - Rust EVM execution engine (reth fork)
- `hanzo/hanzo-web3.md` - Blockchain API access
- `hanzo/hanzo-web3-gateway.md` - Keyless blockchain access via x402

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: solidity, foundry, erc20, staking, vesting, luxfi
**Prerequisites**: Foundry, Solidity basics
