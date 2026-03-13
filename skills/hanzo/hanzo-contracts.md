# Hanzo Smart Contracts - ERC Standards & DeFi Protocols

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-evm.md`, `hanzo/hanzo-commerce-api.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo Smart Contracts provides production-ready **ERC token implementations and DeFi protocols** — ERC20, ERC721, ERC1155, staking, farming, vaults, and governance.

## When to use

- Deploying ERC20/721/1155 tokens
- Building DeFi protocols (staking, farming, vaults)
- Governance contract deployment
- Smart contract integration testing

## Quick reference

| Item | Value |
|------|-------|
| Tech | Solidity, Foundry |
| Repo | `github.com/hanzoai/contracts` |
| Build | `forge build` |
| Test | `forge test -v` |

## One-file quickstart

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract HanzoToken is ERC20 {
    constructor() ERC20("Hanzo Token", "HANZO") {
        _mint(msg.sender, 1_000_000_000 * 10 ** decimals());
    }
}
```

```bash
# Deploy with Foundry
forge create src/HanzoToken.sol:HanzoToken \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY
```

## Contract Types

| Contract | Standard | Purpose |
|----------|----------|---------|
| Token | ERC20 | Fungible tokens |
| NFT | ERC721 | Non-fungible tokens |
| Multi-Token | ERC1155 | Mixed fungible/NFT |
| Staking | Custom | Lock tokens for rewards |
| Farming | Custom | Liquidity mining |
| Vault | ERC4626 | Yield-bearing vaults |
| Governance | Governor | On-chain voting |

## Related Skills

- `hanzo/hanzo-evm.md` - EVM execution engine
- `hanzo/hanzo-web3.md` - Blockchain API access
- `hanzo/hanzo-web3-gateway.md` - Keyless blockchain access

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: solidity, erc20, defi, smart-contracts
**Prerequisites**: Solidity, Foundry, EVM concepts
