# Hanzo Contracts - Legacy Smart Contract Infrastructure

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-evm.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo Contracts is a **legacy smart contract repository** containing 5 infrastructure contracts built with Truffle and Solidity <0.6.0. These are foundational proxy/access-control contracts from Hanzo's earlier blockchain work.

**NOTE**: This is a legacy repo. It uses **Truffle** (not Foundry), **Solidity <0.6.0** (not 0.8+), and contains only 5 infrastructure contracts — NOT a full DeFi suite. For modern EVM work, see `hanzo/hanzo-evm.md`.

### Contracts

| Contract | Purpose |
|----------|---------|
| `Proxy` | Upgradeable proxy pattern |
| `Versioned` | Contract versioning |
| `Blacklist` | Address blacklist access control |
| `Whitelist` | Address whitelist access control |
| `Migrations` | Truffle migration tracking |

### OSS Info

Repo: `hanzoai/contracts`. Stack: Truffle + Solidity <0.6.0.

## When to use

- Understanding Hanzo's legacy contract infrastructure
- Upgrading or migrating legacy proxy contracts
- Reference for access-control patterns
- **NOT for new contract development** — use Foundry + Solidity 0.8+ instead

## Hard requirements

1. **Truffle** framework
2. **Solidity <0.6.0** compiler
3. **Node.js** (for Truffle)

## Quick reference

| Item | Value |
|------|-------|
| Framework | Truffle |
| Solidity | <0.6.0 |
| Repo | `github.com/hanzoai/contracts` |
| Build | `truffle compile` |
| Test | `truffle test` |
| Deploy | `truffle migrate --network <network>` |
| Contracts | 5 (Proxy, Versioned, Blacklist, Whitelist, Migrations) |

## Project Structure

```
contracts/
├── contracts/
│   ├── Proxy.sol           # Upgradeable proxy
│   ├── Versioned.sol       # Version tracking
│   ├── Blacklist.sol       # Address blacklist
│   ├── Whitelist.sol       # Address whitelist
│   └── Migrations.sol      # Truffle migrations
├── migrations/             # Truffle deployment scripts
├── test/                   # JavaScript tests
├── truffle-config.js       # Truffle configuration
└── package.json
```

## Contract Details

### Proxy.sol

Upgradeable proxy pattern allowing contract logic upgrades without changing the contract address:

```solidity
// Solidity <0.6.0
pragma solidity ^0.5.0;

contract Proxy {
    address public implementation;
    address public admin;

    function upgradeTo(address newImplementation) external;
    fallback() external payable;
}
```

### Blacklist.sol / Whitelist.sol

Access control contracts for managing allowed/denied addresses:

```solidity
pragma solidity ^0.5.0;

contract Whitelist {
    mapping(address => bool) public whitelisted;

    function addToWhitelist(address account) external;
    function removeFromWhitelist(address account) external;
    function isWhitelisted(address account) external view returns (bool);
}
```

### Versioned.sol

Tracks contract versions for upgrade management:

```solidity
pragma solidity ^0.5.0;

contract Versioned {
    uint256 public version;

    function setVersion(uint256 newVersion) external;
}
```

## Development

```bash
git clone https://github.com/hanzoai/contracts.git
cd contracts
npm install
truffle compile
truffle test
truffle migrate --network development
```

## Migration to Modern Stack

For new smart contract development, use:
- **Foundry** (forge, cast, anvil) instead of Truffle
- **Solidity 0.8.20+** with OpenZeppelin v5
- **ERC standards**: ERC20, ERC721, ERC1155, ERC4626
- See `hanzo/hanzo-evm.md` for the modern EVM execution engine

## Related Skills

- `hanzo/hanzo-evm.md` - Modern EVM execution engine (Rust, reth fork)
- `hanzo/hanzo-web3.md` - Blockchain API access
- `hanzo/hanzo-web3-gateway.md` - Keyless blockchain access via x402

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: solidity, truffle, legacy, proxy, access-control
**Prerequisites**: Truffle, Solidity basics
