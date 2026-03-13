# Hanzo Smart Contracts - ERC Standards & DeFi Protocols

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-evm.md`, `hanzo/hanzo-commerce-api.md`, `hanzo/hanzo-web3.md`

## Overview

Hanzo Smart Contracts provides production-ready **ERC token implementations and DeFi protocols** — ERC20, ERC721, ERC1155, staking, farming, vaults, and governance. Built with Foundry for fast compilation, testing, and deployment.

### Why Hanzo Contracts?

- **Battle-tested**: OpenZeppelin-based implementations
- **Full DeFi suite**: Staking, farming, vaults, governance
- **Foundry-native**: Fast compilation, fuzz testing, gas snapshots
- **Multi-chain**: Deploy to any EVM chain (Ethereum, Lux, L2s)
- **Auditable**: Clean, documented Solidity code

## When to use

- Deploying ERC20/721/1155 tokens for projects
- Building DeFi protocols (staking, farming, yield vaults)
- Governance contract deployment
- Smart contract integration testing
- Token launches on EVM chains

## Hard requirements

1. **Foundry** (`forge`, `cast`, `anvil`) installed
2. **Solidity 0.8.20+**
3. RPC endpoint for target chain
4. Private key for deployment

## Quick reference

| Item | Value |
|------|-------|
| Framework | Foundry (forge, cast, anvil) |
| Solidity | ^0.8.20 |
| Repo | `github.com/hanzoai/contracts` |
| Build | `forge build` |
| Test | `forge test -v` |
| Gas report | `forge test --gas-report` |
| Deploy | `forge script script/Deploy.s.sol --broadcast` |
| Dependencies | OpenZeppelin Contracts v5.x |

## Contract Types

| Contract | Standard | Purpose |
|----------|----------|---------|
| Token | ERC20 | Fungible tokens (utility, governance) |
| NFT | ERC721 | Non-fungible tokens (collectibles, memberships) |
| Multi-Token | ERC1155 | Mixed fungible + NFT (gaming, marketplace) |
| Staking | Custom | Lock tokens for time-weighted rewards |
| Farming | Custom | Liquidity mining / yield farming |
| Vault | ERC4626 | Yield-bearing tokenized vaults |
| Governor | Governor | On-chain voting + timelock execution |
| Vesting | Custom | Token vesting with cliff/linear release |
| Multisig | Custom | Multi-signature operations |

## One-file quickstart

### ERC20 Token

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";

contract HanzoToken is ERC20, ERC20Burnable, ERC20Permit {
    constructor() ERC20("Hanzo Token", "HANZO") ERC20Permit("Hanzo Token") {
        _mint(msg.sender, 1_000_000_000 * 10 ** decimals());
    }
}
```

### ERC721 NFT

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC721} from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import {ERC721URIStorage} from "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

contract HanzoNFT is ERC721, ERC721URIStorage, Ownable {
    uint256 private _nextTokenId;

    constructor() ERC721("Hanzo NFT", "HNFT") Ownable(msg.sender) {}

    function mint(address to, string memory uri) public onlyOwner returns (uint256) {
        uint256 tokenId = _nextTokenId++;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        return tokenId;
    }
}
```

### Staking Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

contract HanzoStaking {
    using SafeERC20 for IERC20;

    IERC20 public immutable stakingToken;
    IERC20 public immutable rewardToken;
    uint256 public rewardRate; // rewards per second

    mapping(address => uint256) public staked;
    mapping(address => uint256) public lastUpdate;
    mapping(address => uint256) public rewards;

    constructor(address _stakingToken, address _rewardToken, uint256 _rewardRate) {
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
        rewardRate = _rewardRate;
    }

    function stake(uint256 amount) external {
        _updateReward(msg.sender);
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);
        staked[msg.sender] += amount;
    }

    function withdraw(uint256 amount) external {
        _updateReward(msg.sender);
        staked[msg.sender] -= amount;
        stakingToken.safeTransfer(msg.sender, amount);
    }

    function claim() external {
        _updateReward(msg.sender);
        uint256 reward = rewards[msg.sender];
        rewards[msg.sender] = 0;
        rewardToken.safeTransfer(msg.sender, reward);
    }

    function _updateReward(address account) internal {
        if (staked[account] > 0) {
            rewards[account] += staked[account] * rewardRate
                * (block.timestamp - lastUpdate[account]) / 1e18;
        }
        lastUpdate[account] = block.timestamp;
    }
}
```

### ERC4626 Vault

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC4626} from "@openzeppelin/contracts/token/ERC20/extensions/ERC4626.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract HanzoVault is ERC4626 {
    constructor(IERC20 asset)
        ERC4626(asset)
        ERC20("Hanzo Vault Share", "vHANZO")
    {}
}
```

## CLI Usage

```bash
# Build all contracts
forge build

# Run tests with verbosity
forge test -v

# Run specific test
forge test --match-test testStake -vvv

# Gas report
forge test --gas-report

# Deploy to network
forge script script/Deploy.s.sol:DeployScript \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast \
  --verify

# Verify on Etherscan
forge verify-contract $CONTRACT_ADDR src/HanzoToken.sol:HanzoToken \
  --etherscan-api-key $ETHERSCAN_KEY

# Interact with deployed contract
cast call $CONTRACT_ADDR "totalSupply()" --rpc-url $RPC_URL
cast send $CONTRACT_ADDR "transfer(address,uint256)" $TO 1000 \
  --rpc-url $RPC_URL --private-key $PRIVATE_KEY

# Local testing with Anvil
anvil --fork-url $RPC_URL --fork-block-number 18000000
```

## Project Structure

```
contracts/
├── src/
│   ├── tokens/          # ERC20, ERC721, ERC1155
│   ├── defi/            # Staking, farming, vaults
│   ├── governance/      # Governor, timelock
│   └── utils/           # Shared utilities
├── test/                # Foundry tests
├── script/              # Deploy scripts
├── lib/                 # Dependencies (forge install)
└── foundry.toml         # Foundry config
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Compile error | Wrong Solidity version | Check `pragma` matches foundry.toml |
| Test OOG | Gas limit too low | Increase in foundry.toml `gas_limit` |
| Deploy fails | Insufficient funds | Fund deployer address |
| Verify fails | Wrong constructor args | Pass `--constructor-args` |

## Related Skills

- `hanzo/hanzo-evm.md` - EVM execution engine
- `hanzo/hanzo-web3.md` - Blockchain API access
- `hanzo/hanzo-web3-gateway.md` - Keyless blockchain access via x402
- `hanzo/hanzo-vault.md` - PCI card tokenization (different from DeFi vaults)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: solidity, erc20, erc721, defi, smart-contracts, foundry
**Prerequisites**: Solidity, Foundry, EVM concepts
