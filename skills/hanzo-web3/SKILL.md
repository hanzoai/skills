---
name: hanzo-web3
description: "Enterprise blockchain SDK for multi-chain RPC, tokens, NFTs, smart wallets, and ERC-4337 account abstraction via the hanzo-web3 Python SDK."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-web3",
              "label": "Install Hanzo Web3 SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo Web3 — Enterprise Blockchain SDK

`pip install hanzo-web3`

Multi-chain blockchain operations: RPC calls, token/NFT management, smart wallets, ERC-4337 account abstraction.

## Quick Start

```python
from hanzo_web3 import Client

client = Client(api_key="your-api-key")

# Get ETH balance
balance = await client.get_balance("0x...")
print(f"Balance: {balance} ETH")
```

## Async Client

```python
from hanzo_web3 import AsyncClient

client = AsyncClient(
    api_key="your-api-key",
    base_url="https://api.web3.hanzo.ai",
    timeout=30,
    max_retries=3
)

# Token operations
tokens = await client.tokens.list(chain="ethereum")
balance = await client.tokens.balance("0x...", token="USDC")
```

## RPC Operations

```python
from hanzo_web3 import RPCClient

rpc = RPCClient(api_key="...")

# Direct JSON-RPC calls
block = await rpc.call("eth_getBlockByNumber", ["latest", True])
tx = await rpc.call("eth_getTransactionReceipt", [tx_hash])
```

## NFT Operations

```python
# List NFTs owned by address
nfts = await client.nfts.list(owner="0x...", chain="ethereum")

# Get NFT metadata
metadata = await client.nfts.metadata(contract="0x...", token_id=42)
```

## Smart Wallets (ERC-4337)

```python
# Create smart wallet with account abstraction
wallet = await client.wallets.create(
    owner="0x...",
    chain="ethereum"
)

# Execute gasless transaction
tx = await wallet.execute(
    to="0x...",
    value=0,
    data="0x..."
)
```

## Webhooks

```python
# Register webhook for on-chain events
webhook = await client.webhooks.create(
    url="https://my-app.com/hook",
    events=["transfer", "approval"],
    chain="ethereum"
)
```

## Supported Chains

Ethereum, Polygon, Arbitrum, Optimism, Base, and more EVM-compatible chains.

## Configuration

```python
from hanzo_web3 import ClientConfig

config = ClientConfig(
    api_key="your-api-key",
    base_url="https://api.web3.hanzo.ai",
    timeout=30,
    max_retries=3
)
```

## Environment Variables

```bash
HANZO_WEB3_API_KEY=...                      # API key
HANZO_WEB3_BASE_URL=https://api.web3.hanzo.ai  # API endpoint
```
