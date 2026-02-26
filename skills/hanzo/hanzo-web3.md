# Hanzo Web3 - Blockchain API for 100+ Chains

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-web3-gateway.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-commerce.md`

## Overview

Hanzo Web3 is the **blockchain infrastructure layer** of the Hanzo ecosystem. It provides unified API access to 100+ EVM and non-EVM chains including Ethereum, Solana, Base, Arbitrum, Polygon, Lux, and more. Node RPC, structured data, webhooks, smart wallets, and Solana-specific APIs are all available through a single API key.

### Why Hanzo Web3?

- **One key, 100+ chains**: No per-chain configuration or separate providers
- **Structured data**: Token balances, NFT metadata, portfolio views without indexing
- **Push notifications**: Managed webhooks for onchain events
- **Account abstraction**: ERC-4337 smart wallets with gas sponsorship
- **Solana native**: Yellowstone gRPC, DAS, and WebSocket support
- **Part of Hanzo ecosystem**: Integrates with Hanzo Chat, Commerce, Console, and MCP

## When to use

Use this skill when:
- The user wants to read or write blockchain data
- The user needs token balances, NFT metadata, or transaction history
- The user wants to deploy or interact with smart contracts
- The user needs real-time onchain event notifications via webhooks
- The user wants to set up smart wallets or account abstraction

## Hard requirements

1. **API Key required.** If `HANZO_WEB3_API_KEY` is not set, tell the user to get one free at https://web3.hanzo.ai/dashboard, or switch to the `hanzo-web3-gateway` skill for keyless access.
2. **Never expose the API key** in user-visible output, logs, or screenshots.
3. **Rate limits apply.** Free tier: 25 req/s, PAYG: 300 req/s. Respect 429 responses with exponential backoff.

## Preflight checks

Before making any request, silently verify:
- `HANZO_WEB3_API_KEY` environment variable is set and non-empty
- If unset, suggest: `export HANZO_WEB3_API_KEY=<your-key>`

## Quick reference

| Item | Value |
|------|-------|
| Base URL | `https://api.web3.hanzo.ai/v1` |
| Auth header | `X-API-Key: ${HANZO_WEB3_API_KEY}` |
| Alt auth | URL path: `https://api.web3.hanzo.ai/v1/${HANZO_WEB3_API_KEY}/eth/mainnet` |
| WebSocket | `wss://ws.web3.hanzo.ai/v1/${HANZO_WEB3_API_KEY}` |
| Status | https://status.web3.hanzo.ai |
| Docs | https://web3.hanzo.ai/docs |
| Dashboard | https://web3.hanzo.ai/dashboard |

## Network names

| Chain | Network ID | Mainnet | Testnet |
|-------|-----------|---------|---------|
| Ethereum | eth | `eth/mainnet` | `eth/sepolia`, `eth/holesky` |
| Base | base | `base/mainnet` | `base/sepolia` |
| Arbitrum | arb | `arb/mainnet` | `arb/sepolia` |
| Polygon | polygon | `polygon/mainnet` | `polygon/amoy` |
| Optimism | opt | `opt/mainnet` | `opt/sepolia` |
| Solana | sol | `sol/mainnet` | `sol/devnet` |
| Lux | lux | `lux/mainnet` | `lux/testnet` |
| Avalanche | avax | `avax/mainnet` | `avax/fuji` |
| BNB Chain | bsc | `bsc/mainnet` | `bsc/testnet` |
| 90+ more | ... | See docs | See docs |

## One-file quickstart

### Get latest block number (Ethereum)

```bash
curl -X POST https://api.web3.hanzo.ai/v1/eth/mainnet \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${HANZO_WEB3_API_KEY}" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

### Get token balances

```bash
curl "https://api.web3.hanzo.ai/v1/data/tokens/balances?address=0x...&network=eth/mainnet" \
  -H "X-API-Key: ${HANZO_WEB3_API_KEY}"
```

### Get NFT metadata

```bash
curl "https://api.web3.hanzo.ai/v1/data/nfts?owner=0x...&network=eth/mainnet" \
  -H "X-API-Key: ${HANZO_WEB3_API_KEY}"
```

### Create webhook

```bash
curl -X POST https://api.web3.hanzo.ai/v1/webhooks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${HANZO_WEB3_API_KEY}" \
  -d '{
    "url": "https://your-server.com/webhook",
    "network": "eth/mainnet",
    "event_type": "address_activity",
    "addresses": ["0x..."]
  }'
```

### Python (with Hanzo SDK)

```python
from hanzo import Hanzo

client = Hanzo()  # uses HANZO_WEB3_API_KEY from env

# JSON-RPC
block = client.web3.rpc("eth/mainnet", method="eth_blockNumber")

# Structured data
balances = client.web3.tokens.balances(
    address="0x...",
    network="eth/mainnet"
)

# Webhooks
hook = client.web3.webhooks.create(
    url="https://your-server.com/webhook",
    network="eth/mainnet",
    event_type="address_activity",
    addresses=["0x..."]
)
```

## Endpoint selector

| Task | Endpoint | Method |
|------|----------|--------|
| JSON-RPC call | `POST /v1/{network}` | POST |
| Get token balances | `GET /v1/data/tokens/balances` | GET |
| Get token prices | `GET /v1/data/tokens/prices` | GET |
| Get token metadata | `GET /v1/data/tokens/metadata` | GET |
| Get NFTs by owner | `GET /v1/data/nfts` | GET |
| Get NFT metadata | `GET /v1/data/nfts/metadata` | GET |
| Get transfers | `GET /v1/data/transfers` | GET |
| Get transaction history | `GET /v1/data/transactions` | GET |
| Get portfolio | `GET /v1/data/portfolio` | GET |
| Create webhook | `POST /v1/webhooks` | POST |
| List webhooks | `GET /v1/webhooks` | GET |
| Delete webhook | `DELETE /v1/webhooks/{id}` | DELETE |
| WebSocket subscribe | `WS /v1/{api_key}` | WS |
| Smart wallet create | `POST /v1/wallets/create` | POST |
| Smart wallet execute | `POST /v1/wallets/execute` | POST |
| Gas estimate | `POST /v1/gas/estimate` | POST |
| Simulate transaction | `POST /v1/simulate` | POST |

## Skill map

### Node APIs
Core JSON-RPC access for 100+ chains with load balancing and automatic failover.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/{network}` | Standard JSON-RPC proxy (eth_call, eth_sendRawTransaction, etc.) |
| `POST /v1/{network}/debug` | Debug namespace (debug_traceTransaction, debug_traceCall) |
| `POST /v1/{network}/trace` | Trace namespace (trace_block, trace_transaction) |
| `WS /v1/{api_key}` | WebSocket for subscriptions (newHeads, logs, pendingTransactions) |

### Data APIs
Structured onchain data across tokens, NFTs, transfers, and portfolio views.

| Endpoint | Description |
|----------|-------------|
| `GET /v1/data/tokens/balances` | ERC-20 balances for any address |
| `GET /v1/data/tokens/prices` | Real-time token prices (USD, ETH, BTC) |
| `GET /v1/data/tokens/metadata` | Token name, symbol, decimals, logo |
| `GET /v1/data/nfts` | NFTs owned by address |
| `GET /v1/data/nfts/metadata` | NFT metadata, media, traits |
| `GET /v1/data/transfers` | Token and NFT transfer history |
| `GET /v1/data/transactions` | Full transaction history |
| `GET /v1/data/portfolio` | Aggregated portfolio across chains |

### Solana APIs
High-throughput Solana access with Yellowstone gRPC and DAS.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/sol/mainnet` | Solana JSON-RPC |
| `gRPC /v1/sol/yellowstone` | Yellowstone gRPC streams (accounts, transactions, blocks) |
| `POST /v1/sol/das` | Digital Asset Standard API (compressed NFTs, Metaplex) |
| `WS /v1/sol/ws` | Solana WebSocket subscriptions |

### Webhooks
Push-based notifications for onchain events with managed delivery.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/webhooks` | Create webhook subscription |
| `GET /v1/webhooks` | List active webhooks |
| `GET /v1/webhooks/{id}` | Get webhook details |
| `DELETE /v1/webhooks/{id}` | Delete webhook |
| `POST /v1/webhooks/{id}/test` | Send test event |

Supported event types: `address_activity`, `token_transfer`, `nft_transfer`, `new_block`, `contract_event`, `pending_transaction`.

### Smart Wallets
ERC-4337 account abstraction with gas sponsorship and social login.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/wallets/create` | Create smart wallet |
| `POST /v1/wallets/execute` | Execute user operation |
| `POST /v1/wallets/batch` | Batch multiple operations |
| `POST /v1/gas/estimate` | Estimate gas for user operation |
| `POST /v1/gas/sponsor` | Sponsor gas for user |

### Recipes
Practical integration patterns.

| Recipe | Description |
|--------|-------------|
| Token swap monitoring | Watch DEX swaps and notify on price impact |
| NFT mint tracker | Alert when new NFTs are minted in a collection |
| Wallet portfolio | Build multi-chain portfolio dashboard |
| Transaction relay | Submit and track transactions with retry logic |
| Airdrop checker | Check eligibility across protocols |

## MCP Integration

Expose Hanzo Web3 as MCP tools for agentic workflows:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'

const web3Tools: Tool[] = [
  {
    name: 'web3_rpc',
    description: 'Execute JSON-RPC call on any supported chain',
    parameters: {
      network: { type: 'string', required: true },
      method: { type: 'string', required: true },
      params: { type: 'array', default: [] }
    },
    async execute({ network, method, params }) {
      const res = await fetch(`https://api.web3.hanzo.ai/v1/${network}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': process.env.HANZO_WEB3_API_KEY
        },
        body: JSON.stringify({ jsonrpc: '2.0', method, params, id: 1 })
      })
      return await res.json()
    }
  },
  {
    name: 'web3_token_balances',
    description: 'Get token balances for an address',
    parameters: {
      address: { type: 'string', required: true },
      network: { type: 'string', default: 'eth/mainnet' }
    },
    async execute({ address, network }) {
      const res = await fetch(
        `https://api.web3.hanzo.ai/v1/data/tokens/balances?address=${address}&network=${network}`,
        { headers: { 'X-API-Key': process.env.HANZO_WEB3_API_KEY } }
      )
      return await res.json()
    }
  }
]
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check parameters |
| 401 | Unauthorized | Check API key |
| 429 | Rate limited | Exponential backoff (1s, 2s, 4s, 8s, max 30s) |
| 500 | Server error | Retry up to 3 times |
| 502/503 | Upstream error | Retry with different endpoint |

## Official links

- Dashboard: https://web3.hanzo.ai/dashboard
- Documentation: https://web3.hanzo.ai/docs
- API Reference: https://web3.hanzo.ai/docs/api
- Status: https://status.web3.hanzo.ai
- Hanzo AI: https://hanzo.ai
- Hanzo Bot: https://hanzo.bot

---

**Last Updated**: 2026-02-26
**Category**: Hanzo Ecosystem
**Related**: blockchain, web3, infrastructure
**Prerequisites**: HTTP/curl, blockchain basics
