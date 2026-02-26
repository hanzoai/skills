# Hanzo Web3 Gateway - Keyless Blockchain Access via x402

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-web3.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-commerce.md`

## Overview

Hanzo Web3 Gateway provides **keyless blockchain access** for autonomous AI agents. Instead of API keys, agents pay per request with USDC micropayments on Base using the x402 protocol. No account, no signup, no billing dashboard -- just a wallet with USDC.

### Why the Gateway?

- **Zero configuration**: No API key, no account, no billing
- **Agent-native**: Agents autonomously sign up and pay
- **x402 standard**: Open payment protocol (EIP-712 signed USDC on Base)
- **Same coverage**: All 100+ chains available through Hanzo Web3
- **Micropayments**: Sub-cent pricing per request ($0.000001 - $0.0001)

## When to use

Use this skill when:
- The agent needs keyless blockchain access without a Hanzo account
- The agent should pay per request with USDC (x402 protocol)
- Building autonomous agents that sign up and pay independently
- No human needs to create an API key or manage billing

If `HANZO_WEB3_API_KEY` is set, use the `hanzo-web3` skill instead -- it is cheaper for high volume.

## Hard requirements

1. **Wallet required.** `HANZO_WALLET_PRIVATE_KEY` must be set to an Ethereum private key that holds USDC on Base.
2. **USDC on Base.** The wallet must have USDC balance on Base network for x402 payments.
3. **Never expose the private key** in user-visible output, logs, or screenshots.
4. **If `HANZO_WEB3_API_KEY` is set**, prefer the `hanzo-web3` skill (it is cheaper for high volume).

## Preflight checks

Before making any request, silently verify:
1. `HANZO_WALLET_PRIVATE_KEY` is set and non-empty
2. If `HANZO_WEB3_API_KEY` is also set, warn that `hanzo-web3` skill may be more cost-effective

## How x402 works

1. Agent sends request to gateway without auth
2. Gateway responds with `402 Payment Required` + payment details
3. Agent signs USDC payment authorization (EIP-712)
4. Agent resends request with `X-Payment` header containing signed authorization
5. Gateway verifies signature, deducts USDC, processes request
6. Response returned as normal

## Quick reference

| Item | Value |
|------|-------|
| Gateway URL | `https://gateway.web3.hanzo.ai/v1` |
| Payment token | USDC on Base (0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913) |
| Payment protocol | x402 (EIP-712 signed authorization) |
| Auth method | SIWE (Sign-In with Ethereum) |
| Docs | https://web3.hanzo.ai/docs/agents |
| Status | https://status.web3.hanzo.ai |

## One-file quickstart

### 1. Generate a wallet (if needed)

```bash
# Using cast (foundry)
cast wallet new

# Save the private key
export HANZO_WALLET_PRIVATE_KEY=0x...
```

### 2. Fund with USDC on Base

Transfer USDC to your wallet address on Base network. Even $1 covers thousands of requests.

### 3. Make a request

```bash
# First request gets 402 with payment details
curl -X POST https://gateway.web3.hanzo.ai/v1/eth/mainnet \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Response: 402 Payment Required with X-Payment-Details header
# Agent SDK handles payment signing automatically
```

### Using the Hanzo Agent SDK

```typescript
import { HanzoWeb3Agent } from "@hanzo/web3-agent"

const agent = new HanzoWeb3Agent({
  privateKey: process.env.HANZO_WALLET_PRIVATE_KEY,
})

// Automatically handles x402 payment flow
const blockNumber = await agent.rpc("eth/mainnet", {
  method: "eth_blockNumber",
  params: [],
})
```

### Python

```python
from hanzo.web3 import Web3Agent

agent = Web3Agent()  # uses HANZO_WALLET_PRIVATE_KEY from env

# Automatic x402 payment handling
block = agent.rpc("eth/mainnet", method="eth_blockNumber")
balances = agent.tokens.balances(address="0x...", network="eth/mainnet")
```

## Pricing

| Request type | Cost (USDC) |
|-------------|-------------|
| Simple query (eth_blockNumber, eth_gasPrice) | $0.000001 |
| Standard query (eth_getBalance, eth_getBlock) | $0.000005 |
| Compute query (eth_call, eth_estimateGas) | $0.00001 |
| Log query (eth_getLogs) | $0.000025 |
| Write (eth_sendRawTransaction) | $0.00005 |
| Debug/Trace | $0.0001 |
| Data API (tokens, NFTs) | $0.00001 |

## Endpoint selector

Same endpoints as `hanzo-web3` skill, but using gateway URL:

| Task | Endpoint |
|------|----------|
| JSON-RPC | `POST https://gateway.web3.hanzo.ai/v1/{network}` |
| Token data | `GET https://gateway.web3.hanzo.ai/v1/data/tokens/*` |
| NFT data | `GET https://gateway.web3.hanzo.ai/v1/data/nfts/*` |
| Portfolio | `GET https://gateway.web3.hanzo.ai/v1/data/portfolio` |
| Transfers | `GET https://gateway.web3.hanzo.ai/v1/data/transfers` |
| Transactions | `GET https://gateway.web3.hanzo.ai/v1/data/transactions` |
| Webhooks | `POST https://gateway.web3.hanzo.ai/v1/webhooks` |
| Smart wallets | `POST https://gateway.web3.hanzo.ai/v1/wallets/*` |

## MCP Integration

Expose keyless blockchain access as MCP tools for autonomous agents:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'
import { HanzoWeb3Agent } from '@hanzo/web3-agent'

const agent = new HanzoWeb3Agent({
  privateKey: process.env.HANZO_WALLET_PRIVATE_KEY,
})

const gatewayTools: Tool[] = [
  {
    name: 'web3_gateway_rpc',
    description: 'Keyless JSON-RPC call via x402 micropayment',
    parameters: {
      network: { type: 'string', required: true },
      method: { type: 'string', required: true },
      params: { type: 'array', default: [] }
    },
    async execute({ network, method, params }) {
      return await agent.rpc(network, { method, params })
    }
  },
  {
    name: 'web3_gateway_balance',
    description: 'Check USDC balance available for gateway payments',
    parameters: {},
    async execute() {
      return await agent.getBalance()
    }
  }
]
```

## Error handling

| Code | Meaning | Action |
|------|---------|--------|
| 402 | Payment required | Sign payment and retry with X-Payment header |
| 401 | Invalid signature | Re-sign with correct wallet |
| 402 + insufficient | Wallet balance too low | Fund wallet with more USDC |
| 429 | Rate limited | Backoff and retry |
| 500 | Server error | Retry up to 3 times |

## Choosing between Web3 and Web3 Gateway

| Factor | hanzo-web3 | hanzo-web3-gateway |
|--------|-----------|-------------------|
| Auth | API key | Wallet + x402 |
| Setup | Dashboard signup | Just a wallet |
| Billing | Monthly/PAYG | Per-request USDC |
| Best for | High volume, teams | Autonomous agents |
| Rate limit | 25-300 req/s | Per-payment |
| Cost at scale | Lower | Higher |

## Official links

- Agent docs: https://web3.hanzo.ai/docs/agents
- x402 protocol: https://x402.org
- Dashboard: https://web3.hanzo.ai/dashboard
- Hanzo AI: https://hanzo.ai
- Hanzo Bot: https://hanzo.bot

---

**Last Updated**: 2026-02-26
**Category**: Hanzo Ecosystem
**Related**: blockchain, web3, agents, x402
**Prerequisites**: Ethereum wallet, USDC on Base, blockchain basics
