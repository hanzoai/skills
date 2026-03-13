# How to Use AI with Hanzo and Lux

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-extension.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-dev.md`

## Overview

This guide covers setting up Hanzo's full AI development stack — extensions, CLI agents, MCP tools, Zen models, and skills for cross-stack work across Hanzo AI and Lux blockchain.

## 1. Install Hanzo Extension (All Platforms)

The Hanzo extension provides AI chat, code completion, and MCP tool access across all browsers and IDEs. Everything ships from a single monorepo (`github.com/hanzoai/extension` v1.8.0).

### Browser Extensions

| Browser | Install |
|---------|---------|
| Chrome | [Chrome Web Store](https://chrome.google.com/webstore) — search "Hanzo" |
| Firefox | [Firefox AMO](https://addons.mozilla.org) — search "Hanzo" |
| Safari | Available via macOS App Store |

### IDE Extensions

| IDE | Install |
|-----|---------|
| VS Code / Cursor | Marketplace → search `@hanzo/extension` |
| Open VSX (VSCodium) | Open VSX Registry → search "Hanzo" |
| JetBrains | JetBrains Marketplace → search "Hanzo" |

### What You Get

- AI chat sidebar in browser and IDE
- Code completion with Zen models
- MCP tool access (13 unified tools)
- Auth via hanzo.id (implicit OAuth2)
- Access to 435 models (28 Zen + 400+ third-party) via `api.hanzo.ai/v1/chat/completions`

### Configuration

After installing, sign in via hanzo.id. The extension authenticates using implicit OAuth2 flow — no API key management needed. LLM requests route through `api.hanzo.ai/v1/chat/completions`.

## 2. Hanzo Dev for CLI Coding

Hanzo Dev is a terminal-based AI coding agent for agentic development workflows.

### Install

```bash
# Via npm
npm install -g @hanzo/dev

# Or run directly
npx @hanzo/dev
```

### Usage

```bash
# Interactive chat mode
hanzo-dev chat

# Non-interactive task mode
hanzo-dev task "Add authentication middleware to the Express API"

# With specific Zen model
hanzo-dev --model zen-70b chat

# With MCP tool servers
hanzo-dev --mcp hanzo-node,hanzo-ui chat
```

### Key Commands

| Command | Purpose |
|---------|---------|
| `/plan` | Create implementation plan |
| `/code` | Generate code |
| `/solve` | Debug and fix issues |
| `/auto` | Autonomous mode |
| `/browser` | Browser automation |

## 3. MCP vs ZAP — Choosing Your Protocol

Two protocols for connecting AI agents to tools:

### Hanzo MCP (`@hanzo/mcp` v2.4.1)

Standard Model Context Protocol. Works with Claude Code, Cursor, and any MCP-compatible client.

```bash
# Install globally
npm install -g @hanzo/mcp

# Add to Claude Code
claude mcp add hanzo-mcp -- hanzo-mcp serve

# List available tools
hanzo-mcp list-tools
```

**13 HIP-0300 unified tools** — action-routed, so each tool handles multiple operations.

**Best for**: Claude Code, Cursor, any standard MCP client.

### ZAP (Zero-copy Agent Protocol)

Hanzo's native protocol. Uses significantly less memory than MCP because it avoids JSON serialization overhead — direct memory-mapped communication.

```bash
# Use ZAP when running hanzo-dev natively
hanzo-dev --protocol zap chat
```

**Best for**: Hanzo Dev native mode, intensive workflows, memory-constrained environments.

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Claude Code | MCP (only option) |
| Cursor IDE | MCP |
| Hanzo Dev (native) | ZAP (less memory, faster) |
| Heavy multi-agent workflows | ZAP |
| Third-party MCP clients | MCP |

## 4. Zen AI — Hanzo's Frontier Models

Zen is Hanzo's frontier model family built on MoDE (Mixture of Diverse Experts) architecture. Available from 600M to 480B parameters.

### Available Models

28 Zen models accessible via the LLM gateway at `api.hanzo.ai/v1`. Use in any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.hanzo.ai/v1",
    api_key="your-hanzo-api-key"
)

response = client.chat.completions.create(
    model="zen-70b",
    messages=[{"role": "user", "content": "Explain post-quantum cryptography"}]
)
```

### Local Inference

Run Zen models locally via Hanzo Engine or Hanzo Node:

```bash
# Via Hanzo Node (AI agent node)
hanzo-node start --model zen-8b

# Or via the engine directly
hanzo-engine serve --model zen-8b --port 8080
```

### Using with Hanzo Dev

```bash
# Use Zen models natively
hanzo-dev --model zen-70b chat

# Auto-route (picks best model for task)
hanzo-dev --auto-route chat
```

## 5. Install Skills for Cross-Stack Development

Skills provide progressive disclosure of Hanzo and Lux documentation — AI agents load only what they need, when they need it.

### Install Hanzo Skills (49 skills)

```bash
git clone https://github.com/hanzoai/skills.git
```

Covers: AI infrastructure, cloud services, platform, SDKs (Python/JS/Go/Rust), identity, security, developer tools, ML, data, observability.

### Install Lux Skills (31 skills)

```bash
git clone https://github.com/luxfi/skills.git
```

Covers: blockchain node, Quasar consensus, EVM, DEX (CLOB), exchange (AMM), cryptography (BLS/FHE/lattice), wallet, bridge, monitoring, governance.

### Configure in Claude Code

Add to your project's `CLAUDE.md`:

```markdown
## Skills

Load Hanzo skills for AI infrastructure context:
cat path/to/hanzo/skills/skills/hanzo/INDEX.md

Load Lux skills for blockchain context:
cat path/to/lux/skills/skills/lux/INDEX.md
```

### Progressive Disclosure

```
Tier 1: Gateway (auto-discovers relevant skills)
  → discover-hanzo/SKILL.md or discover-lux/SKILL.md

Tier 2: INDEX (full catalog with decision tree)
  → skills/hanzo/INDEX.md or skills/lux/INDEX.md

Tier 3: Individual skills (deep technical detail)
  → skills/hanzo/hanzo-mcp.md, skills/lux/lux-consensus.md, etc.
```

## 6. Cross-Stack Workflow Examples

### Example 1: AI-Powered DeFi Dashboard

```
Skills needed:
  Hanzo: hanzo-chat.md, hanzo-ui.md, python-sdk.md
  Lux: lux-exchange.md, lux-dex.md, lux-evm.md

Flow:
1. Use Zen model to analyze market data (python-sdk)
2. Query Lux DEX CLOB for order book (lux-dex API)
3. Display in Hanzo UI components (hanzo-ui)
4. Deploy to Hanzo Platform (hanzo-platform)
```

### Example 2: Privacy-First Blockchain Analytics

```
Skills needed:
  Hanzo: hanzo-search.md, hanzo-engine.md
  Lux: lux-explorer.md, lux-fhe.md, lux-evm.md

Flow:
1. Index blockchain data with Hanzo Search (Meilisearch, Rust)
2. Run analytics on encrypted data via Lux FHE
3. Serve results via local inference (hanzo-engine)
4. Query via Lux Explorer (Blockscout)
```

### Example 3: Multi-Agent Trading Bot

```
Skills needed:
  Hanzo: hanzo-agent.md, hanzo-mcp.md, hanzo-kms.md
  Lux: lux-dex.md, lux-bridge.md, lux-oracle.md

Flow:
1. Agent monitors oracle prices (lux-oracle)
2. Routes orders to CLOB engine (lux-dex, 434M orders/sec)
3. Cross-chain settlement via bridge (lux-bridge, MPC)
4. API keys secured in KMS (hanzo-kms, 8 Vault subsystems)
```

## 7. Quick Start Checklist

1. **Sign up** at hanzo.ai for API key
2. **Install browser extension** (Chrome/Firefox/Safari)
3. **Install IDE extension** (VS Code/JetBrains)
4. **Install Hanzo Dev CLI**: `npm install -g @hanzo/dev`
5. **Install MCP tools**: `npm install -g @hanzo/mcp`
6. **Clone Hanzo skills**: `git clone https://github.com/hanzoai/skills.git`
7. **Clone Lux skills**: `git clone https://github.com/luxfi/skills.git`
8. **Configure Claude Code** (if using): add MCP server via `claude mcp add`
9. **Try a Zen model**: `hanzo-dev --model zen-70b chat`
10. **Explore skills**: `cat skills/hanzo/INDEX.md` or `cat skills/lux/INDEX.md`

## Key Endpoints

| Service | URL |
|---------|-----|
| LLM API | `api.hanzo.ai/v1/chat/completions` |
| Auth | `hanzo.id` |
| Chat UI | `chat.hanzo.ai` |
| Cloud | `cloud.hanzo.ai` |
| Console | `console.hanzo.ai` |
| KMS | `kms.hanzo.ai` |
| Platform | `platform.hanzo.ai` |
| Search | `search.hanzo.ai` |
| Studio | `studio.hanzo.ai` |
| Lux Gateway | `api.lux.network` |
| Lux Explorer | `explorer.lux.network` |
| Lux Monitor | `monitor.lux.network` |

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: getting-started, setup, ai, development
**Prerequisites**: Node.js 18+, npm/pnpm
