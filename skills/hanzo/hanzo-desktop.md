# Hanzo Desktop - AI-Powered Productivity & Mining App

**Category**: Hanzo Ecosystem
**Difficulty**: Beginner
**Prerequisites**: macOS 12+
**Related Skills**: zenlm.md, hanzo-node.md, python-sdk.md

## Overview

**Hanzo Desktop** is an AI-powered command palette and local AI assistant for macOS that merges powerful app launcher capabilities with advanced local AI features. Built on Tauri with embedded Rust high-performance engine, it's the ultimate productivity tool with **AI mining capabilities**.

**Key Features**:
- 🚀 **Native Rust Engine**: Blazingly fast LLM & embedding inference (Hanzo Engine)
- 💰 **Mine $AI**: Earn $AI tokens by sharing compute with ngrok/localxpose
- 🔗 **Hanzo Node Integrated**: Full node capabilities in desktop app
- 🤖 **Local AI Assistant**: Powered by ZenLM models (zen-nano, zen-eco)
- ⚡ **Instant Access**: Press Tab to access AI from anywhere
- 🔒 **100% Private**: All AI processing happens on your device
- 💚 **Support Open Source**: Collect payments for compute powering your projects

## Installation

### Via Homebrew (Recommended)

```bash
brew install --cask hanzo
```

### Manual Download

Download the latest release:
```bash
# Visit https://github.com/hanzoai/app/releases
# Download Hanzo-Desktop-latest.dmg
# Drag to Applications folder
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/hanzoai/app-v2.git
cd app-v2/apps/desktop

# Install dependencies
mise install

# Build desktop app
bun run build:desktop

# Run locally
bun desktop
```

## Quick Start

### First Launch

1. **Open Hanzo**: Press `Tab` (default hotkey)
2. **Choose Mode**:
   - AI Chat (Tab) - Instant AI assistance
   - App Launcher (Cmd+Space alternative)
   - Command Palette (productivity tools)

### AI Assistant (Zen Mode)

```
# Press Tab anywhere → AI chat opens

You: "Explain quantum computing in simple terms"

Zen: [Powered by zen-nano-instruct running locally]
"Quantum computing uses quantum mechanics principles..."

# Context-aware assistance
You: "Write a Python function to reverse a string"
Zen: [Generates code with explanation]

# Natural language to action
You: "Open Chrome and search for Hanzo AI"
Zen: [Executes AppleScript to open browser and search]
```

## Mining $AI Tokens

### What is AI Mining?

**Mine $AI** by sharing your idle compute power with the Hanzo Network. Your Mac becomes a node that serves AI inference requests, earning you $AI tokens.

**How it Works**:
1. You expose your local Hanzo Desktop to the internet via ngrok or localxpose
2. The Hanzo Network routes AI inference requests to your machine
3. You earn $AI tokens for each inference served
4. Open source developers collect payments for compute powering their projects

### Setup Mining (ngrok)

```bash
# 1. Install ngrok
brew install ngrok

# 2. Get ngrok authtoken (free tier: https://dashboard.ngrok.com)
ngrok config add-authtoken YOUR_TOKEN

# 3. Configure Hanzo Desktop
# Open Hanzo Settings → Mining → Enable Mining
# Select "ngrok" as tunnel provider
# Enter your ngrok authtoken

# 4. Start mining
# Hanzo automatically starts ngrok tunnel on port 36900 (Hanzo Engine)
# Your mining address: https://YOUR-TUNNEL.ngrok.io

# 5. Register mining node
# Hanzo Desktop automatically registers with Hanzo Network
# Mining dashboard: https://hanzo.ai/mining
```

### Setup Mining (localxpose)

```bash
# 1. Install localxpose
brew install localxpose

# 2. Get access token (free tier: https://localxpose.io)
loclx account login

# 3. Configure Hanzo Desktop
# Open Hanzo Settings → Mining → Enable Mining
# Select "localxpose" as tunnel provider
# Enter your localxpose token

# 4. Start mining
# Hanzo creates tunnel: https://YOUR-TUNNEL.loclx.io
# Mining active on port 36900

# 5. Monitor earnings
# Real-time dashboard in Hanzo Desktop
# Track: requests served, $AI earned, uptime
```

### Mining Dashboard

```
┌─────────────────────────────────────┐
│      Hanzo Mining Dashboard         │
├─────────────────────────────────────┤
│ Status:        ✅ Mining Active     │
│ Tunnel:        ngrok (authenticated)│
│ Address:       xyz.ngrok.io:36900  │
│                                     │
│ Today's Stats:                      │
│  • Requests:   1,247                │
│  • $AI Earned: 12.47 ($3.74)       │
│  • Uptime:     97.3%                │
│                                     │
│ All-Time:                           │
│  • Total $AI:  1,842.50 ($552.75)  │
│  • Requests:   184,250              │
│  • Since:      Jan 15, 2025         │
└─────────────────────────────────────┘
```

### Mining Rewards

**Payout Structure**:
- **Inference Request**: 0.01 $AI (~$0.003)
- **Embedding Generation**: 0.001 $AI (~$0.0003)
- **Function Call**: 0.005 $AI (~$0.0015)

**Performance Bonuses**:
- **<100ms latency**: +20% bonus
- **99% uptime**: +10% bonus
- **First 1000 requests**: +50% bonus

**Example Earnings** (M1 Pro, 8-hour mining/day):
```
Daily:   ~600 requests  = 6 $AI    = $1.80
Weekly:  ~4,200 requests = 42 $AI   = $12.60
Monthly: ~18,000 requests = 180 $AI = $54.00
Yearly:  ~216,000 requests = 2,160 $AI = $648.00
```

**Electricity Cost** (M1 Pro at 15W):
```
Daily:   8 hours × 15W  = 0.12 kWh = $0.02
Monthly: 3.6 kWh        = $0.60
Yearly:  43.2 kWh       = $7.20

Net yearly: $648 - $7.20 = $640.80
```

## Open Source Compute Payments

### For Open Source Authors

**Link Your Project** to collect payments for compute used to power it:

```bash
# 1. Register your OSS project
hanzo register-project \
    --name "my-awesome-app" \
    --repo "github.com/user/my-awesome-app" \
    --wallet "0xYourEthereumAddress"

# 2. Generate project API key
hanzo project-key create --project my-awesome-app

# 3. Users point their apps at your project
export HANZO_PROJECT_KEY="proj_abc123..."
export HANZO_PROJECT_NAME="my-awesome-app"

# 4. Collect payments automatically
# Every inference request routes payment to your wallet
# 80% to you, 20% to miners
```

### Payment Flow

```
User Request → Hanzo Network → Miner (you)
    ↓              ↓              ↓
   Free      Payment Split:   0.01 $AI
              ├─ 80% → OSS Dev  (0.008 $AI)
              └─ 20% → Miner    (0.002 $AI)
```

### Example: OSS Project Earnings

**Project**: "hanzo-chatbot" (1000 users, 10K requests/day)

```
Daily:   10,000 requests × 0.008 $AI = 80 $AI   = $24
Weekly:  70,000 requests             = 560 $AI  = $168
Monthly: 300,000 requests            = 2,400 $AI = $720
Yearly:  3.6M requests               = 28,800 $AI = $8,640
```

**Your users get free AI, you get paid, miners earn** - everyone wins!

### Integrate in Your App

```python
from hanzo import Hanzo

# Initialize with your project key
hanzo = Hanzo(
    project_key="proj_abc123...",
    project_name="my-awesome-app",
    inference_mode='network'  # Use Hanzo Network miners
)

# Every request supports your project
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

# 80% of inference cost goes to you (OSS author)
# 20% goes to miner serving the request
# User pays nothing
```

## Native Rust Engine (Hanzo Engine)

### Architecture

```
┌─────────────────────────────────────┐
│    Hanzo Desktop (Tauri/React)      │
│      (Command Palette + UI)         │
└────────────┬────────────────────────┘
             │ IPC
             ▼
┌─────────────────────────────────────┐
│    Hanzo Engine (Rust - Port 36900) │
│   (High-Performance Inference)      │
├─────────────────────────────────────┤
│  • Embedding Generation (Qwen3-8B)  │
│  • LLM Inference (zen-nano/eco)     │
│  • Reranking (Qwen3-Reranker)       │
│  • Model Management                 │
└─────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│      Hanzo Node (Port 3690)         │
│    (AI Agent Infrastructure)        │
└─────────────────────────────────────┘
```

### Engine Performance

**Apple Silicon (M1 Pro)**:
- **Embedding Generation**: 44K+ tokens/second
- **LLM Inference**: 50+ tokens/second
- **Startup Time**: <2 seconds
- **Memory Usage**: 2-4GB (depending on model)

**Intel Mac (i9-9900K)**:
- **Embedding Generation**: 15K+ tokens/second
- **LLM Inference**: 30+ tokens/second
- **Startup Time**: <3 seconds
- **Memory Usage**: 3-5GB

### Models Supported

#### Embedding Models (Optimized)
- **Qwen3-Embedding-8B**: 4096 dims, #1 on MTEB multilingual
- **Qwen3-Embedding-4B**: 2048 dims, balanced
- **Qwen3-Embedding-0.6B**: 1024 dims, lightweight

#### LLM Models
- **zen-nano-instruct**: 4B, ultra-fast
- **zen-nano-thinking**: 4B, reasoning
- **zen-eco**: 4B, production-grade
- All models supported by mistral.rs

### Engine API

```bash
# Generate embeddings
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-8b",
    "input": "Hello, Hanzo Engine!"
  }'

# Chat completion
curl -X POST http://localhost:36900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-nano-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Hanzo Node Integration

### What is Hanzo Node?

Hanzo Node is the full AI agent infrastructure embedded in Hanzo Desktop:

- **MCP Server**: Model Context Protocol for tools/resources
- **Agent Runtime**: Run multi-step AI workflows
- **Vector Database**: Local semantic search
- **Workflow Engine**: Chain multiple AI operations

### Using Hanzo Node

```bash
# Hanzo Node runs on port 3690 (auto-starts with Desktop)

# Check status
curl http://localhost:3690/health

# Agent endpoint
curl -X POST http://localhost:3690/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Research AI mining profitability",
    "tools": ["search", "calculator", "web_scrape"]
  }'
```

### MCP Tools Available

- **File System**: Read/write local files
- **Web Browser**: Search and scrape
- **Code Execution**: Run Python/Node.js
- **Database**: SQLite queries
- **Calendar**: Manage events
- **Email**: Send/receive (with permission)

## Features

### AI Assistant (Zen)

**Press Tab** → Instant AI chat

```
# Code generation
You: "Write a Rust function to parse JSON"
Zen: [Generates optimized Rust code]

# Debugging
You: "Why is my Python script crashing?"
Zen: [Analyzes error and suggests fix]

# Natural language queries
You: "What's the capital of France?"
Zen: "Paris"

# Context-aware
You: "Summarize my last 3 commits"
Zen: [Reads git log and summarizes]
```

### App Launcher

**Cmd+Space alternative** with AI search:

```
# Type app name → Launch
"chrome" → Opens Google Chrome

# AI-powered search
"browser for testing" → Suggests Chrome, Firefox, Safari

# Recent apps prioritized
"code" → Opens VS Code (you use it daily)
```

### Productivity Tools

- **Google Translate**: Translate text instantly
- **Calendar Integration**: View upcoming appointments
- **Custom AppleScript**: Run custom commands
- **Browser Bookmarks**: Import and search
- **Window Manager**: Tile windows
- **Emoji Picker**: Quick emoji access
- **Clipboard Manager**: History and search
- **Notes Scratchpad**: Quick notes

### Utilities

- **Wi-Fi Password**: Retrieve current network password
- **IP Address**: Show public/private IP
- **Google Meet**: Start instant meeting
- **Theme Switcher**: Toggle light/dark mode
- **Process Killer**: Force quit apps
- **XCode Cleanup**: Clear derived data
- **ID Generator**: NanoID/UUID
- **Lorem Ipsum**: Generate placeholder text
- **JSON Formatter**: Format and paste JSON
- **Math Eval**: Quick calculations

## Configuration

### Settings

```yaml
# ~/.hanzo/desktop/config.yaml

general:
  hotkey: "Tab"
  theme: "auto"  # auto, light, dark
  startup: true

ai:
  model: "zen-nano-instruct"
  temperature: 0.7
  max_tokens: 2048
  context_window: 8192

mining:
  enabled: true
  tunnel: "ngrok"  # ngrok or localxpose
  port: 36900
  auto_register: true

engine:
  port: 36900
  models:
    - "qwen3-embedding-8b"
    - "zen-nano-instruct"
  quantization: "q4_k_m"

node:
  port: 3690
  tools_enabled: true
  mcp_servers:
    - "file-system"
    - "web-browser"
```

### Hotkey Customization

```
Preferences → General → Hotkey
Default: Tab
Alternatives: Cmd+Space, Ctrl+Space, Opt+Space
```

### Model Management

```bash
# List installed models
hanzo-engine list

# Pull new model
hanzo-engine pull zen-eco

# Delete model
hanzo-engine delete zen-nano-thinking

# Set default
hanzo-engine default zen-eco
```

## Comparison: Traditional Launcher vs Hanzo Desktop

### Alfred/Raycast (Traditional)

```
# Limited to pre-defined commands
Type: "weather san francisco" → Runs specific workflow
Type: "calculate tip" → Opens calculator

# No AI understanding
Type: "find that Python file I edited yesterday" → No results

# Cloud API required for AI
Monthly cost: $20-40
Privacy: Data sent to cloud
```

### Hanzo Desktop (AI-Powered)

```
# Natural language understanding
Type: "what's the weather like?" → Fetches SF weather
Type: "help me calculate 18% tip on $85" → Instant answer

# AI-powered search
Type: "Python file from yesterday" → Finds file intelligently

# Local AI + Mining
Monthly cost: -$54 (you earn money!)
Privacy: 100% local processing
```

**Result**: Hanzo Desktop is **smarter**, **more private**, and **pays you** to use it.

## Privacy & Security

### Data Privacy

- **100% Local Processing**: All AI runs on your Mac
- **No Telemetry**: Zero tracking or analytics
- **No Cloud Sync**: Your data stays on your device
- **Open Source**: Auditable code

### Mining Security

- **Sandboxed Inference**: Miners can't access your data
- **Rate Limiting**: Prevent abuse
- **Wallet Security**: Private keys never shared
- **Tunnel Authentication**: Secure ngrok/localxpose

### Permissions

```
# Hanzo Desktop requests:
✅ Accessibility (for global hotkey)
✅ Screen Recording (for window manager)
✅ Automation (for AppleScript)

# Hanzo Mining requires:
✅ Network access (for tunnel)
✅ Port 36900 open (for inference)
```

## Troubleshooting

### Mining Not Starting

```bash
# Check tunnel status
hanzo mining status

# Restart tunnel
hanzo mining restart

# Check logs
tail -f ~/.hanzo/desktop/logs/mining.log

# Common issues:
# 1. ngrok authtoken invalid → Regenerate at ngrok.com
# 2. Port 36900 blocked → Check firewall settings
# 3. Low uptime → Improve network stability
```

### Engine Not Responding

```bash
# Check engine status
curl http://localhost:36900/health

# Restart engine
hanzo-engine restart

# Check logs
tail -f ~/.hanzo/engine/logs/engine.log

# Common issues:
# 1. Model not loaded → hanzo-engine pull <model>
# 2. Out of memory → Close other apps
# 3. Port conflict → Change port in config
```

### AI Responses Slow

```bash
# Check model quantization
hanzo-engine list

# Switch to lighter model
hanzo-engine default zen-nano-instruct-4bit

# Enable performance mode
# Preferences → AI → Performance Mode: Enabled
```

## Related Skills

- **zenlm.md**: Learn about ZenLM models powering Hanzo Desktop
- **hanzo-node.md**: Deep dive into Hanzo Node capabilities
- **python-sdk.md**: Build apps using Hanzo Desktop API
- **hanzo-gym.md**: Train custom models for Hanzo Desktop

## Additional Resources

- **Website**: [hanzo.app](https://hanzo.app)
- **GitHub**: [github.com/hanzoai/app-v2](https://github.com/hanzoai/app-v2)
- **Engine Repo**: [github.com/hanzoai/engine](https://github.com/hanzoai/engine)
- **Discord**: [discord.gg/hanzoai](https://discord.gg/hanzoai)
- **Mining Dashboard**: [hanzo.ai/mining](https://hanzo.ai/mining)
- **Documentation**: [docs.hanzo.ai/desktop](https://docs.hanzo.ai/desktop)

---

**Hanzo Desktop** - AI-powered productivity that pays you.

**Mine $AI while you sleep. Support open source. Own your AI.**

© 2025 Hanzo Industries Inc • MIT License