# Discover Hanzo - AI Infrastructure & Development Ecosystem

**Gateway Skill**: Auto-activates on keywords like "hanzo", "local ai", "mcp", "privacy ai", "agentic workflow"

## Overview

Hanzo is a comprehensive AI infrastructure ecosystem focused on **privacy-first, locally-run AI** with composable, production-ready abstractions. Rather than building AI systems line-by-line, Hanzo provides higher-level "legos" that integrate seamlessly for agentic workflows, distributed AI, and blockchain-powered applications.

### Core Philosophy

- **Local-First AI**: Run foundational models on your infrastructure for privacy and cost control
- **Composable Abstractions**: Use high-level SDKs and components instead of raw code
- **MCP-Native**: Built for Model Context Protocol to enable seamless agentic workflows
- **Open Source**: All core infrastructure is open and extensible

### Hanzo Product Stack

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer                           │
│  @hanzo/ui (React) │ Hanzo Dev (Terminal) │ Custom Apps │
└─────────────────────────────────────────────────────────┘
                    ↓ (MCP, HTTP, gRPC)
┌─────────────────────────────────────────────────────────┐
│           Integration & Orchestration Layer              │
│  Hanzo MCP (Agentic) │ Python SDK │ Ruby SDK │ Terraform│
└─────────────────────────────────────────────────────────┘
                    ↓ (Hanzo Protocol)
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                        │
│  Hanzo Node (Rust) - Mining, Inference, P2P Networking  │
│  Hanzo Live - Real-time Generative AI Pipelines         │
└─────────────────────────────────────────────────────────┘
```

## Quick Reference

### Hanzo Node (Rust)
**Purpose**: Distributed AI compute network for mining and local inference  
**Key Features**: GPU/CPU scheduling, P2P networking, consensus engine  
**Use When**: Running local models privately, joining AI network

```bash
cargo build --release
hanzo-node start --mine --gpu
hanzo-node status
```

### Hanzo MCP
**Purpose**: Model Context Protocol integration for agentic workflows  
**Key Features**: Tool orchestration, context sharing, multi-agent coordination  
**Use When**: Building AI agents, exposing capabilities via MCP

```typescript
import { MCPServer } from '@hanzo/mcp'

const server = new MCPServer({
  tools: [hanzoNodeTool, uiComponentTool],
  resources: [localAI, vectorDB]
})
```

### @hanzo/ui
**Purpose**: React component library for AI+Blockchain applications  
**Key Features**: Pre-built AI components, wallet integration, elegant design  
**Use When**: Building AI dashboards, blockchain UIs, agent interfaces

```bash
pnpm add @hanzo/ui
```

```typescript
import { AIChat, ModelSelector, WalletConnect } from '@hanzo/ui'

<AIChat model="local-llama" onMessage={handleMessage} />
```

### Python SDK
**Purpose**: Unified gateway to foundational models and AI cloud  
**Key Features**: Multi-provider support, local inference, automatic routing  
**Use When**: Integrating AI into Python applications

```python
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='local')  # or 'cloud', 'hybrid'
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{'role': 'user', 'content': 'Explain Rust'}]
)
```

### Hanzo Dev
**Purpose**: Terminal-based AI coding agent (like Claude Code, but CLI)  
**Key Features**: MCP integration, local AI, project understanding  
**Use When**: Agentic coding workflows, terminal-based development

```bash
hanzo-dev chat
hanzo-dev task "Add authentication to API"
hanzo-dev --mcp hanzo-node,hanzo-ui
```

### Hanzo Live
**Purpose**: Real-time generative AI pipeline execution  
**Key Features**: Streaming inference, pipeline composition, live updates  
**Use When**: Building real-time AI features, streaming generation

### Additional Tools
- **Hanzo Chat**: AI conversation interface
- **Hanzo Desktop**: Desktop application for AI workflows
- **Ruby SDK**: Ruby language support
- **Terraform Provider**: Infrastructure-as-code for Hanzo deployments
- **Skills Repository**: This repository - contextual AI workflows

## When to Use Hanzo

### ✅ Perfect For

**Privacy-First AI**
- Healthcare, finance, legal applications with sensitive data
- GDPR, HIPAA, SOC 2 compliance requirements
- No data leaves your infrastructure

**Local AI Deployment**
- On-premise inference with consumer GPUs
- Edge deployment on devices
- No API costs, no rate limits

**Agentic Development**
- Building multi-agent workflows with MCP
- Tool orchestration across AI capabilities
- Terminal-based coding agents

**AI+Blockchain Integration**
- DeFi with AI features
- NFT platforms with AI generation
- DAO governance with AI analysis

**Distributed AI Compute**
- Joining mining network for inference
- Load balancing across nodes
- High-availability AI services

### 🔧 Hanzo's Higher-Level Abstractions

**Instead of writing:**
```python
# Manual model routing
if complexity == "high":
    model = "gpt-4"
elif cost_sensitive:
    model = "llama-3-70b-local"
else:
    model = "claude-sonnet"

response = requests.post(f"{model_endpoint}/chat", ...)
```

**Use Hanzo:**
```python
# Automatic routing with cost optimization
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='hybrid')
response = hanzo.chat.completions.create(
    messages=[...],
    auto_route=True  # Handles complexity, cost, latency
)
```

**Instead of:**
```typescript
// Manual UI for AI chat
const [messages, setMessages] = useState([])
const [loading, setLoading] = useState(false)
// 200+ lines of chat logic, streaming, error handling...
```

**Use Hanzo:**
```typescript
// Production-ready AI chat component
import { AIChat } from '@hanzo/ui'

<AIChat 
  model="local-llama"
  onMessage={handleMessage}
  streaming
  darkMode
/>
```

## Loading Skills by Category

### Core AI Infrastructure
```bash
cat skills/hanzo/hanzo-engine.md        # Rust inference engine
cat skills/hanzo/hanzo-node.md          # Distributed AI node
cat skills/hanzo/hanzo-network.md       # AI agent topology
cat skills/hanzo/hanzo-llm-gateway.md   # Unified LLM proxy (100+ providers)
cat skills/hanzo/hanzo-mcp.md           # Model Context Protocol tools
cat skills/hanzo/hanzo-agent.md         # Multi-agent SDK
```

### Cloud Services
```bash
cat skills/hanzo/hanzo-chat.md          # Unified LLM API (86+ models)
cat skills/hanzo/hanzo-cloud.md         # Cloud dashboard & management
cat skills/hanzo/hanzo-console.md       # AI observability & tracing
cat skills/hanzo/hanzo-commerce-api.md  # Billing & payments
cat skills/hanzo/hanzo-web3.md          # Blockchain API (100+ chains)
```

### Platform & Deployment
```bash
cat skills/hanzo/hanzo-platform.md      # PaaS (Agnost fork)
cat skills/hanzo/hanzo-studio.md        # Visual AI workflows (ComfyUI fork)
cat skills/hanzo/hanzo-search.md        # AI-powered search
cat skills/hanzo/hanzo-flow.md          # Visual workflow builder
cat skills/hanzo/hanzo-stack.md         # Local dev environment
cat skills/hanzo/hanzo-universe.md      # Production K8s infrastructure
```

### SDKs & Libraries
```bash
cat skills/hanzo/python-sdk.md          # Python SDK
cat skills/hanzo/js-sdk.md             # TypeScript SDK
cat skills/hanzo/go-sdk.md             # Go SDK
cat skills/hanzo/rust-sdk.md           # Rust SDK
cat skills/hanzo/hanzo-orm.md          # Go generics ORM
```

### Identity & Security
```bash
cat skills/hanzo/hanzo-id.md           # Identity & OAuth2/OIDC
cat skills/hanzo/hanzo-kms.md          # Secret management (Infisical)
cat skills/hanzo/hanzo-vault.md        # PCI card tokenization
```

### Developer Tools
```bash
cat skills/hanzo/hanzo-extension.md    # Browser & IDE extensions
cat skills/hanzo/hanzo-cli.md          # Command-line interface
cat skills/hanzo/hanzo-operative.md    # Computer use agent
cat skills/hanzo/hanzo-dev.md          # Terminal coding agent
```

### AI/ML & Research
```bash
cat skills/hanzo/zenlm.md             # Zen frontier models
cat skills/hanzo/hanzo-jin.md          # Multimodal LLM (text/vision/audio)
cat skills/hanzo/hanzo-candle.md       # Rust ML framework
cat skills/hanzo/hanzo-ane.md          # Apple Neural Engine training
cat skills/hanzo/hanzo-evm.md          # Rust EVM execution engine
```

### Data & Observability
```bash
cat skills/hanzo/hanzo-database.md     # PostgreSQL, Redis, pgvector
cat skills/hanzo/hanzo-datastore.md    # Vector database
cat skills/hanzo/hanzo-o11y.md         # Monitoring, tracing, errors
cat skills/hanzo/hanzo-insights.md     # Analytics SDKs
```

### Browse All Hanzo Skills (50+)
```bash
cat skills/hanzo/INDEX.md
```

## Integration Patterns

### Pattern 1: Local AI with Privacy
```python
from hanzo import Hanzo

# All inference happens locally - no external API calls
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'
)

# Process sensitive data privately
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{
        'role': 'user', 
        'content': sensitive_patient_data
    }]
)
```

### Pattern 2: MCP-Powered Agentic Workflows
```bash
# Configure MCP servers
hanzo-dev config mcp add hanzo-node http://localhost:8080
hanzo-dev config mcp add hanzo-ui http://localhost:3000

# Execute multi-step workflow with MCP tools
hanzo-dev workflow "
  1. Generate component using @hanzo/ui tool
  2. Deploy model to Hanzo Node
  3. Run integration tests
  4. Update documentation
"
```

### Pattern 3: Real-Time AI UI
```typescript
import { AIChat, useHanzoLive } from '@hanzo/ui'

export function LiveAIDashboard() {
  // Real-time pipeline updates
  const { data, status } = useHanzoLive({
    pipeline: 'text-generation-stream'
  })
  
  // Local inference via Hanzo Node
  return (
    <AIChat 
      inference="local"
      model="llama-3-8b"
      streaming
      live
    />
  )
}
```

### Pattern 4: Distributed AI Cluster
```yaml
# kubernetes/hanzo-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hanzo-node-cluster
spec:
  serviceName: hanzo-nodes
  replicas: 3
  template:
    spec:
      containers:
      - name: hanzo-node
        image: hanzoai/node:latest
        args:
          - --mine
          - --cluster-mode
          - --gpu
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Common Workflows

### 1. Setting Up Local AI Environment

**Objective**: Run AI locally for privacy and cost savings

```bash
# Install Hanzo Node
cargo install hanzo-node

# Initialize configuration
hanzo-node init --network mainnet

# Download models (GGUF format)
hanzo-node models pull llama-3-8b
hanzo-node models pull mistral-7b

# Start node with GPU acceleration
hanzo-node start --mine --gpu --layers 35

# Verify node status
hanzo-node status
hanzo-node peers  # See connected nodes
```

**Python SDK Connection**:
```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'
)

# Test inference
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response.choices[0].message.content)
```

### 2. Building AI Dashboard with @hanzo/ui

**Objective**: Create production-ready AI interface quickly

```bash
# Create Next.js app
pnpm create next-app my-ai-app
cd my-ai-app

# Install Hanzo UI
pnpm add @hanzo/ui @hanzo/live
```

```typescript
// app/page.tsx
import { AIChat, ModelSelector, TokenUsage, MetricsCard } from '@hanzo/ui'
import { useHanzoNode } from '@hanzo/ui/hooks'

export default function Dashboard() {
  const node = useHanzoNode({ url: 'http://localhost:8080' })
  
  return (
    <div className="hanzo-dashboard">
      <ModelSelector 
        models={node.models}
        onSelect={node.selectModel}
      />
      
      <AIChat 
        inference={node.infer}
        streaming
        darkMode
      />
      
      <div className="metrics">
        <MetricsCard title="Tokens" value={node.tokenUsage} />
        <MetricsCard title="Latency" value={node.latency} />
      </div>
    </div>
  )
}
```

### 3. Exposing Capabilities via MCP

**Objective**: Make Hanzo tools available to AI agents

```typescript
// mcp-server.ts
import { MCPServer, Tool, Resource } from '@hanzo/mcp'
import { HanzoNode } from '@hanzo/node'

const node = new HanzoNode({ url: 'http://localhost:8080' })

// Define tools
const inferTool: Tool = {
  name: 'hanzo_infer',
  description: 'Run inference on local Hanzo Node',
  parameters: {
    model: { type: 'string', required: true },
    prompt: { type: 'string', required: true },
    temperature: { type: 'number', default: 0.7 }
  },
  async execute({ model, prompt, temperature }) {
    return await node.infer({ model, prompt, temperature })
  }
}

// Define resources
const modelsResource: Resource = {
  uri: 'hanzo://models',
  name: 'Available Models',
  description: 'List of models on local Hanzo Node',
  async read() {
    return await node.listModels()
  }
}

// Start MCP server
const server = new MCPServer({
  name: 'hanzo-node-mcp',
  tools: [inferTool],
  resources: [modelsResource]
})

server.listen(8081)
```

### 4. Agentic Coding with Hanzo Dev

**Objective**: Use AI agent for terminal-based development

```bash
# Interactive chat mode
hanzo-dev chat

> "Add authentication middleware to Express API"
# Hanzo Dev analyzes codebase, generates middleware, updates routes

# Task mode (non-interactive)
hanzo-dev task "Refactor database layer to use repository pattern"

# Workflow with MCP tools
hanzo-dev --mcp hanzo-node,hanzo-ui workflow "
  1. Generate React dashboard using @hanzo/ui components
  2. Add real-time updates via Hanzo Live
  3. Deploy inference backend to Hanzo Node
  4. Write integration tests
"
```

## Hanzo vs Traditional Approaches

| Task | Traditional | Hanzo |
|------|------------|-------|
| **Model Inference** | 200 lines (HTTP, retries, parsing) | 5 lines (SDK) |
| **UI Chat Component** | 500 lines (state, streaming, errors) | 1 component |
| **MCP Integration** | Custom protocol implementation | Import `@hanzo/mcp` |
| **Local AI Setup** | Docker, CUDA, model conversion | `hanzo-node start` |
| **Distributed Compute** | K8s, load balancers, health checks | Hanzo Node cluster |
| **Privacy** | Custom encryption, auth, auditing | Built-in (local-first) |

## Related Skills

**Prerequisites**:
- `zig/zig-project-setup.md` - For understanding Rust patterns (similar concepts)
- `containers/docker-compose-development.md` - For local Hanzo Node setup
- `frontend/react-state-management.md` - For @hanzo/ui integration

**Related Workflows**:
- `ml/llm-model-routing.md` - Hanzo SDK handles this automatically
- `ml/llm-model-selection.md` - Updated with Hanzo patterns
- `workflow/beads-workflow.md` - Agentic task management
- `observability/prometheus-monitoring.md` - Monitoring Hanzo Node

**Next Steps**:
- `hanzo/hanzo-node.md` - Deep dive into Rust node architecture
- `hanzo/hanzo-mcp.md` - Comprehensive MCP patterns
- `hanzo/python-sdk.md` - Full SDK API reference

---

**Last Updated**: 2025-10-28  
**Category**: Hanzo Ecosystem  
**Related**: ml, infrastructure, frontend, workflow  
**Prerequisites**: Docker, Node.js, Rust (for Hanzo Node development)
