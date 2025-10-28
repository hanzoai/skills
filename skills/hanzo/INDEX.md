# Hanzo Ecosystem Skills

**Complete guide to Hanzo's AI infrastructure, components, and integration patterns**

## Overview

Hanzo is a comprehensive ecosystem for **privacy-first, locally-run AI** with production-ready abstractions. These skills cover the entire stack from distributed Rust infrastructure to React UI components, with emphasis on composability and agentic workflows.

### Philosophy

- **Local-First**: Run AI on your infrastructure for privacy and control
- **MCP-Native**: Built for Model Context Protocol agentic workflows
- **Higher-Level Abstractions**: Use Hanzo "legos" instead of line-by-line code
- **Production-Ready**: Battle-tested components and patterns

## Skill Catalog

### Core Infrastructure (6 skills)

**Hanzo Engine** (`hanzo-engine.md`) - 650 lines
**Purpose**: Native Rust inference & embedding engine for the entire Hanzo ecosystem
**Key Topics**:
- Blazingly fast LLM & embedding inference (Rust)
- Multimodal: text, vision, audio, embeddings, reranking
- PagedAttention, FlashAttention, ISQ, MLX support
- Optimized for Qwen3-Embedding (#1 on MTEB)
- OpenAI-compatible API (port 36900)
- MCP support built-in
- Embedded in Hanzo Node, Cloud Nodes, Hanzo Desktop

**Use When**:
- Running ZenLM models natively
- Maximum inference performance needed
- Embedding generation (Qwen3-Embedding)
- Multimodal AI applications
- Direct Rust integration for AI

**Organizations**:
- Engine: github.com/hanzoai/engine
- Based on mistral.rs (Eric Buehler)

**Hanzo Node** (`hanzo-node.md`) - 320 lines
**Purpose**: Distributed AI mining and networking infrastructure (Rust)
**Key Topics**:
- GPU/CPU scheduling for local inference
- P2P networking and consensus
- Model serving (OpenAI API, gRPC, MCP)
- Production deployment (Docker, K8s)

**Use When**:
- Running local AI models privately
- Joining distributed AI network
- High-performance inference needs
- Privacy-first applications

**Hanzo Desktop** (`hanzo-desktop.md`) - NEW
**Purpose**: AI-powered productivity app with mining capabilities (github.com/hanzoai)
**Key Topics**:
- Native Rust inference engine (Hanzo Engine)
- Mine $AI with ngrok/localxpose tunneling
- ZenLM models for local AI assistant
- Collect compute payments for open source projects
- Part of Hanzo AI Ecosystem (One, Team, hanzo.app)

**Use When**:
- macOS productivity with AI
- Mining $AI tokens passively
- Privacy-first local AI assistant
- Monetizing open source projects

**Hanzo Ecosystem**:
- AI Tools: github.com/hanzoai (Desktop, Node, SDK, Engine)
- Business Apps: Hanzo One, Hanzo Team, hanzo.app
- Most AI users: use hanzoai ecosystem

**Hanzo MCP** (`hanzo-mcp.md`) - 380 lines
**Purpose**: Model Context Protocol integration for agentic workflows
**Key Topics**:
- MCP servers (tools, resources, prompts)
- Multi-agent coordination patterns
- Tool orchestration across Hanzo ecosystem
- Integration with Hanzo Dev

**Use When**:
- Building AI agents with MCP
- Multi-agent workflows
- Terminal-based coding agents
- Context sharing between agents

**@hanzo/ui** (`hanzo-ui.md`) - 340 lines
**Purpose**: React component library for AI+Blockchain apps
**Key Topics**:
- AI components (AIChat, ModelSelector, StreamingText)
- Blockchain components (WalletConnect, TransactionStatus)
- Production patterns and performance
- Integration with Hanzo Node and Hanzo Live

**Use When**:
- Building AI dashboards
- Blockchain user interfaces
- Real-time AI applications
- Next.js/React projects

**Hanzo Python SDK** (`python-sdk.md`)
**Purpose**: Unified gateway to foundational models and AI cloud
**Key Topics**:
- Multi-provider support (OpenAI, Anthropic, local)
- Local inference via Hanzo Node
- Model routing and fallback
- Cost optimization
- API key signup with free credits

**Use When**:
- Python AI applications
- Backend API development
- Local + cloud hybrid inference
- Multi-model orchestration

**Hanzo Go SDK** (`go-sdk.md`) - NEW
**Purpose**: Type-safe Go interface for Hanzo AI with high-performance concurrency
**Key Topics**:
- Idiomatic Go API with context support
- Goroutine-safe concurrent requests
- HTTP/gRPC integration patterns
- Blockchain node integration
- Feature parity with Python/Rust SDKs

**Use When**:
- Backend services requiring concurrency
- Blockchain node applications
- Microservices architectures
- Systems requiring goroutine-based parallelism

**Organizations**:
- SDK: github.com/hanzoai/go-sdk
- Coming Soon: Golang Node (blockchain node in Go)

### AI Models & Training (2 skills)

**ZenLM** (`zenlm.md`) - NEW
**Purpose**: Next-generation local AI models (zen-nano, zen-eco, zen-agent)
**Key Topics**:
- 4B models achieving 70B-class performance
- 100% local processing, 95% less energy than cloud
- Runs on phones, laptops, Raspberry Pi at 50+ tokens/sec
- Integration with Hanzo Desktop, Python SDK, Zoo Gym
- Hanzo AI × Zoo Labs Foundation collaboration

**Use When**:
- Privacy-critical applications
- Offline AI capabilities
- Cost-optimized production
- Edge AI / IoT deployments

**Organizations**:
- Models: github.com/zenlm, huggingface.co/zenlm
- Training: github.com/zooai/gym
- Deployment: github.com/hanzoai

**Zoo Gym** (`hanzo-gym.md`) - NEW
**Purpose**: Unified training platform for ZenLM AI models (Zoo Labs Foundation)
**Key Topics**:
- GRPO (training-free reinforcement learning) innovation
- LoRA, QLoRA, GSPO, DPO, PPO training methods
- 2-5x faster with Unsloth, FlashAttention-2
- GGUF, MLX, AWQ, GPTQ quantization
- Train 4B models on 8GB GPUs
- Wildlife conservation AI research (501c3)

**Use When**:
- Fine-tuning ZenLM models
- Training-free reinforcement learning
- Custom domain adaptation
- Model quantization for deployment

**Organizations**:
- Zoo Gym: github.com/zooai/gym
- ZenLM Models: github.com/zenlm, huggingface.co/zenlm
- Hanzo AI: github.com/hanzoai

### Development Tools (3 skills)

**Hanzo Dev** (`hanzo-dev.md`) - Coming Soon
**Purpose**: Terminal-based AI coding agent
**Key Topics**:
- MCP integration for tool usage
- Local AI via Hanzo Node
- Agentic coding workflows
- Project understanding

**Use When**:
- Terminal-based development
- Agentic code generation
- Privacy-focused coding assistance
- CI/CD integration

**Hanzo Live** (`hanzo-live.md`) - Coming Soon
**Purpose**: Real-time generative AI pipelines
**Key Topics**:
- Streaming inference
- Pipeline composition
- Live updates for UIs
- WebSocket integration

**Use When**:
- Real-time AI features
- Streaming text/image generation
- Live dashboards
- Interactive AI applications

**Hanzo Chat** (`hanzo-chat.md`) - Coming Soon
**Purpose**: AI conversation interface
**Key Topics**:
- Multi-model chat
- Conversation history
- Context management
- UI integration

**Use When**:
- Building chat interfaces
- Multi-turn conversations
- Chat-based AI applications
- Customer support bots

### Architecture & Patterns (4 skills)

**Local AI Architecture** (`local-ai-architecture.md`) - Coming Soon
**Purpose**: Privacy-first AI deployment patterns
**Key Topics**:
- Hardware sizing and setup
- Single node vs cluster deployment
- Network topology
- Security and compliance

**Use When**:
- Deploying local AI infrastructure
- GDPR/HIPAA compliance needs
- Cost optimization strategies
- Enterprise AI deployments

**Agentic Workflows** (`agentic-workflows.md`) - Coming Soon
**Purpose**: Multi-agent coordination patterns
**Key Topics**:
- Orchestrator patterns
- Swarm patterns (parallel agents)
- Context sharing strategies
- Error handling and recovery

**Use When**:
- Building multi-agent systems
- Complex AI workflows
- Parallel task execution
- Agent collaboration

**MCP Patterns** (`mcp-patterns.md`) - Coming Soon
**Purpose**: Best practices for MCP integration
**Key Topics**:
- Tool design patterns
- Resource management
- Prompt templates
- Security and auth

**Use When**:
- Designing MCP tools
- Exposing capabilities to agents
- Building MCP servers
- Agent system architecture

**Distributed AI** (`distributed-ai.md`) - Coming Soon
**Purpose**: Multi-node cluster deployment
**Key Topics**:
- Load balancing strategies
- Failover and recovery
- Consensus mechanisms
- Monitoring and observability

**Use When**:
- High-availability AI services
- Scaling inference workloads
- Distributed training
- Geographic distribution

### Integration Guides (3 skills)

**Hanzo + Next.js** (`hanzo-nextjs-integration.md`) - Coming Soon
**Purpose**: Full-stack AI apps with Next.js
**Key Topics**:
- @hanzo/ui component integration
- Server actions with Hanzo SDK
- Streaming with Hanzo Live
- Deployment patterns

**Use When**:
- Building Next.js AI applications
- Full-stack TypeScript projects
- Server-side rendering with AI
- Vercel deployments

**Hanzo + Rust** (`hanzo-rust-integration.md`) - Coming Soon
**Purpose**: Integrating Hanzo in Rust applications
**Key Topics**:
- Hanzo Node client library
- gRPC integration
- Custom Rust tools for MCP
- Performance optimization

**Use When**:
- High-performance Rust services
- Systems programming with AI
- Custom Hanzo Node extensions
- Embedded AI applications

**Hanzo + Blockchain** (`hanzo-blockchain.md`) - Coming Soon
**Purpose**: AI+Blockchain integration patterns
**Key Topics**:
- Smart contract AI features
- On-chain AI inference verification
- Wallet integration with @hanzo/ui
- DeFi + AI use cases

**Use When**:
- Building DeFi with AI
- NFT platforms with AI generation
- DAO governance with AI analysis
- Blockchain + AI hybrid apps

## Learning Paths

### Path 1: Local AI Setup (Beginner)
**Goal**: Run AI models locally on your machine

```bash
1. cat skills/hanzo/hanzo-node.md
   # Learn Hanzo Node setup and configuration
   
2. cat skills/hanzo/python-sdk.md
   # Connect to local node from Python
   
3. cat skills/hanzo/local-ai-architecture.md
   # Understand architecture and hardware sizing
```

**Outcome**: Local Llama 3 8B running on your GPU, accessible via Python SDK

### Path 2: Building AI Dashboard (Intermediate)
**Goal**: Create production-ready AI interface

```bash
1. cat skills/hanzo/hanzo-ui.md
   # Learn @hanzo/ui components
   
2. cat skills/hanzo/hanzo-live.md
   # Add real-time streaming
   
3. cat skills/hanzo/hanzo-nextjs-integration.md
   # Full Next.js integration
   
4. cat skills/frontend/nextjs-app-router.md
   # Next.js fundamentals
```

**Outcome**: Real-time AI dashboard with streaming inference and elegant UI

### Path 3: Agentic Development (Advanced)
**Goal**: Build multi-agent AI workflows

```bash
1. cat skills/hanzo/hanzo-mcp.md
   # Understand MCP protocol
   
2. cat skills/hanzo/hanzo-dev.md
   # Terminal AI coding agent
   
3. cat skills/hanzo/agentic-workflows.md
   # Multi-agent patterns
   
4. cat skills/workflow/beads-workflow.md
   # Task management for agents
```

**Outcome**: Terminal-based coding agent using MCP to orchestrate Hanzo tools

### Path 4: Production Deployment (Expert)
**Goal**: Deploy distributed AI cluster

```bash
1. cat skills/hanzo/hanzo-node.md
   # Node architecture and configuration
   
2. cat skills/hanzo/distributed-ai.md
   # Cluster deployment patterns
   
3. cat skills/cloud/kubernetes-deployment.md
   # Kubernetes fundamentals
   
4. cat skills/observability/prometheus-monitoring.md
   # Monitoring and alerting
```

**Outcome**: Production Hanzo cluster with 3+ nodes, load balancing, and monitoring

## Common Workflows

### Workflow 1: Setup Local AI (30 minutes)

```bash
# 1. Install Hanzo Node
cargo install hanzo-node
hanzo-node init --network mainnet

# 2. Download model
hanzo-node models pull llama-3-8b

# 3. Start node
hanzo-node start --mine --gpu

# 4. Test with Python
python -c "
from hanzo import Hanzo
hanzo = Hanzo(inference_mode='local')
print(hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
))
"
```

**Result**: Local AI running, accessible via SDK

### Workflow 2: Build AI Chat Interface (1 hour)

```bash
# 1. Create Next.js app
pnpm create next-app my-ai-app
cd my-ai-app

# 2. Install @hanzo/ui
pnpm add @hanzo/ui

# 3. Create chat page
cat > app/chat/page.tsx << 'EOF'
import { AIChat, useHanzoNode } from '@hanzo/ui'

export default function ChatPage() {
  const node = useHanzoNode({ url: 'http://localhost:8080' })
  
  return <AIChat inference={node.infer} streaming />
}
EOF

# 4. Run dev server
pnpm dev
```

**Result**: Production-ready AI chat interface

### Workflow 3: Expose MCP Tool (45 minutes)

```typescript
// mcp-server.ts
import { MCPServer, Tool } from '@hanzo/mcp'
import { HanzoNode } from '@hanzo/node-client'

const node = new HanzoNode({ url: 'http://localhost:8080' })

const tools: Tool[] = [
  {
    name: 'hanzo_infer',
    description: 'Run local AI inference',
    parameters: {
      model: { type: 'string', required: true },
      prompt: { type: 'string', required: true }
    },
    async execute({ model, prompt }) {
      return await node.infer({ model, prompt })
    }
  }
]

const server = new MCPServer({
  name: 'hanzo-node',
  tools
})

await server.listen(8081)
```

**Result**: Hanzo Node capabilities exposed via MCP for agents

### Workflow 4: Agentic Coding with Hanzo Dev (20 minutes)

```bash
# 1. Configure MCP servers
hanzo-dev config mcp add hanzo-node http://localhost:8081
hanzo-dev config mcp add hanzo-ui http://localhost:8082

# 2. Execute workflow
hanzo-dev workflow "
  1. Generate React dashboard using @hanzo/ui
  2. Add AI chat with local inference
  3. Deploy to Vercel
  4. Run smoke tests
"
```

**Result**: Complete feature implemented by AI agent using MCP tools

## Integration Matrix

| Component | Hanzo Node | Hanzo MCP | @hanzo/ui | Python SDK | Hanzo Dev |
|-----------|-----------|-----------|-----------|------------|-----------|
| **Hanzo Node** | - | Expose via MCP | Use in UI | SDK client | Config backend |
| **Hanzo MCP** | Tool server | - | UI generator | Python tool | Orchestration |
| **@hanzo/ui** | `useHanzoNode` | MCP tool | - | Backend API | Component gen |
| **Python SDK** | Node client | Tool wrapper | API backend | - | Python runner |
| **Hanzo Dev** | Inference | MCP client | Generate UI | Execute code | - |

## Hanzo vs Alternatives

| Need | Traditional | Hanzo Solution |
|------|------------|----------------|
| **Local AI** | Docker + CUDA + custom code (200 lines) | `hanzo-node start` (1 command) |
| **UI Chat** | Custom React component (500 lines) | `<AIChat />` (1 component) |
| **MCP Integration** | Manual protocol implementation | `@hanzo/mcp` (built-in) |
| **Model Routing** | Custom logic (100 lines) | Hanzo SDK auto-routing |
| **Privacy** | Custom encryption, auth, logging | Built-in (local-first) |
| **Deployment** | K8s manifests, monitoring, scaling | Hanzo cluster mode |

## Related Skill Categories

**Prerequisites**:
- `frontend/` - React, Next.js fundamentals
- `containers/` - Docker, Kubernetes basics
- `workflow/` - Agentic task management
- `plt/` - Programming language theory (for MCP)

**Integration**:
- `ml/` - Updated with Hanzo patterns
- `api/` - REST/GraphQL API design
- `observability/` - Monitoring and metrics
- `database/` - Data persistence

**Advanced**:
- `distributed-systems/` - Consensus and replication
- `cryptography/` - Security and encryption
- `formal/` - Verification and correctness

## Gateway Skill

**Start here**: `discover-hanzo/SKILL.md` (auto-activates on Hanzo keywords)

The gateway skill provides:
- Overview of entire Hanzo ecosystem
- Quick reference for all products
- When to use each component
- Common integration patterns

## Contributing

These skills are maintained to reflect the latest Hanzo ecosystem patterns. As new components are released or patterns emerge, skills are updated or added.

**Skill Template**: `skills/_SKILL_TEMPLATE.md`

---

**Last Updated**: 2025-10-28
**Total Skills**: 10 (7 complete, 3 planned)
**Estimated Time**: 8-12 hours to master core stack
**Difficulty**: Beginner to Advanced
