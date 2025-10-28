# Hanzo MCP - Agentic Development with Model Context Protocol

**Category**: Hanzo Ecosystem  
**Related Skills**: `workflow/beads-workflow.md`, `plt/typed-holes-llm.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo MCP is the **agentic workflow layer** of the Hanzo ecosystem. It implements the [Model Context Protocol](https://modelcontextprotocol.io/) - a standard for exposing tools, resources, and prompts to AI agents. This enables seamless multi-agent coordination, context sharing, and tool orchestration.

### What is MCP?

**Model Context Protocol** is an open protocol that stan

dardizes how AI agents access:
- **Tools**: Executable functions (e.g., run inference, query database)
- **Resources**: Contextual data (e.g., codebase, documentation)
- **Prompts**: Reusable workflow templates

### Why MCP for Hanzo?

**Traditional Approach** (without MCP):
```typescript
// Each agent needs custom integration
const result1 = await hanzoNode.infer({...})          // Custom API
const result2 = await database.query({...})            // Different API
const result3 = await customTool.execute({...})        // Another API
```

**With Hanzo MCP**:
```typescript
// Unified MCP interface
const result1 = await mcp.callTool('hanzo_infer', {...})
const result2 = await mcp.callTool('db_query', {...})
const result3 = await mcp.callTool('custom_tool', {...})
```

Agents can now discover and use capabilities without custom integration.

## Quick Start

### Installation

```bash
pnpm add @hanzo/mcp
```

### MCP Server (Expose Capabilities)

```typescript
import { MCPServer, Tool, Resource, Prompt } from '@hanzo/mcp'

// Define a tool
const inferTool: Tool = {
  name: 'hanzo_infer',
  description: 'Run inference on local Hanzo Node',
  parameters: {
    model: {
      type: 'string',
      enum: ['llama-3-8b', 'llama-3-70b', 'mistral-7b'],
      required: true
    },
    prompt: { type: 'string', required: true },
    temperature: { type: 'number', default: 0.7 }
  },
  async execute({ model, prompt, temperature }) {
    const response = await hanzoNode.infer({
      model,
      prompt,
      temperature
    })
    return response.text
  }
}

// Define a resource
const modelsResource: Resource = {
  uri: 'hanzo://models',
  name: 'Available Models',
  description: 'List of models on local Hanzo Node',
  mimeType: 'application/json',
  async read() {
    return JSON.stringify(await hanzoNode.listModels())
  }
}

// Create MCP server
const server = new MCPServer({
  name: 'hanzo-node-mcp',
  version: '1.0.0',
  tools: [inferTool],
  resources: [modelsResource]
})

// Start server
await server.listen(8081)
```

### MCP Client (Use Capabilities)

```typescript
import { MCPClient } from '@hanzo/mcp'

// Connect to MCP servers
const client = new MCPClient({
  servers: [
    { name: 'hanzo-node', url: 'http://localhost:8081' },
    { name: 'hanzo-ui', url: 'http://localhost:8082' }
  ]
})

await client.connect()

// List available tools
const tools = await client.listTools()
// ['hanzo_infer', 'ui_generate_component', ...]

// Call a tool
const result = await client.callTool('hanzo_infer', {
  model: 'llama-3-8b',
  prompt: 'Explain Rust ownership'
})

console.log(result)  // Generated text from Hanzo Node
```

## Core Concepts

### Tools

Tools are **executable functions** exposed via MCP. They allow agents to take actions.

**Anatomy of a Tool**:
```typescript
interface Tool {
  name: string                    // Unique identifier
  description: string             // What the tool does
  parameters: ParameterSchema     // Input schema
  execute: (params) => Promise<any>  // Implementation
}
```

**Example Tool - Hanzo Node Inference**:
```typescript
const inferTool: Tool = {
  name: 'hanzo_infer',
  description: 'Run local AI inference on Hanzo Node. Returns generated text.',
  parameters: {
    model: {
      type: 'string',
      description: 'Model ID to use',
      enum: ['llama-3-8b', 'llama-3-70b', 'mistral-7b'],
      required: true
    },
    prompt: {
      type: 'string',
      description: 'User prompt for generation',
      required: true
    },
    temperature: {
      type: 'number',
      description: 'Sampling temperature (0-2)',
      default: 0.7,
      minimum: 0,
      maximum: 2
    },
    max_tokens: {
      type: 'number',
      description: 'Maximum tokens to generate',
      default: 500
    }
  },
  async execute({ model, prompt, temperature = 0.7, max_tokens = 500 }) {
    try {
      const response = await hanzoNode.infer({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        max_tokens
      })
      
      return {
        text: response.choices[0].message.content,
        tokens: response.usage
      }
    } catch (error) {
      throw new MCPError(`Inference failed: ${error.message}`)
    }
  }
}
```

### Resources

Resources are **contextual data** that agents can read. They provide context without executing code.

**Anatomy of a Resource**:
```typescript
interface Resource {
  uri: string                      // Unique URI
  name: string                     // Human-readable name
  description: string              // What the resource contains
  mimeType: string                 // Content type
  read: () => Promise<string | Buffer>  // Fetch content
}
```

**Example Resource - Codebase Context**:
```typescript
const codebaseResource: Resource = {
  uri: 'hanzo://codebase',
  name: 'Codebase Context',
  description: 'Current project file structure and contents',
  mimeType: 'application/json',
  async read() {
    const files = await readProjectFiles()
    return JSON.stringify({
      structure: await getFileTree(),
      files: files.map(f => ({
        path: f.path,
        content: f.content,
        language: f.language
      }))
    })
  }
}
```

**Example Resource - Documentation**:
```typescript
const docsResource: Resource = {
  uri: 'hanzo://docs',
  name: 'API Documentation',
  description: 'OpenAPI spec and usage examples',
  mimeType: 'text/markdown',
  async read() {
    return await fs.readFile('./docs/API.md', 'utf-8')
  }
}
```

### Prompts

Prompts are **reusable workflow templates** that agents can execute.

**Anatomy of a Prompt**:
```typescript
interface Prompt {
  name: string                     // Unique identifier
  description: string              // What the workflow does
  parameters?: ParameterSchema     // Input parameters
  template: string | PromptTemplate  // Prompt template
}
```

**Example Prompt - Code Review**:
```typescript
const codeReviewPrompt: Prompt = {
  name: 'code_review',
  description: 'Perform comprehensive code review of a file',
  parameters: {
    file_path: {
      type: 'string',
      description: 'Path to file to review',
      required: true
    },
    focus: {
      type: 'string',
      description: 'Review focus area',
      enum: ['security', 'performance', 'style', 'all'],
      default: 'all'
    }
  },
  template: `
You are a senior software engineer reviewing code.

File: {{file_path}}
Focus: {{focus}}

Review the following code:

\`\`\`
{{file_content}}
\`\`\`

Provide:
1. Issues found (security, bugs, performance)
2. Suggested improvements
3. Code quality rating (1-10)
  `
}
```

## Hanzo-Specific MCP Patterns

### Pattern 1: Hanzo Node Tool

Expose local AI inference as MCP tool:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'
import { HanzoNode } from '@hanzo/node-client'

const node = new HanzoNode({ url: 'http://localhost:8080' })

const hanzoTools: Tool[] = [
  // Inference tool
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
  },
  
  // Model listing tool
  {
    name: 'hanzo_list_models',
    description: 'List available models on Hanzo Node',
    parameters: {},
    async execute() {
      return await node.listModels()
    }
  },
  
  // Model download tool
  {
    name: 'hanzo_download_model',
    description: 'Download model to Hanzo Node',
    parameters: {
      model_id: { type: 'string', required: true },
      quantization: {
        type: 'string',
        enum: ['q4_k_m', 'q5_k_m', 'q8_0'],
        default: 'q4_k_m'
      }
    },
    async execute({ model_id, quantization }) {
      return await node.downloadModel(model_id, quantization)
    }
  }
]

const server = new MCPServer({
  name: 'hanzo-node',
  tools: hanzoTools
})
```

### Pattern 2: @hanzo/ui Component Generation

Expose UI component generation as MCP tool:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'
import { generateComponent } from '@hanzo/ui/generator'

const uiTool: Tool = {
  name: 'hanzo_ui_generate',
  description: 'Generate React component using @hanzo/ui',
  parameters: {
    type: {
      type: 'string',
      enum: ['AIChat', 'Dashboard', 'Form', 'Table', 'Chart'],
      required: true
    },
    props: {
      type: 'object',
      description: 'Component props',
      required: true
    },
    styling: {
      type: 'string',
      enum: ['default', 'minimal', 'pro'],
      default: 'default'
    }
  },
  async execute({ type, props, styling }) {
    const code = await generateComponent({
      componentType: type,
      props,
      theme: styling
    })
    
    return {
      code,
      imports: code.match(/import .* from ['"]@hanzo\/ui['"]/g),
      usage: `<${type} ${Object.entries(props).map(([k, v]) => `${k}={${JSON.stringify(v)}}`).join(' ')} />`
    }
  }
}
```

### Pattern 3: Python SDK Tool

Bridge Python SDK to MCP:

```typescript
import { MCPServer, Tool } from '@hanzo/mcp'
import { spawn } from 'child_process'

const pythonSdkTool: Tool = {
  name: 'hanzo_py_sdk',
  description: 'Execute Python code using Hanzo SDK',
  parameters: {
    code: { type: 'string', required: true },
    requirements: {
      type: 'array',
      items: { type: 'string' },
      description: 'pip packages to install',
      default: []
    }
  },
  async execute({ code, requirements }) {
    // Install requirements
    if (requirements.length > 0) {
      await execPromise(`pip install ${requirements.join(' ')}`)
    }
    
    // Execute Python code
    const script = `
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='local')

${code}
    `
    
    const result = await execPython(script)
    return result
  }
}
```

### Pattern 4: Agentic Workflow Prompts

Define multi-step workflows as prompts:

```typescript
const workflowPrompts: Prompt[] = [
  {
    name: 'feature_development',
    description: 'Full feature development workflow',
    parameters: {
      feature_name: { type: 'string', required: true },
      description: { type: 'string', required: true }
    },
    template: `
You are implementing a new feature: {{feature_name}}

Description: {{description}}

Execute the following workflow:

1. **Design**: 
   - Call hanzo_ui_generate to create UI mockup
   - Review design with user

2. **Implementation**:
   - Generate backend API (FastAPI + Hanzo Python SDK)
   - Generate frontend component (@hanzo/ui)
   - Integrate with Hanzo Node for AI features

3. **Testing**:
   - Write unit tests
   - Write integration tests
   - Manual testing checklist

4. **Documentation**:
   - Update API docs
   - Update user guide
   - Record demo video

Use available MCP tools at each step. Confirm with user before proceeding to next step.
    `
  }
]
```

## Multi-Agent Coordination

### Orchestrator Pattern

```typescript
import { MCPClient } from '@hanzo/mcp'

class AgentOrchestrator {
  private clients: Map<string, MCPClient> = new Map()
  
  async initialize() {
    // Connect to multiple MCP servers
    const hanzoNode = new MCPClient({
      servers: [{ name: 'hanzo-node', url: 'http://localhost:8081' }]
    })
    await hanzoNode.connect()
    this.clients.set('hanzo-node', hanzoNode)
    
    const hanzoUI = new MCPClient({
      servers: [{ name: 'hanzo-ui', url: 'http://localhost:8082' }]
    })
    await hanzoUI.connect()
    this.clients.set('hanzo-ui', hanzoUI)
  }
  
  async executeWorkflow(workflow: string) {
    // Parse workflow steps
    const steps = parseWorkflow(workflow)
    
    const results = []
    for (const step of steps) {
      // Route to appropriate agent
      const agent = this.getAgentForStep(step)
      const client = this.clients.get(agent)
      
      // Execute step
      const result = await client.callTool(step.tool, step.params)
      results.push(result)
      
      // Pass context to next step
      step.context = result
    }
    
    return results
  }
  
  private getAgentForStep(step: WorkflowStep): string {
    if (step.tool.startsWith('hanzo_infer')) return 'hanzo-node'
    if (step.tool.startsWith('hanzo_ui')) return 'hanzo-ui'
    throw new Error(`Unknown tool: ${step.tool}`)
  }
}
```

### Swarm Pattern (Parallel Agents)

```typescript
async function parallelAgentSwarm(tasks: Task[]) {
  const client = new MCPClient({
    servers: [
      { name: 'agent-1', url: 'http://localhost:8081' },
      { name: 'agent-2', url: 'http://localhost:8082' },
      { name: 'agent-3', url: 'http://localhost:8083' }
    ]
  })
  
  await client.connect()
  
  // Distribute tasks across agents
  const promises = tasks.map(async (task, index) => {
    const agent = `agent-${(index % 3) + 1}`
    const server = client.getServer(agent)
    
    return await server.callTool(task.tool, task.params)
  })
  
  // Wait for all agents to complete
  const results = await Promise.all(promises)
  
  // Aggregate results
  return aggregateResults(results)
}
```

## Integration with Hanzo Dev

Hanzo Dev (terminal coding agent) can use MCP to orchestrate tools:

```bash
# Configure MCP servers in Hanzo Dev
hanzo-dev config mcp add hanzo-node http://localhost:8081
hanzo-dev config mcp add hanzo-ui http://localhost:8082
hanzo-dev config mcp add database http://localhost:8083

# Execute workflow using MCP tools
hanzo-dev workflow "
  1. Generate React dashboard component using hanzo-ui MCP tool
  2. Add real-time metrics from database MCP tool
  3. Deploy model to Hanzo Node using hanzo-node MCP tool
  4. Run integration tests
"
```

**Under the hood**:
```typescript
// Hanzo Dev internally uses MCP client
const client = new MCPClient({
  servers: config.mcp.servers
})

// For each workflow step, call appropriate MCP tool
for (const step of workflow.steps) {
  const tool = await client.findTool(step.description)
  const result = await client.callTool(tool.name, step.params)
  context.addResult(step, result)
}
```

## MCP vs REST/GraphQL

| Feature | REST/GraphQL | MCP |
|---------|--------------|-----|
| **Discovery** | Manual (docs) | Automatic (schema) |
| **Versioning** | Explicit endpoints | Built-in |
| **Context Sharing** | Manual passing | Resources |
| **Tool Composition** | Manual orchestration | Native |
| **Agent Integration** | Custom per API | Standardized |
| **Type Safety** | API-specific | Protocol-level |

## Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  hanzo-node-mcp:
    image: hanzoai/mcp-server:latest
    ports:
      - "8081:8081"
    environment:
      MCP_NAME: hanzo-node
      HANZO_NODE_URL: http://hanzo-node:8080
    depends_on:
      - hanzo-node
  
  hanzo-ui-mcp:
    image: hanzoai/ui-mcp-server:latest
    ports:
      - "8082:8082"
    environment:
      MCP_NAME: hanzo-ui
  
  hanzo-node:
    image: hanzoai/node:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/root/.hanzo/models
```

### Kubernetes

```yaml
# mcp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-mcp-cluster
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hanzo-mcp
  template:
    metadata:
      labels:
        app: hanzo-mcp
    spec:
      containers:
      - name: mcp-server
        image: hanzoai/mcp-server:latest
        ports:
        - containerPort: 8081
        env:
        - name: MCP_NAME
          value: "hanzo-node"
        - name: HANZO_NODE_URL
          value: "http://hanzo-node:8080"
---
apiVersion: v1
kind: Service
metadata:
  name: hanzo-mcp
spec:
  selector:
    app: hanzo-mcp
  ports:
  - port: 8081
    targetPort: 8081
  type: LoadBalancer
```

## Security

### Authentication

```typescript
const server = new MCPServer({
  name: 'hanzo-node',
  auth: {
    type: 'bearer',
    validate: async (token) => {
      const user = await validateJWT(token)
      return user
    }
  },
  tools: [...]
})
```

### Rate Limiting

```typescript
const server = new MCPServer({
  name: 'hanzo-node',
  rateLimits: {
    'hanzo_infer': {
      windowMs: 60000,  // 1 minute
      max: 10           // 10 requests per minute
    }
  },
  tools: [...]
})
```

## Related Skills

**Prerequisites**:
- `workflow/beads-workflow.md` - Agentic task management
- `plt/typed-holes-llm.md` - AI-assisted programming
- `api/rest-api-design.md` - API fundamentals

**Integration**:
- `hanzo/hanzo-node.md` - Expose Node capabilities
- `hanzo/hanzo-ui.md` - UI generation tools
- `hanzo/python-sdk.md` - Python integration
- `hanzo/hanzo-dev.md` - Terminal agent usage

**Next Steps**:
- `hanzo/agentic-workflows.md` - Advanced multi-agent patterns
- `hanzo/mcp-patterns.md` - Best practices and patterns

---

**Last Updated**: 2025-10-28  
**Category**: Hanzo Ecosystem  
**Related**: workflow, ai, infrastructure  
**Prerequisites**: TypeScript, async/await, AI agent concepts
