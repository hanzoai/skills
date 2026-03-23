# Hanzo MCP - Model Context Protocol

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-zap.md`, `hanzo/hanzo-agent.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo MCP implements the **Model Context Protocol** -- an open standard for exposing tools, resources, and prompts to AI agents. Ships **13 HIP-0300 unified tools** (7 core + 6 optional) with action-based routing. 260+ tools across the ecosystem. ZAP-native: any ZAP service gets MCP tools for free.

## When to use

- Exposing tools, resources, and prompts to AI agents
- Building MCP servers for custom services
- Connecting MCP clients to Hanzo infrastructure
- Agentic workflows with multi-agent coordination
- Browser extensions that need MCP tool access

## Hard requirements

1. **Node.js >= 18** for the MCP server
2. **MCP protocol compliance**: tools, resources, and prompts follow the spec
3. **ZAP integration**: ZAP services automatically expose MCP interfaces

## Quick reference

| Item | Value |
|------|-------|
| npm | `@hanzo/mcp` |
| Version | 2.4.1 |
| Binary | `hanzo-mcp` |
| Node | >= 18 |
| Repo | `github.com/hanzoai/mcp` |
| Total tools | 260+ across ecosystem |
| Core tools | 13 HIP-0300 unified |

## What is MCP?

Model Context Protocol standardizes how AI agents access:
- **Tools**: Executable functions (run inference, query database, deploy service)
- **Resources**: Contextual data (codebase, documentation, schemas)
- **Prompts**: Reusable workflow templates

## Install and CLI

```bash
# Install globally
npm install -g @hanzo/mcp

# Serve MCP server
hanzo-mcp serve

# List available tools
hanzo-mcp list-tools

# Install for desktop apps (Claude, etc.)
hanzo-mcp install-desktop
```

## 13 HIP-0300 unified tools

The core tool set uses action-based routing where each tool handles multiple operations:

| Tool | Actions | Purpose |
|------|---------|---------|
| `hanzo_read` | file, dir, url, resource | Read files, directories, URLs |
| `hanzo_write` | file, create, append | Write and create files |
| `hanzo_edit` | replace, insert, delete | Edit file contents |
| `hanzo_exec` | shell, script, process | Execute commands |
| `hanzo_search` | grep, glob, find | Search files and content |
| `hanzo_fetch` | http, graphql, websocket | Network requests |
| `hanzo_think` | reason, plan, decide | Structured reasoning |
| `hanzo_browser` | navigate, click, type | Browser automation (optional) |
| `hanzo_memory` | store, recall, forget | Persistent memory (optional) |
| `hanzo_code` | analyze, refactor, test | Code intelligence (optional) |
| `hanzo_git` | status, diff, commit | Git operations (optional) |
| `hanzo_agent` | spawn, coordinate, delegate | Multi-agent (optional) |
| `hanzo_computer` | screenshot, click, type | Computer use (optional) |

## MCP server example

```typescript
import { MCPServer, Tool, Resource } from '@hanzo/mcp'

const inferTool: Tool = {
  name: 'hanzo_infer',
  description: 'Run inference on Hanzo LLM Gateway',
  parameters: {
    model: { type: 'string', required: true },
    prompt: { type: 'string', required: true },
    temperature: { type: 'number', default: 0.7 }
  },
  async execute({ model, prompt, temperature }) {
    const response = await fetch('https://api.hanzo.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.HANZO_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature
      })
    })
    return await response.json()
  }
}

const server = new MCPServer({
  name: 'hanzo-node-mcp',
  version: '1.0.0',
  tools: [inferTool]
})

await server.listen(8081)
```

## MCP client example

```typescript
import { MCPClient } from '@hanzo/mcp'

const client = new MCPClient({
  servers: [
    { name: 'hanzo-node', url: 'http://localhost:8081' },
    { name: 'hanzo-db', url: 'http://localhost:8082' }
  ]
})

await client.connect()

// Discover tools
const tools = await client.listTools()

// Call a tool
const result = await client.callTool('hanzo_infer', {
  model: 'zen-70b',
  prompt: 'Explain Rust ownership'
})
```

## ZAP-MCP bridge

Any ZAP service automatically gets MCP tools. The ZAP Gateway (`zapd`) translates:

| ZAP | MCP |
|-----|-----|
| ZAP tools | `listTools`, `callTool` |
| ZAP resources | `listResources`, `readResource` |
| ZAP prompts | `listPrompts`, `getPrompt` |

This means PostgreSQL, Redis, ClickHouse, and MongoDB all get MCP tools when running with a ZAP sidecar.

## Optional dependencies

| Package | Purpose |
|---------|---------|
| `@lancedb/lancedb` | Vector storage for memory tools |
| `@xenova/transformers` | Local embeddings |
| `playwright` | Browser automation tools |

## Integration with Claude Desktop

```bash
# Auto-configure for Claude Desktop
hanzo-mcp install-desktop

# This adds to ~/Library/Application Support/Claude/claude_desktop_config.json:
{
  "mcpServers": {
    "hanzo": {
      "command": "hanzo-mcp",
      "args": ["serve"]
    }
  }
}
```

## Production deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-mcp
  namespace: hanzo
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: mcp-server
          image: ghcr.io/hanzoai/mcp:latest
          ports:
            - containerPort: 8081
          env:
            - name: HANZO_API_KEY
              valueFrom:
                secretKeyRef:
                  name: mcp-secrets
                  key: HANZO_API_KEY
```

## Security

```typescript
const server = new MCPServer({
  name: 'hanzo-node',
  auth: {
    type: 'bearer',
    validate: async (token) => {
      return await validateJWT(token)
    }
  },
  rateLimits: {
    'hanzo_infer': { windowMs: 60000, max: 10 }
  },
  tools: [...]
})
```

## Related Skills

- `hanzo/hanzo-zap.md` -- ZAP protocol (provides MCP for free)
- `hanzo/hanzo-agent.md` -- Multi-agent SDK
- `hanzo/hanzo-dev.md` -- Terminal AI agent using MCP
- `hanzo/hanzo-bot.md` -- Bot extensions expose MCP tools

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: mcp, agentic, tools, protocol, zap
**Prerequisites**: TypeScript, async/await, AI agent concepts
