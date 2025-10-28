# Hanzo Live - Real-Time AI Streaming

**Category**: Hanzo Ecosystem  
**Skill Level**: Intermediate to Advanced
**Prerequisites**: WebSockets, async programming, streaming concepts

## Overview

Hanzo Live provides real-time AI streaming infrastructure for building responsive, collaborative AI applications. Unlike traditional request-response AI APIs that require waiting for complete responses, Hanzo Live delivers token-by-token streaming with sub-50ms latency, enabling fluid conversational interfaces and real-time collaboration.

**Core Philosophy**: Stream everything - tokens, thoughts, tool calls, and state updates for immediate user feedback.

## Key Features

### ⚡ Ultra-Low Latency Streaming
- **Sub-50ms first token**: Immediate response start
- **Token-by-token delivery**: Smooth text generation
- **WebSocket transport**: Bidirectional real-time communication
- **Edge deployment**: Minimize geographic latency

### 🔄 Real-Time Collaboration
- **Multi-user sessions**: Share AI conversations live
- **Presence indicators**: See who's online and typing
- **Collaborative editing**: Multiple users, one AI stream
- **Cursor tracking**: Real-time user positions

### 🎯 Advanced Streaming Patterns
- **Thought streaming**: See AI reasoning process
- **Tool call streaming**: Watch tool execution in real-time
- **Progress indicators**: Loading states and completion metrics
- **Error recovery**: Graceful handling of stream interruptions

### 🛠 Developer Experience
- **TypeScript SDK**: Type-safe streaming clients
- **React hooks**: Drop-in streaming components
- **State management**: Automatic stream state tracking
- **Reconnection**: Automatic resume on disconnect

## Architecture

```
┌─────────────┐         WebSocket         ┌──────────────┐
│   Client    │ ◄─────────────────────► │  Hanzo Live  │
│  (Browser)  │                          │    Server    │
└─────────────┘                          └──────────────┘
                                                │
                                                │ HTTP/gRPC
                                                ▼
                                         ┌──────────────┐
                                         │  Hanzo Node  │
                                         │   (Local)    │
                                         └──────────────┘
```

## Installation

### Server Setup

```bash
# Install Hanzo Live server
npm install -g @hanzo/live-server

# Or with Docker
docker pull hanzoai/live-server

# Start server
hanzo-live serve \
  --port 3001 \
  --node-url http://localhost:8080 \
  --redis redis://localhost:6379
```

### Client Installation

```bash
# React/Next.js projects
npm install @hanzo/live

# Vanilla JavaScript
npm install @hanzo/live-client

# Python clients
pip install hanzo-live
```

## Quick Start

### Basic Streaming (React)

```tsx
import { useHanzoLive } from '@hanzo/live'

function ChatApp() {
  const { 
    messages,
    isStreaming,
    sendMessage 
  } = useHanzoLive({
    url: 'ws://localhost:3001',
    model: 'llama-3-8b'
  })

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i}>
          <strong>{msg.role}:</strong> {msg.content}
        </div>
      ))}
      
      {isStreaming && <div>AI is typing...</div>}
      
      <input 
        onKeyPress={(e) => {
          if (e.key === 'Enter') {
            sendMessage(e.target.value)
          }
        }}
      />
    </div>
  )
}
```

### Streaming with TypeScript SDK

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  model: 'llama-3-8b'
})

// Connect
await client.connect()

// Start streaming
const stream = await client.chat({
  messages: [
    { role: 'user', content: 'Explain quantum computing' }
  ]
})

// Handle tokens as they arrive
for await (const token of stream) {
  process.stdout.write(token)
}
```

### Streaming with Python

```python
from hanzo_live import HanzoLiveClient

client = HanzoLiveClient('ws://localhost:3001')

# Synchronous streaming
for token in client.stream_chat(
    messages=[
        {'role': 'user', 'content': 'Write a poem'}
    ],
    model='llama-3-8b'
):
    print(token, end='', flush=True)

# Async streaming
async for token in client.stream_chat_async(...):
    await display_token(token)
```

## Core Features

### 1. Token-by-Token Streaming

```tsx
import { useStreamingCompletion } from '@hanzo/live'

function StreamingChat() {
  const {
    content,          // Current accumulated content
    isStreaming,      // Is actively streaming?
    tokensPerSecond,  // Real-time speed metric
    stop              // Cancel stream
  } = useStreamingCompletion({
    model: 'llama-3-8b',
    onToken: (token) => {
      // Called for each token
      console.log('Token:', token)
    },
    onComplete: (fullText) => {
      // Called when stream ends
      console.log('Complete:', fullText)
    }
  })

  return (
    <div>
      <div className="content">{content}</div>
      {isStreaming && (
        <div className="metrics">
          {tokensPerSecond.toFixed(1)} tokens/sec
          <button onClick={stop}>Stop</button>
        </div>
      )}
    </div>
  )
}
```

### 2. Thought Process Streaming

```tsx
import { useThoughtStream } from '@hanzo/live'

function ThoughtfulChat() {
  const {
    thoughts,   // Array of thinking steps
    answer,     // Final answer
    isThinking  // Is currently thinking?
  } = useThoughtStream({
    model: 'qwen-2-7b',
    showThoughts: true
  })

  return (
    <div>
      {thoughts.map((thought, i) => (
        <div key={i} className="thought">
          <span className="step">Step {i+1}:</span> {thought}
        </div>
      ))}
      
      {isThinking && <div className="loader">Thinking...</div>}
      
      {answer && (
        <div className="answer">
          <strong>Answer:</strong> {answer}
        </div>
      )}
    </div>
  )
}
```

### 3. Tool Call Streaming

```tsx
import { useToolStream } from '@hanzo/live'

function ToolAwareChat() {
  const {
    messages,
    toolCalls,      // In-progress tool calls
    toolResults,    // Completed tool results
    isExecuting     // Is executing tools?
  } = useToolStream({
    model: 'gpt-4',
    tools: [
      {
        name: 'search_web',
        description: 'Search the web',
        parameters: { query: 'string' }
      }
    ]
  })

  return (
    <div>
      {toolCalls.map((call, i) => (
        <div key={i} className="tool-call">
          <span className="tool-name">{call.name}</span>
          <span className="tool-args">{JSON.stringify(call.arguments)}</span>
          {call.status === 'running' && <Spinner />}
          {call.status === 'complete' && <CheckMark />}
        </div>
      ))}
      
      {toolResults.map((result, i) => (
        <div key={i} className="tool-result">
          {result.output}
        </div>
      ))}
    </div>
  )
}
```

### 4. Multi-User Collaboration

```tsx
import { useCollaborativeSession } from '@hanzo/live'

function CollaborativeChat() {
  const {
    messages,
    users,          // Connected users
    userActivity,   // Who's typing, thinking, etc
    sendMessage,
    inviteUser
  } = useCollaborativeSession({
    sessionId: 'project-brainstorm',
    userId: currentUser.id
  })

  return (
    <div>
      <div className="users">
        {users.map(user => (
          <div key={user.id} className="user">
            <Avatar src={user.avatar} />
            <span>{user.name}</span>
            {userActivity[user.id] === 'typing' && <TypingIndicator />}
          </div>
        ))}
      </div>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.userId}`}>
            <Avatar src={getUserAvatar(msg.userId)} />
            <div className="content">{msg.content}</div>
          </div>
        ))}
      </div>

      <button onClick={() => inviteUser(email)}>
        Invite Collaborator
      </button>
    </div>
  )
}
```

### 5. Progress Indicators

```tsx
import { useStreamingProgress } from '@hanzo/live'

function ProgressiveChat() {
  const {
    progress,        // 0-100 completion percentage
    estimatedTime,   // ETA in seconds
    tokensGenerated, // Tokens so far
    totalTokens      // Expected total
  } = useStreamingProgress({
    model: 'llama-3-70b',
    maxTokens: 1000
  })

  return (
    <div>
      <ProgressBar value={progress} max={100} />
      <div className="stats">
        <span>{tokensGenerated} / {totalTokens} tokens</span>
        <span>ETA: {estimatedTime}s</span>
      </div>
    </div>
  )
}
```

## Advanced Patterns

### Streaming with Error Recovery

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  
  // Automatic reconnection
  reconnect: {
    enabled: true,
    maxAttempts: 5,
    delayMs: 1000,
    exponentialBackoff: true
  },
  
  // Resume incomplete streams
  resumeOnReconnect: true
})

client.on('error', (error) => {
  console.error('Stream error:', error)
  // Error handled, stream continues
})

client.on('reconnect', (attempt) => {
  console.log(`Reconnecting (attempt ${attempt})...`)
})

client.on('reconnected', () => {
  console.log('Reconnected, resuming stream')
})

// Start streaming with error handling
try {
  for await (const token of client.stream({...})) {
    display(token)
  }
} catch (error) {
  if (error.code === 'STREAM_INTERRUPTED') {
    // Stream was interrupted but will resume
    await client.waitForReconnect()
  }
}
```

### Streaming with Caching

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  
  // Enable response caching
  cache: {
    enabled: true,
    backend: 'redis',
    ttl: 3600,  // 1 hour
    
    // Semantic caching (cache similar queries)
    semantic: true,
    similarityThreshold: 0.95
  }
})

// First request streams from model
const stream1 = await client.stream({
  messages: [{ role: 'user', content: 'What is 2+2?' }]
})

for await (const token of stream1) {
  console.log(token)
}

// Second request streams from cache (instant)
const stream2 = await client.stream({
  messages: [{ role: 'user', content: 'What is 2+2?' }]
})

for await (const token of stream2) {
  console.log(token)  // Same output, but instant
}
```

### Server-Sent Events Alternative

```typescript
// For clients that don't support WebSockets
import { HanzoLiveSSE } from '@hanzo/live-client'

const client = new HanzoLiveSSE({
  url: 'https://api.hanzo.ai/live'
})

// SSE streaming (one-way, server → client)
const eventSource = await client.stream({
  messages: [{ role: 'user', content: 'Hello' }]
})

eventSource.addEventListener('token', (event) => {
  console.log('Token:', event.data)
})

eventSource.addEventListener('complete', (event) => {
  console.log('Stream complete')
  eventSource.close()
})

eventSource.addEventListener('error', (event) => {
  console.error('Stream error:', event)
})
```

### Multiplexed Streams

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  multiplexing: true  // Multiple streams over one WebSocket
})

// Start multiple streams concurrently
const [stream1, stream2, stream3] = await Promise.all([
  client.stream({ messages: [{ role: 'user', content: 'Query 1' }] }),
  client.stream({ messages: [{ role: 'user', content: 'Query 2' }] }),
  client.stream({ messages: [{ role: 'user', content: 'Query 3' }] })
])

// Process streams in parallel
await Promise.all([
  processStream(stream1, 'Stream 1'),
  processStream(stream2, 'Stream 2'),
  processStream(stream3, 'Stream 3')
])
```

## Integration with Hanzo Ecosystem

### With Hanzo Node (Local Streaming)

```typescript
// Hanzo Live automatically routes to Hanzo Node for local streaming

import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  
  // Prefer local inference
  preferLocal: true,
  localNodeUrl: 'http://localhost:8080',
  
  // Fallback to cloud if needed
  fallbackToCloud: true
})

// Stream from local Hanzo Node (privacy-first)
const stream = await client.stream({
  messages: [{ role: 'user', content: 'Sensitive data query' }],
  forceLocal: true  // Never use cloud
})
```

### With Hanzo MCP (Tool Streaming)

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  
  // MCP integration for tools
  mcp: {
    enabled: true,
    servers: ['http://localhost:8081']
  }
})

// Stream with MCP tools
const stream = await client.stream({
  messages: [{ role: 'user', content: 'Search the codebase' }],
  tools: 'auto',  // Auto-discover from MCP
  streamToolCalls: true  // Stream tool execution
})

for await (const event of stream) {
  if (event.type === 'token') {
    console.log('Token:', event.data)
  } else if (event.type === 'tool_call') {
    console.log('Tool:', event.tool, event.args)
  } else if (event.type === 'tool_result') {
    console.log('Result:', event.result)
  }
}
```

### With @hanzo/ui Components

```tsx
import { AIChat, StreamingText } from '@hanzo/ui'
import { HanzoLiveProvider } from '@hanzo/live'

function App() {
  return (
    <HanzoLiveProvider
      url="ws://localhost:3001"
      model="llama-3-8b"
    >
      {/* AIChat automatically uses Hanzo Live for streaming */}
      <AIChat 
        streaming  
        showThoughts
        enableTools
      />
      
      {/* Or use individual streaming components */}
      <StreamingText 
        source="hanzo-live"
        onToken={(token) => console.log(token)}
      />
    </HanzoLiveProvider>
  )
}
```

### With Hanzo Python SDK

```python
from hanzo import Hanzo
from hanzo_live import HanzoLiveServer

# Initialize Hanzo SDK for local inference
hanzo = Hanzo(inference_mode='local')

# Create Hanzo Live server that uses SDK
server = HanzoLiveServer(
    hanzo_client=hanzo,
    host='0.0.0.0',
    port=3001,
    
    # Enable multiplexing for concurrent streams
    multiplexing=True,
    max_concurrent_streams=100
)

# Custom stream handler
@server.on_stream
async def handle_stream(messages, model, **kwargs):
    # Use Hanzo SDK for inference
    response = hanzo.chat.completions.create(
        messages=messages,
        model=model,
        stream=True
    )
    
    # Stream tokens to client
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield {'type': 'token', 'data': token}
    
    yield {'type': 'complete'}

# Start server
await server.serve()
```

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile for Hanzo Live server
FROM node:20-alpine

WORKDIR /app

# Install Hanzo Live
RUN npm install -g @hanzo/live-server

# Copy configuration
COPY hanzo-live.config.js .

# Expose WebSocket port
EXPOSE 3001

# Start server
CMD ["hanzo-live", "serve", \
     "--port", "3001", \
     "--node-url", "http://hanzo-node:8080", \
     "--redis", "redis://redis:6379"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  hanzo-live:
    build: .
    ports:
      - "3001:3001"
    depends_on:
      - hanzo-node
      - redis
    environment:
      - NODE_URL=http://hanzo-node:8080
      - REDIS_URL=redis://redis:6379
      - MAX_CONNECTIONS=10000
      - STREAM_TIMEOUT=300

  hanzo-node:
    image: hanzoai/node:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - GPU_LAYERS=auto

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Kubernetes Deployment

```yaml
# hanzo-live-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-live
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hanzo-live
  template:
    metadata:
      labels:
        app: hanzo-live
    spec:
      containers:
      - name: hanzo-live
        image: hanzoai/live-server:latest
        ports:
        - containerPort: 3001
        env:
        - name: NODE_URL
          value: "http://hanzo-node:8080"
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: hanzo-live
spec:
  type: LoadBalancer
  ports:
  - port: 3001
    targetPort: 3001
    protocol: TCP
  selector:
    app: hanzo-live
```

### Edge Deployment (Cloudflare Workers)

```typescript
// Hanzo Live on Cloudflare Workers for ultra-low latency
import { HanzoLiveEdge } from '@hanzo/live-edge'

export default {
  async fetch(request: Request, env: Env) {
    const upgradeHeader = request.headers.get('Upgrade')
    
    if (upgradeHeader === 'websocket') {
      // Handle WebSocket upgrade
      const server = new HanzoLiveEdge({
        nodeUrl: env.HANZO_NODE_URL,
        kv: env.CACHE_KV,
        durable: env.SESSIONS_DO
      })
      
      return server.handleWebSocket(request)
    }
    
    return new Response('Hanzo Live Edge', { status: 200 })
  }
}
```

## Monitoring and Observability

### Real-Time Metrics

```typescript
import { HanzoLiveClient } from '@hanzo/live-client'

const client = new HanzoLiveClient({
  url: 'ws://localhost:3001',
  
  // Enable metrics collection
  metrics: {
    enabled: true,
    endpoint: 'http://prometheus:9090'
  }
})

// Metrics automatically tracked:
// - hanzo_live_connections_total
// - hanzo_live_streams_active
// - hanzo_live_tokens_per_second
// - hanzo_live_latency_ms (p50, p95, p99)
// - hanzo_live_errors_total
// - hanzo_live_cache_hits_total
```

### Custom Analytics

```typescript
client.on('stream_start', (event) => {
  analytics.track('Stream Started', {
    model: event.model,
    userId: event.userId
  })
})

client.on('stream_complete', (event) => {
  analytics.track('Stream Complete', {
    duration: event.duration,
    tokens: event.totalTokens,
    tokensPerSecond: event.tokensPerSecond
  })
})

client.on('stream_error', (event) => {
  analytics.track('Stream Error', {
    error: event.error,
    model: event.model
  })
})
```

## Best Practices

### 1. Always Show Streaming Indicators

```tsx
// ✅ Good - clear streaming feedback
{isStreaming && <StreamingIndicator />}

// ❌ Avoid - no user feedback
{/* No indicator */}
```

### 2. Handle Disconnections Gracefully

```typescript
// ✅ Good - automatic reconnection
const client = new HanzoLiveClient({
  reconnect: { enabled: true, maxAttempts: 5 },
  resumeOnReconnect: true
})

// ❌ Avoid - no reconnection logic
const client = new HanzoLiveClient({})
```

### 3. Use Appropriate Buffer Sizes

```typescript
// ✅ Good - balanced buffering
const client = new HanzoLiveClient({
  bufferSize: 16,  // Buffer 16 tokens
  flushInterval: 50  // Flush every 50ms
})

// ❌ Avoid - no buffering (choppy) or too much (laggy)
```

### 4. Implement Proper Error States

```tsx
// ✅ Good - comprehensive error handling
{error && <ErrorMessage error={error} onRetry={retry} />}

// ❌ Avoid - silent failures
{/* No error display */}
```

### 5. Clean Up Resources

```typescript
// ✅ Good - proper cleanup
useEffect(() => {
  const client = new HanzoLiveClient({...})
  return () => client.disconnect()
}, [])

// ❌ Avoid - memory leaks
// (no cleanup on unmount)
```

## Next Steps

1. **Read @hanzo/ui Documentation** - Streaming UI components
2. **Explore Hanzo Node** - Local inference for streaming
3. **Check Hanzo MCP** - Tool integration in streams
4. **See WebSocket Guide** - Advanced WebSocket patterns

## Related Skills

- **hanzo-ui.md** - React streaming components
- **hanzo-node.md** - Local inference infrastructure
- **hanzo-mcp.md** - Tool calling in streams
- **realtime/INDEX.md** - WebSocket and SSE patterns

---

**Remember**: Hanzo Live enables responsive, collaborative AI experiences with ultra-low latency streaming - use it for any real-time AI application.
