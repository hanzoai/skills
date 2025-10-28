# @hanzo/ui - AI+Blockchain Component Library

**Category**: Hanzo Ecosystem  
**Related Skills**: `frontend/react-component-patterns.md`, `frontend/nextjs-app-router.md`, `elegant-design/SKILL.md`

## Overview

@hanzo/ui is a production-ready React component library designed specifically for **AI-powered and blockchain-enabled applications**. Rather than building UI from scratch, use these battle-tested, accessible, and elegantly designed components to ship faster.

### Philosophy

- **Composable**: Mix and match components for any use case
- **Accessible**: WCAG 2.1 AA compliant out of the box
- **Elegant**: World-class design inspired by modern AI interfaces
- **Type-Safe**: Full TypeScript support with IntelliSense
- **Framework-Agnostic**: Works with Next.js, Vite, Remix

### Built On

- **React 18+**: Server Components, Suspense, Transitions
- **Tailwind CSS**: Utility-first styling with CSS variables
- **Radix UI**: Unstyled accessible primitives
- **Framer Motion**: Smooth animations and gestures

## Quick Start

### Installation

```bash
# pnpm (recommended)
pnpm add @hanzo/ui

# npm
npm install @hanzo/ui

# yarn
yarn add @hanzo/ui
```

### Setup Tailwind

```javascript
// tailwind.config.js
import { hanzoPreset } from '@hanzo/ui/tailwind'

export default {
  presets: [hanzoPreset],
  content: [
    './app/**/*.{ts,tsx}',
    './node_modules/@hanzo/ui/**/*.{js,ts,jsx,tsx}'
  ]
}
```

### Basic Usage

```typescript
import { Button, Card, AIChat } from '@hanzo/ui'

export default function App() {
  return (
    <Card>
      <AIChat 
        model="llama-3-8b"
        onMessage={handleMessage}
        streaming
      />
      <Button variant="primary">Send</Button>
    </Card>
  )
}
```

## Core Components

### AI Components

#### AIChat

Full-featured AI chat interface with streaming, history, and model selection.

```typescript
import { AIChat } from '@hanzo/ui'

<AIChat
  model="llama-3-8b"
  inference={customInferenceFunc}  // or use default
  onMessage={(message) => console.log(message)}
  streaming={true}
  darkMode={true}
  placeholder="Ask me anything..."
  systemPrompt="You are a helpful assistant"
/>
```

**Props**:
- `model`: Model ID (string)
- `inference`: Custom inference function
- `onMessage`: Callback for new messages
- `streaming`: Enable token streaming
- `darkMode`: Dark theme support
- `systemPrompt`: System instructions
- `placeholder`: Input placeholder text

**With Hanzo Node**:
```typescript
import { AIChat } from '@hanzo/ui'
import { useHanzoNode } from '@hanzo/ui/hooks'

export function LocalAIChat() {
  const node = useHanzoNode({ url: 'http://localhost:8080' })
  
  return (
    <AIChat 
      inference={node.infer}
      model={node.selectedModel}
      streaming
    />
  )
}
```

#### ModelSelector

Dropdown to choose between available AI models.

```typescript
import { ModelSelector } from '@hanzo/ui'

<ModelSelector
  models={[
    { id: 'llama-3-8b', name: 'Llama 3 8B', size: '4.6GB' },
    { id: 'mistral-7b', name: 'Mistral 7B', size: '4.1GB' }
  ]}
  selected="llama-3-8b"
  onSelect={(modelId) => setModel(modelId)}
  showSize={true}
  showLatency={true}
/>
```

#### StreamingText

Animated streaming text with typewriter effect.

```typescript
import { StreamingText } from '@hanzo/ui'

<StreamingText
  text={streamingResponse}
  speed={50}  // ms per character
  onComplete={() => console.log('Done')}
/>
```

#### TokenUsage

Display token usage with visual meters.

```typescript
import { TokenUsage } from '@hanzo/ui'

<TokenUsage
  tokens={{
    prompt: 120,
    completion: 450,
    total: 570
  }}
  limit={4096}
  showCost={true}
  costPerToken={0.00001}
/>
```

### Blockchain Components

#### WalletConnect

One-click wallet connection with multi-chain support.

```typescript
import { WalletConnect } from '@hanzo/ui'

<WalletConnect
  chains={['ethereum', 'lux', 'polygon']}
  onConnect={(address, chain) => {
    console.log(`Connected: ${address} on ${chain}`)
  }}
  onDisconnect={() => console.log('Disconnected')}
  showBalance={true}
/>
```

#### TransactionStatus

Visual transaction progress indicator.

```typescript
import { TransactionStatus } from '@hanzo/ui'

<TransactionStatus
  status="pending"  // pending | confirmed | failed
  txHash="0x123..."
  explorerUrl="https://etherscan.io/tx/0x123..."
  confirmations={12}
  requiredConfirmations={20}
/>
```

#### GasEstimator

Real-time gas price estimation and fee display.

```typescript
import { GasEstimator } from '@hanzo/ui'

<GasEstimator
  chain="ethereum"
  priority="medium"  // slow | medium | fast
  onEstimate={(gasPrice, totalCost) => {
    console.log(`Gas: ${gasPrice} gwei, Cost: ${totalCost} ETH`)
  }}
/>
```

### Layout Components

#### Dashboard

Full application shell with sidebar, header, and content area.

```typescript
import { Dashboard } from '@hanzo/ui'

<Dashboard
  sidebar={<Sidebar />}
  header={<Header />}
  sidebarCollapsible={true}
>
  <YourContent />
</Dashboard>
```

#### AppShell

Flexible application layout.

```typescript
import { AppShell } from '@hanzo/ui'

<AppShell
  nav={<Navigation />}
  header={<Header />}
  footer={<Footer />}
  aside={<Sidebar />}
>
  {children}
</AppShell>
```

#### Card

Versatile card component for content grouping.

```typescript
import { Card } from '@hanzo/ui'

<Card
  title="AI Metrics"
  subtitle="Last 24 hours"
  actions={<Button>Refresh</Button>}
  footer={<p>Updated 2m ago</p>}
>
  <MetricsChart data={metrics} />
</Card>
```

### Data Display

#### DataTable

Feature-rich table with sorting, filtering, and pagination.

```typescript
import { DataTable } from '@hanzo/ui'

<DataTable
  columns={[
    { id: 'name', header: 'Name', sortable: true },
    { id: 'status', header: 'Status', filterable: true },
    { id: 'created', header: 'Created', type: 'date' }
  ]}
  data={items}
  onSort={(column, direction) => {}}
  onFilter={(filters) => {}}
  pagination={{ page: 1, pageSize: 20 }}
/>
```

#### Chart

Recharts-based chart components.

```typescript
import { LineChart, BarChart, PieChart } from '@hanzo/ui'

<LineChart
  data={timeSeriesData}
  xAxis="timestamp"
  yAxis="value"
  title="Token Usage Over Time"
  height={300}
/>
```

#### MetricsCard

Display key performance indicators.

```typescript
import { MetricsCard } from '@hanzo/ui'

<MetricsCard
  title="Active Users"
  value={12_459}
  change={+15.3}  // % change
  trend="up"
  subtitle="Last 30 days"
/>
```

### Form Components

#### Form

Form with validation and error handling.

```typescript
import { Form, Input, Select, Button } from '@hanzo/ui'
import { z } from 'zod'

const schema = z.object({
  email: z.string().email(),
  model: z.enum(['llama-3-8b', 'mistral-7b'])
})

<Form
  schema={schema}
  onSubmit={(data) => console.log(data)}
>
  <Input name="email" label="Email" />
  <Select
    name="model"
    label="Model"
    options={[
      { value: 'llama-3-8b', label: 'Llama 3 8B' },
      { value: 'mistral-7b', label: 'Mistral 7B' }
    ]}
  />
  <Button type="submit">Submit</Button>
</Form>
```

## Design System

### Theming

@hanzo/ui uses CSS variables for theming:

```css
/* tailwind.config.js or global.css */
:root {
  --hanzo-primary: 262.1 83.3% 57.8%;
  --hanzo-secondary: 220 14.3% 95.9%;
  --hanzo-accent: 12 6.5% 15.1%;
  --hanzo-background: 0 0% 100%;
  --hanzo-foreground: 222.2 84% 4.9%;
  --hanzo-border: 214.3 31.8% 91.4%;
}

.dark {
  --hanzo-primary: 263.4 70% 50.4%;
  --hanzo-background: 222.2 84% 4.9%;
  --hanzo-foreground: 210 40% 98%;
}
```

### Dark Mode

Automatic dark mode support with next-themes:

```typescript
import { ThemeProvider } from '@hanzo/ui'

<ThemeProvider>
  <YourApp />
</ThemeProvider>
```

Toggle theme programmatically:
```typescript
import { useTheme } from '@hanzo/ui'

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  
  return (
    <Button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
      Toggle Theme
    </Button>
  )
}
```

### Responsive Design

All components are mobile-first and responsive:

```typescript
<Card className="col-span-1 md:col-span-2 lg:col-span-3">
  {/* Spans 1 column on mobile, 2 on tablet, 3 on desktop */}
</Card>
```

### Accessibility

- ✅ Keyboard navigation (Tab, Arrow keys, Enter, Escape)
- ✅ Screen reader support (ARIA labels, roles, descriptions)
- ✅ Focus management (visible focus rings, focus traps)
- ✅ Color contrast (WCAG AA compliant)
- ✅ Reduced motion support (respects prefers-reduced-motion)

## Integration Patterns

### With Hanzo Live (Real-Time Updates)

```typescript
import { AIChat, MetricsCard } from '@hanzo/ui'
import { useHanzoLive } from '@hanzo/ui/hooks'

export function LiveDashboard() {
  const { data, status } = useHanzoLive({
    pipeline: 'metrics-stream',
    url: 'ws://localhost:3001'
  })
  
  return (
    <div className="grid gap-4">
      <MetricsCard
        title="Active Requests"
        value={data?.activeRequests}
        live  // Shows pulse indicator
      />
      
      <AIChat 
        streaming
        onStream={(token) => {
          // Real-time token streaming from Hanzo Live
        }}
      />
    </div>
  )
}
```

### With Hanzo Node (Local AI)

```typescript
import { AIChat, ModelSelector, TokenUsage } from '@hanzo/ui'
import { useHanzoNode } from '@hanzo/ui/hooks'

export function LocalAIInterface() {
  const node = useHanzoNode({
    url: 'http://localhost:8080',
    autoReconnect: true
  })
  
  if (node.status === 'disconnected') {
    return <Card>Connecting to Hanzo Node...</Card>
  }
  
  return (
    <Dashboard>
      <ModelSelector
        models={node.models}
        selected={node.selectedModel}
        onSelect={node.selectModel}
      />
      
      <AIChat
        inference={node.infer}
        model={node.selectedModel}
        streaming
      />
      
      <TokenUsage
        tokens={node.tokenUsage}
        limit={node.contextSize}
      />
    </Dashboard>
  )
}
```

### With Hanzo Python SDK

```typescript
// Server component (Next.js)
import { AIChat } from '@hanzo/ui'

async function getModels() {
  const response = await fetch('http://localhost:8000/api/models')
  return response.json()
}

export default async function Page() {
  const models = await getModels()
  
  return <AIChat models={models} />
}
```

### With Blockchain

```typescript
import { WalletConnect, TransactionStatus, GasEstimator } from '@hanzo/ui'
import { useWallet } from '@hanzo/ui/hooks'

export function BlockchainDashboard() {
  const { address, chain, balance, sendTransaction } = useWallet()
  
  const handleTransaction = async () => {
    const tx = await sendTransaction({
      to: '0x...',
      value: ethers.utils.parseEther('0.1'),
      data: '0x...'
    })
    
    // tx.hash, tx.status, etc.
  }
  
  return (
    <>
      <WalletConnect
        chains={['ethereum', 'lux']}
        showBalance
      />
      
      {address && (
        <>
          <GasEstimator chain={chain} />
          <Button onClick={handleTransaction}>
            Send Transaction
          </Button>
        </>
      )}
    </>
  )
}
```

## Advanced Usage

### Custom Inference Function

```typescript
import { AIChat } from '@hanzo/ui'

async function customInference(
  model: string,
  messages: Message[],
  options?: InferenceOptions
): Promise<string | AsyncIterable<string>> {
  // Your custom logic
  const response = await fetch('/api/inference', {
    method: 'POST',
    body: JSON.stringify({ model, messages, ...options })
  })
  
  if (options?.stream) {
    return streamResponse(response)
  }
  
  const data = await response.json()
  return data.text
}

<AIChat
  inference={customInference}
  model="custom-model"
  streaming
/>
```

### Custom Theme

```typescript
import { ThemeProvider, createTheme } from '@hanzo/ui'

const customTheme = createTheme({
  colors: {
    primary: { h: 220, s: 100, l: 50 },
    secondary: { h: 180, s: 80, l: 60 }
  },
  fonts: {
    sans: 'Inter, system-ui, sans-serif',
    mono: 'JetBrains Mono, monospace'
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '1rem'
  }
})

<ThemeProvider theme={customTheme}>
  <App />
</ThemeProvider>
```

### Component Composition

```typescript
import { Card, AIChat, TokenUsage, Button } from '@hanzo/ui'

function AIAssistantCard() {
  return (
    <Card
      title="AI Assistant"
      actions={
        <Button variant="outline" size="sm">
          Clear Chat
        </Button>
      }
      footer={
        <TokenUsage
          tokens={{ prompt: 100, completion: 300, total: 400 }}
          limit={4096}
        />
      }
    >
      <AIChat model="llama-3-8b" streaming />
    </Card>
  )
}
```

## Production Patterns

### Error Boundaries

```typescript
import { ErrorBoundary } from '@hanzo/ui'

<ErrorBoundary
  fallback={<div>Something went wrong</div>}
  onError={(error) => logToSentry(error)}
>
  <AIChat />
</ErrorBoundary>
```

### Loading States

```typescript
import { Skeleton } from '@hanzo/ui'

{loading ? (
  <Skeleton className="h-96 w-full" />
) : (
  <AIChat {...props} />
)}
```

### Optimistic Updates

```typescript
import { AIChat } from '@hanzo/ui'

<AIChat
  onMessage={(message) => {
    // Optimistically add message
    setMessages(prev => [...prev, message])
    
    // Send to server
    sendMessage(message).catch(() => {
      // Rollback on error
      setMessages(prev => prev.filter(m => m.id !== message.id))
    })
  }}
/>
```

## Performance

### Code Splitting

```typescript
import dynamic from 'next/dynamic'

const AIChat = dynamic(() => import('@hanzo/ui').then(mod => mod.AIChat), {
  loading: () => <Skeleton className="h-96" />,
  ssr: false  // Client-side only if needed
})
```

### Bundle Size

```
@hanzo/ui (full):          234 KB
@hanzo/ui (tree-shaken):    45 KB (typical usage)

Components:
- AIChat:                   12 KB
- ModelSelector:             3 KB
- WalletConnect:            18 KB
- DataTable:                15 KB
```

## TypeScript Support

Full type definitions included:

```typescript
import type {
  Message,
  InferenceOptions,
  Model,
  WalletConfig,
  Theme
} from '@hanzo/ui/types'

const messages: Message[] = [
  {
    role: 'user',
    content: 'Hello',
    timestamp: Date.now()
  }
]
```

## Related Skills

**Prerequisites**:
- `frontend/react-component-patterns.md` - React fundamentals
- `frontend/nextjs-app-router.md` - Next.js integration
- `elegant-design/SKILL.md` - Design principles

**Integration**:
- `hanzo/hanzo-node.md` - Backend AI infrastructure
- `hanzo/hanzo-live.md` - Real-time pipelines
- `hanzo/python-sdk.md` - API backend
- `hanzo/hanzo-mcp.md` - MCP integration

**Next Steps**:
- `frontend/web-accessibility.md` - Accessibility best practices
- `frontend/react-state-management.md` - Advanced state patterns
- `frontend/nextjs-seo.md` - SEO optimization

---

**Last Updated**: 2025-10-28  
**Category**: Hanzo Ecosystem  
**Related**: frontend, ai, blockchain  
**Prerequisites**: React, TypeScript, Tailwind CSS
