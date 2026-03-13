# Hanzo Insights - Multi-Language Analytics SDK

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-o11y.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Insights provides **privacy-first analytics** across Go, JavaScript, and Rust SDKs. Event tracking, user identification, funnel analysis, and behavioral analytics — self-hostable with cloud dashboard integration.

### Why Hanzo Insights?

- **Privacy-first**: No third-party data sharing, self-hostable
- **Multi-language**: Go, JavaScript/TypeScript, Rust SDKs
- **Lightweight**: Minimal bundle size, async batching
- **Real-time**: Events available immediately in dashboard
- **Composable**: Works alongside OpenTelemetry (o11y) for full observability

## When to use

- Tracking user behavior in web/mobile/desktop apps
- Funnel analysis and conversion tracking
- Feature usage analytics
- A/B test measurement
- Product analytics without third-party dependencies

## Quick reference

| SDK | Package | Repo |
|-----|---------|------|
| Go | `github.com/hanzoai/insights-go` | `github.com/hanzoai/insights-go` |
| JavaScript | `@hanzo/insights` | `github.com/hanzoai/insights-js` |
| Rust | `hanzo-insights` | `github.com/hanzoai/insights-rs` |

| Item | Value |
|------|-------|
| API Endpoint | `https://api.hanzo.ai/v1/insights` |
| Dashboard | Integrated in Hanzo Cloud |
| Batch interval | 5 seconds (configurable) |
| Max batch size | 100 events |

## JavaScript SDK

```bash
pnpm add @hanzo/insights
```

```typescript
import { Insights } from "@hanzo/insights"

const insights = new Insights({
  apiKey: process.env.HANZO_API_KEY,
  endpoint: "https://api.hanzo.ai/v1/insights",  // default
  batchSize: 50,           // Events per batch (default: 100)
  flushInterval: 3000,     // Flush every 3s (default: 5000)
})

// Track events
insights.track("page_view", {
  path: "/dashboard",
  referrer: document.referrer,
})

insights.track("button_click", {
  button: "upgrade",
  page: "/pricing",
})

// Identify user
insights.identify("user_123", {
  name: "Alice",
  plan: "pro",
  created_at: "2026-01-15",
})

// Group (organization)
insights.group("org_456", {
  name: "Acme Corp",
  plan: "enterprise",
})

// Feature flags
insights.track("feature_used", {
  feature: "ai_search",
  variant: "v2",
})

// Revenue
insights.track("purchase", {
  amount: 2999,
  currency: "usd",
  product: "pro_plan",
})

// Flush before page unload
window.addEventListener("beforeunload", () => {
  insights.flush()
})
```

### React Integration

```typescript
import { InsightsProvider, useInsights } from "@hanzo/insights/react"

function App() {
  return (
    <InsightsProvider apiKey={process.env.NEXT_PUBLIC_HANZO_KEY}>
      <MyApp />
    </InsightsProvider>
  )
}

function Dashboard() {
  const { track } = useInsights()

  return (
    <button onClick={() => track("cta_click", { variant: "hero" })}>
      Get Started
    </button>
  )
}
```

## Go SDK

```bash
go get github.com/hanzoai/insights-go
```

```go
import insights "github.com/hanzoai/insights-go"

client := insights.New(insights.Config{
    APIKey:   os.Getenv("HANZO_API_KEY"),
    Endpoint: "https://api.hanzo.ai/v1/insights",
})
defer client.Close() // Flush remaining events

// Track
client.Track("api_call", insights.Properties{
    "endpoint": "/v1/chat/completions",
    "model":    "zen-70b",
    "latency":  142,
    "tokens":   1500,
})

// Identify
client.Identify("user_123", insights.Traits{
    "plan":  "enterprise",
    "email": "user@example.com",
})

// Group
client.Group("org_456", insights.Traits{
    "name": "Acme Corp",
})

// Batch track
client.BatchTrack([]insights.Event{
    {Name: "request", Props: insights.Properties{"path": "/api"}},
    {Name: "request", Props: insights.Properties{"path": "/dashboard"}},
})
```

## Rust SDK

```toml
# Cargo.toml
[dependencies]
hanzo-insights = "0.1"
```

```rust
use hanzo_insights::Client;

let client = Client::new(std::env::var("HANZO_API_KEY")?);

client.track("inference", &[
    ("model", "zen-70b"),
    ("latency_ms", "142"),
    ("tokens", "1500"),
]).await?;

client.identify("user_123", &[
    ("plan", "pro"),
]).await?;

// Flush on shutdown
client.flush().await?;
```

## Event Schema

```json
{
  "event": "page_view",
  "timestamp": "2026-03-13T12:00:00Z",
  "user_id": "user_123",
  "anonymous_id": "anon_abc",
  "properties": {
    "path": "/dashboard",
    "referrer": "https://google.com"
  },
  "context": {
    "ip": "auto",
    "user_agent": "auto",
    "locale": "en-US"
  }
}
```

## Privacy

- **No cookies by default** — uses anonymous IDs
- **IP anonymization** — last octet zeroed before storage
- **Data retention** — configurable (default: 90 days)
- **Self-hostable** — run your own endpoint
- **GDPR compliant** — delete user data via API

## Related Skills

- `hanzo/hanzo-o11y.md` - Technical observability (metrics, traces)
- `hanzo/hanzo-console.md` - LLM-specific observability
- `hanzo/hanzo-cloud.md` - Dashboard with analytics views

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: analytics, tracking, privacy, events
**Prerequisites**: JavaScript/Go/Rust basics
