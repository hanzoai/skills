# Hanzo Insights - Product Analytics Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-o11y.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Insights is a **full product analytics platform** — fork of PostHog with feature flags, session recording, A/B testing, InsightsQL, and ClickHouse-powered analytics. Self-hostable with privacy-first architecture.

**NOTE**: This is a **PostHog fork**, NOT just analytics SDKs. It includes the full PostHog feature set: event analytics, session recording, feature flags, A/B testing, heatmaps, and a custom query language (InsightsQL).

### Why Hanzo Insights?

- **PostHog fork**: Full product analytics suite, self-hosted
- **Feature flags**: Gradual rollouts, user targeting, percentage-based
- **Session recording**: Replay user sessions with DOM snapshots
- **A/B testing**: Experiment framework with statistical significance
- **InsightsQL**: Custom query language for complex analytics queries
- **ClickHouse**: High-performance columnar storage for event data
- **Privacy-first**: No third-party data sharing, self-hostable
- **Multi-language SDKs**: Go, JavaScript/TypeScript, Rust

## When to use

- Product analytics beyond basic event tracking
- Feature flag management and gradual rollouts
- A/B testing with statistical analysis
- Session recording and user behavior replay
- Funnel analysis and conversion tracking
- Self-hosted analytics without third-party dependencies

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/insights` |
| Upstream | PostHog fork |
| Storage | ClickHouse (events), PostgreSQL (metadata) |
| Dashboard | Integrated in Hanzo Cloud |
| API Endpoint | `https://api.hanzo.ai/v1/insights` |

| SDK | Package |
|-----|---------|
| Go | `github.com/hanzoai/insights-go` |
| JavaScript | `@hanzo/insights` |
| Rust | `hanzo-insights` |

## Features

### Event Analytics

```typescript
import { Insights } from "@hanzo/insights"

const insights = new Insights({
  apiKey: process.env.HANZO_API_KEY,
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
```

### Feature Flags

```typescript
// Check feature flag
const showNewUI = await insights.isFeatureEnabled("new-dashboard-ui", "user_123")

if (showNewUI) {
  renderNewDashboard()
} else {
  renderClassicDashboard()
}

// Get feature flag payload (variant data)
const variant = await insights.getFeatureFlag("pricing-experiment", "user_123")
// Returns: { key: "variant-b", payload: { price: 29.99, cta: "Start Free Trial" } }

// Feature flag with properties for targeting
insights.identify("user_123", {
  plan: "pro",
  company_size: 50,
  country: "US",
})
// Flag rules can target based on these properties
```

### Session Recording

```typescript
// Enable session recording
const insights = new Insights({
  apiKey: process.env.HANZO_API_KEY,
  sessionRecording: {
    enabled: true,
    sampleRate: 0.1,        // Record 10% of sessions
    maskTextContent: true,   // Mask sensitive text
    maskInputs: true,        // Mask form inputs
    blockSelectors: [".pii"], // Block specific elements
  },
})

// Recordings available in dashboard for replay
// Includes DOM snapshots, clicks, scrolls, console logs
```

### A/B Testing

```typescript
// Run experiment
const experiment = await insights.getExperiment("checkout-flow", "user_123")

switch (experiment.variant) {
  case "control":
    showClassicCheckout()
    break
  case "streamlined":
    showStreamlinedCheckout()
    break
  case "one-click":
    showOneClickCheckout()
    break
}

// Track conversion
insights.track("purchase_completed", {
  experiment: "checkout-flow",
  variant: experiment.variant,
  revenue: 49.99,
})
```

### InsightsQL

Custom query language for complex analytics:

```sql
-- Funnel analysis
SELECT funnel(
  step("page_view", properties.path = "/pricing"),
  step("button_click", properties.button = "start_trial"),
  step("purchase_completed")
) FROM events
WHERE timestamp > now() - interval 30 day
GROUP BY properties.plan

-- Retention cohorts
SELECT retention(
  first("signup"),
  returning("page_view"),
  period("week")
) FROM events
WHERE timestamp > now() - interval 90 day

-- User paths
SELECT paths(
  start("page_view"),
  end("purchase_completed"),
  max_steps(5)
) FROM events
WHERE timestamp > now() - interval 7 day
```

## Go SDK

```go
import insights "github.com/hanzoai/insights-go"

client := insights.New(insights.Config{
    APIKey:   os.Getenv("HANZO_API_KEY"),
})
defer client.Close()

// Track
client.Track("api_call", insights.Properties{
    "endpoint": "/v1/chat/completions",
    "model":    "zen-70b",
    "latency":  142,
})

// Feature flags
enabled := client.IsFeatureEnabled("new-api", "user_123")

// Identify
client.Identify("user_123", insights.Traits{
    "plan": "enterprise",
})
```

## Rust SDK

```rust
use hanzo_insights::Client;

let client = Client::new(std::env::var("HANZO_API_KEY")?);

client.track("inference", &[
    ("model", "zen-70b"),
    ("latency_ms", "142"),
]).await?;

let enabled = client.is_feature_enabled("new-api", "user_123").await?;

client.flush().await?;
```

## React Integration

```typescript
import { InsightsProvider, useInsights, useFeatureFlag } from "@hanzo/insights/react"

function App() {
  return (
    <InsightsProvider apiKey={process.env.NEXT_PUBLIC_HANZO_KEY}>
      <MyApp />
    </InsightsProvider>
  )
}

function Dashboard() {
  const { track } = useInsights()
  const showNewUI = useFeatureFlag("new-dashboard-ui")

  return (
    <div>
      {showNewUI ? <NewDashboard /> : <ClassicDashboard />}
      <button onClick={() => track("cta_click", { variant: "hero" })}>
        Get Started
      </button>
    </div>
  )
}
```

## Self-Hosting

```yaml
# compose.yml
services:
  insights:
    image: hanzoai/insights:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - CLICKHOUSE_URL=clickhouse://...
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - clickhouse
      - postgres

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: insights
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  clickhouse_data:
  pg_data:
```

## Privacy

- **No cookies by default** — uses anonymous IDs
- **IP anonymization** — last octet zeroed before storage
- **Data retention** — configurable (default: 90 days)
- **Self-hostable** — run your own endpoint
- **GDPR compliant** — delete user data via API
- **Session recording masking** — auto-mask inputs and PII elements

## Related Skills

- `hanzo/hanzo-o11y.md` - Technical observability (metrics, traces — different from product analytics)
- `hanzo/hanzo-console.md` - LLM-specific observability
- `hanzo/hanzo-cloud.md` - Dashboard with analytics views

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: analytics, posthog, feature-flags, ab-testing, session-recording, clickhouse
**Prerequisites**: JavaScript/Go/Rust basics, analytics concepts
