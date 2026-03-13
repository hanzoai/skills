# Hanzo Insights - Multi-Language Analytics SDK

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cloud.md`, `hanzo/hanzo-o11y.md`

## Overview

Hanzo Insights provides **privacy-first analytics** across Go, JavaScript, and Rust SDKs. Event tracking, user analytics, funnel analysis — self-hostable with cloud dashboard integration.

## Quick reference

| SDK | Package | Path |
|-----|---------|------|
| Go | `github.com/hanzoai/insights-go` | ``github.com/hanzoai/insights-go`` |
| JS | `@hanzo/insights` | ``github.com/hanzoai/insights-js`` |
| Rust | `hanzo-insights` | ``github.com/hanzoai/insights-rs`` |

## One-file quickstart

### JavaScript

```typescript
import { Insights } from "@hanzo/insights"

const insights = new Insights({
  apiKey: process.env.HANZO_API_KEY,
  endpoint: "https://api.hanzo.ai/v1/insights",
})

insights.track("page_view", { path: "/dashboard", userId: "user_123" })
insights.identify("user_123", { name: "Alice", plan: "pro" })
```

### Go

```go
import insights "github.com/hanzoai/insights-go"

client := insights.New(os.Getenv("HANZO_API_KEY"))
client.Track("page_view", insights.Properties{"path": "/dashboard"})
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
