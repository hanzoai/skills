---
name: hanzo-insights
description: "Product analytics, feature flags, A/B testing, and session recording with Hanzo Insights. Track events, create funnels, manage rollouts."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "posthog",
              "label": "Install Hanzo Insights Client (pip)",
            },
          ],
      },
  }
---

# Hanzo Insights — Product Analytics

`pip install posthog`

Full-stack product analytics: event tracking, funnels, session recording, feature flags, A/B testing, and cohort analysis.

## Quick Start

```python
import posthog

posthog.project_api_key = "phc_..."
posthog.host = "https://insights.hanzo.ai"

# Track event
posthog.capture(
    distinct_id="user-123",
    event="button_clicked",
    properties={"button": "deploy", "page": "/dashboard"}
)
```

## Event Tracking

```python
# Track with properties
posthog.capture("user-123", "deployment_started", {
    "project": "my-app",
    "environment": "production",
    "method": "docker"
})

# Track page view
posthog.capture("user-123", "$pageview", {"$current_url": "/settings"})

# Identify user
posthog.identify("user-123", {
    "email": "alice@hanzo.ai",
    "name": "Alice",
    "plan": "enterprise"
})
```

## Feature Flags

```python
# Check feature flag
if posthog.feature_enabled("new-dashboard", "user-123"):
    show_new_dashboard()
else:
    show_old_dashboard()

# Get flag payload
payload = posthog.get_feature_flag_payload("new-dashboard", "user-123")

# Get all flags for user
flags = posthog.get_all_flags("user-123")
```

## A/B Testing

```python
# Get experiment variant
variant = posthog.get_feature_flag("pricing-test", "user-123")

if variant == "control":
    show_original_pricing()
elif variant == "test":
    show_new_pricing()
```

## Group Analytics

```python
# Track group events (teams, organizations)
posthog.group_identify("company", "hanzo-ai", {
    "name": "Hanzo AI",
    "plan": "enterprise",
    "employee_count": 50
})

posthog.capture("user-123", "feature_used", groups={"company": "hanzo-ai"})
```

## LLM Analytics

```python
# Track AI/LLM events
posthog.capture("user-123", "llm_query", {
    "model": "gpt-4",
    "tokens_used": 1500,
    "latency_ms": 2300,
    "cost_usd": 0.045
})
```

## Ports

- API Backend: `8000`
- Dashboard: `3000`

## Environment Variables

```bash
POSTHOG_API_KEY=phc_...
POSTHOG_HOST=https://insights.hanzo.ai
```
