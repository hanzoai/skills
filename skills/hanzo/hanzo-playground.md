# Hanzo Playground - Bot Control Plane

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-bot.md`, `hanzo/hanzo-agent.md`, `hanzo/hanzo-operative.md`

## Overview

Hanzo Playground is a **Kubernetes-style control plane for AI bots**. Provides production infrastructure for deploying, orchestrating, and observing multi-bot systems with cryptographic identity (DID/VC), workflow execution, memory scoping, and an embedded web UI. Three-tier monorepo: Go control plane, Python/Go/TypeScript SDKs, React admin interface. Live at `app.hanzo.bot`.

## When to use

- Deploying and orchestrating multi-bot AI systems
- Running multi-step workflows across multiple AI agents
- Central coordination for bots built with Python, Go, or TypeScript SDKs
- Auditing bot executions with cryptographic proofs (DID/VC)
- Provisioning cloud agents (K8s pods or VMs)
- Monitoring bot health, workflow state, and memory via web UI

## Hard requirements

1. **Go 1.23+** for control plane
2. **Node.js 20+** for web UI development
3. **Python 3.8+** for Python SDK
4. **PostgreSQL 15+** for production (SQLite for local dev)

## Quick reference

| Item | Value |
|------|-------|
| URL | `https://app.hanzo.bot` (alias: `playground.hanzo.bot`) |
| Version | 0.1.41-rc.197 |
| Control plane | Go 1.24 (Gin, GORM, zerolog, Cobra, Viper) |
| Python SDK | FastAPI/Uvicorn, bot builder pattern |
| Go SDK | Native Go bot builder |
| TypeScript SDK | TypeScript bot client |
| Web UI | React 18, TypeScript, Vite, Tailwind, Radix UI |
| Database | SQLite/BoltDB (local), PostgreSQL (prod) |
| Image | `ghcr.io/hanzoai/playground:latest` |
| K8s manifests | in team repo `k8s/` |
| Repo | `github.com/hanzoai/playground` |

## Architecture

```
                 app.hanzo.bot
                      |
               +------+------+
               |             |
           Web UI        REST API
           (React)       (Go/Gin)
               |             |
               +------+------+
                      |
           +----------+----------+
           |          |          |
        Bot Nodes  Workflow    Memory
        (registry) Engine     Scopes
           |       (DAG)      (4 levels)
           |          |          |
      +----+----+    |     +----+----+
      |    |    |    |     |    |    |
    Python Go  TS   Job   Global Bot
    SDK   SDK  SDK  Queue Session Run
```

## Core concepts

### Node registry

Bot instances register with the control plane and report:
- Health status (heartbeat)
- Available skills/capabilities
- Current execution state
- Resource utilization

### Workflow DAGs

Compose multi-bot workflows with dependency tracking:

```json
{
  "name": "research-pipeline",
  "steps": [
    {"id": "search", "bot": "web-researcher", "skill": "search"},
    {"id": "analyze", "bot": "analyst", "skill": "summarize", "depends": ["search"]},
    {"id": "report", "bot": "writer", "skill": "generate", "depends": ["analyze"]}
  ]
}
```

### Memory scopes (4 levels)

| Scope | Lifetime | Visibility | Use case |
|-------|----------|------------|----------|
| **Global** | Permanent | All bots | Shared knowledge base |
| **Bot** | Permanent | Single bot | Bot-specific context |
| **Session** | Session | Single user session | Conversation context |
| **Run** | Single execution | Single workflow run | Execution scratch space |

### Cryptographic identity (DID/VC)

Every bot execution can optionally produce W3C DID/VC audit trails:
- Bot identity via DID (Decentralized Identifier)
- Execution proofs via VC (Verifiable Credential)
- Enables trustworthy audit logs for regulated environments

## Python SDK quickstart

```python
from hanzo_playground import Bot, Skill

@Skill(name="greet", description="Greet a user")
async def greet(name: str) -> str:
    return f"Hello, {name}!"

bot = Bot(
    name="greeter",
    control_plane="https://app.hanzo.bot",
    skills=[greet],
)

bot.run()  # Registers with control plane and starts serving
```

## Go SDK quickstart

```go
package main

import (
    playground "github.com/hanzoai/playground/sdk/go"
)

func main() {
    bot := playground.NewBot("greeter", playground.Config{
        ControlPlane: "https://app.hanzo.bot",
    })

    bot.RegisterSkill("greet", func(ctx playground.Context) (string, error) {
        name := ctx.Param("name")
        return fmt.Sprintf("Hello, %s!", name), nil
    })

    bot.Run()
}
```

## Cloud provisioning

The control plane can provision agents as:
- **K8s pods** (Linux): Standard container deployment
- **VMs via Visor** (Mac/Windows): For desktop automation tasks

```bash
# Provision a new bot instance
curl -X POST https://app.hanzo.bot/api/v1/bots \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "name": "web-researcher",
    "image": "ghcr.io/hanzoai/bot:latest",
    "provisioner": "kubernetes",
    "replicas": 1
  }'
```

## K8s deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: playground
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: playground
          image: ghcr.io/hanzoai/playground:latest
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              value: postgresql://playground:pass@postgres.hanzo.svc:5432/playground
            - name: PLAYGROUND_MODE
              value: cloud
```

## Network policy

The playground pod must be accessible from:
- Hanzo Ingress (external access via `app.hanzo.bot`)
- Bot pods within the cluster (registration and heartbeat)
- Team front pod (for embedded playground views)

Ensure network policies allow ingress to playground pods.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Bot not registering | Control plane unreachable | Check `control_plane` URL and network policy |
| Workflow stuck | Dependency cycle | Validate DAG has no cycles |
| Memory not persisting | SQLite in ephemeral pod | Use PostgreSQL for production |
| Web UI 404 | Ingress not configured | Add IngressRoute for `app.hanzo.bot` |

## Related Skills

- `hanzo/hanzo-bot.md` -- Bot gateway (registers with Playground)
- `hanzo/hanzo-agent.md` -- Multi-agent SDK
- `hanzo/hanzo-operative.md` -- Computer use agent
- `hanzo/hanzo-k8s.md` -- K8s infrastructure

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: playground, bot-orchestration, control-plane, workflows, did, memory
**Prerequisites**: Go or Python, PostgreSQL for production
