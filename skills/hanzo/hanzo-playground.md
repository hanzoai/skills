# Hanzo Playground - Control Plane for AI Bots

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-bot.md`, `hanzo/hanzo-operative.md`

## Overview

Hanzo Playground is a **Kubernetes-style control plane for AI bots**. It provides production infrastructure for deploying, orchestrating, and observing multi-bot systems with cryptographic identity (DID/VC), workflow execution, memory scoping, and an embedded web UI. Three-tier monorepo: Go control plane, Python/Go/TypeScript SDKs, React admin interface.

### Why Hanzo Playground?

- **Bot orchestration**: Register, invoke, and monitor bots through a central control plane
- **Workflow DAGs**: Compose multi-bot workflows with dependency tracking and observability
- **Memory scopes**: Global, bot, session, and run-scoped memory synced by control plane
- **Cryptographic identity**: W3C DID/VC for every bot execution (opt-in audit trails)
- **Dual storage**: SQLite/BoltDB for local dev, PostgreSQL for production
- **Cloud provisioning**: Provision agents as K8s pods (Linux) or VMs via Visor (Mac/Windows)
- **Embedded Web UI**: React/TypeScript admin dashboard served from the Go binary
- **Multi-SDK**: Python (FastAPI), Go, and TypeScript SDKs for building bots

### Tech Stack

- **Control Plane**: Go 1.24, Gin, GORM, zerolog, Cobra, Viper, Prometheus
- **Python SDK**: FastAPI/Uvicorn, bot builder pattern
- **Go SDK**: Native Go bot builder with skills registration
- **TypeScript SDK**: TypeScript bot client
- **Web UI**: React 18, TypeScript, Vite, Tailwind CSS, Radix UI
- **Database**: SQLite/BoltDB (local), PostgreSQL (cloud), Goose migrations
- **Version**: 0.1.41-rc.197
- **Image**: `ghcr.io/hanzoai/playground:latest`
- **Production**: `playground.hanzo.bot` (alias: `app.hanzo.bot`)

## When to use

- Deploying and orchestrating multi-bot AI systems
- Building bots with Python or Go that need central coordination
- Running multi-step workflows across multiple AI agents
- Auditing bot executions with cryptographic proofs (DID/VC)
- Provisioning cloud agents (K8s pods or VMs)
- Monitoring bot health, workflow state, and memory via web UI

## Hard requirements

1. **Go 1.23+** for control plane
2. **Node.js 20+** for web UI development
3. **Python 3.8+** for Python SDK
4. **PostgreSQL 15+** for cloud/production mode (SQLite for local dev)

## Quick reference

| Item | Value |
|------|-------|
| Control plane port | 8080 |
| Web UI | `http://localhost:8080/ui/` |
| Version | 0.1.41-rc.197 |
| Image | `ghcr.io/hanzoai/playground:latest` |
| Production URL | `playground.hanzo.bot` / `app.hanzo.bot` |
| Go module | `github.com/hanzoai/playground/control-plane` |
| Python package | `playground` (PyPI) |
| Repo | `github.com/hanzoai/playground` |
| Branch | `main` |
| License | Apache 2.0 |

## Repository Structure

```
control-plane/
  cmd/
    playground/           # Unified CLI (server + dev/init commands)
    playground-server/    # Standalone server binary
  internal/
    cli/                  # CLI command definitions
    server/               # HTTP server (Gin), middleware, routing
    handlers/             # REST/gRPC request handlers
    services/             # Business logic (workflow, bot registry, DID/VC)
    storage/              # Data layer (SQLite/BoltDB, PostgreSQL)
    events/               # Event bus, SSE streaming
    core/                 # Domain models and interfaces
    cloud/                # Cloud provisioning (K8s + Visor)
    mcp/                  # Model Context Protocol integration
    encryption/           # Cryptographic primitives (DID/VC)
    config/               # Configuration (Viper)
    logger/               # Structured logging (zerolog)
  migrations/             # Goose SQL migrations
  web/client/             # React/TypeScript admin UI (Vite)
  go.mod                  # Go module definition
sdk/
  python/                 # Python SDK (FastAPI bot builder)
  go/                     # Go SDK (bot builder + skills)
  typescript/             # TypeScript SDK
deployments/
  docker/                 # Docker Compose + Dockerfile
  helm/                   # Helm chart
  kubernetes/             # Raw K8s manifests
  railway/                # Railway deployment config
tests/                    # Integration tests
examples/                 # Example bots
Makefile                  # Build, test, lint targets
.goreleaser.yml           # Multi-platform release builds
VERSION                   # Current version
```

## Quick Start

### Local Mode (No External Dependencies)

```bash
cd control-plane
go run ./cmd/playground dev
# Runs at http://localhost:8080, UI at http://localhost:8080/ui/
# Uses SQLite + BoltDB (no PostgreSQL needed)
```

### Cloud Mode (PostgreSQL)

```bash
cd control-plane
export PLAYGROUND_DATABASE_URL="postgres://playground:playground@localhost:5432/playground?sslmode=disable"

# Run migrations
goose -dir ./migrations postgres "$PLAYGROUND_DATABASE_URL" up

# Start server
PLAYGROUND_STORAGE_MODE=postgresql go run ./cmd/playground-server
```

### Docker Compose

```bash
cd deployments/docker
docker compose up
```

## Building Bots

### Python Bot

```bash
# Scaffold a new bot
playground init my-bot
cd my-bot

# Edit bot code, then run
playground run
```

```python
# Python SDK: register "reasoners" (decorated functions become REST endpoints)
from playground import Bot

bot = Bot("my-bot", server="http://localhost:8080")

@bot.reasoner("greet")
async def greet(input):
    return {"message": f"Hello, {input.get('name', 'world')}!"}

bot.run()
```

### Go Bot

```go
import playgroundbot "github.com/hanzoai/playground/sdk/go/bot"

bot, _ := playgroundbot.New(playgroundbot.Config{
    NodeID:        "my-bot",
    PlaygroundURL: "http://localhost:8080",
})
bot.RegisterSkill("greet", func(ctx context.Context, input map[string]any) (any, error) {
    return map[string]any{"message": "hello"}, nil
})
bot.Run(context.Background())
```

## Key Concepts

### Bot-to-Bot Communication

Bots call each other through the control plane -- never direct HTTP:

```python
result = await bot.call("other-bot.function", input={"key": "value"})
# Control plane routes request, tracks workflow DAG, injects metrics
```

### Memory Scopes

| Scope | Description |
|-------|-------------|
| **Global** | Shared across all bots and sessions |
| **Bot** | Scoped to one bot, all sessions |
| **Session** | Scoped to one session (multi-turn conversation) |
| **Run** | Scoped to a single execution/workflow run |

Access via SDK: `bot.memory.get/set(scope, key, value)`

### DID/VC (Cryptographic Identity)

Opt-in per bot. Control plane generates W3C Verifiable Credentials for each execution:

```bash
# Export audit trail
GET /api/v1/did/workflow/{workflow_id}/vc-chain

# Verify offline
playground verify audit.json
```

### Cloud Agent Provisioning

| API | Method | Description |
|-----|--------|-------------|
| `/api/v1/cloud/nodes/provision` | POST | Provision new agent |
| `/api/v1/cloud/nodes/:node_id` | DELETE | Deprovision agent |
| `/api/v1/cloud/nodes` | GET | List cloud agents |
| `/api/v1/cloud/nodes/:node_id/logs` | GET | Get agent logs |
| `/api/v1/cloud/teams/provision` | POST | Batch provision |

Linux agents run as K8s pods (agent + operative sidecar). Mac/Windows agents use Visor for VM management.

### Storage Modes

| Mode | Backend | Use Case |
|------|---------|----------|
| `local` | SQLite + BoltDB | Development, testing |
| `postgresql` | PostgreSQL | Production, cloud |
| `cloud` | PostgreSQL | Distributed deployments |

## Build and Test

```bash
make install             # Install all dependencies
make build               # Build all components
make test                # Run all tests

# Component-specific
cd control-plane && go test ./...
cd sdk/python && pytest
cd sdk/go && go test ./...

# Lint and format
make lint
make fmt
make tidy
```

## Environment Variables

```bash
PLAYGROUND_PORT=8080                    # HTTP server port
PLAYGROUND_MODE=local                   # local or cloud
PLAYGROUND_STORAGE_MODE=local           # local, postgresql, or cloud
PLAYGROUND_DATABASE_URL=postgres://...  # PostgreSQL connection
PLAYGROUND_UI_ENABLED=true              # Enable web UI
PLAYGROUND_UI_MODE=embedded             # embedded or development
PLAYGROUND_CONFIG_FILE=config.yaml      # Config file path
GIN_MODE=release                        # debug or release
LOG_LEVEL=info                          # debug, info, warn, error
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Control plane won't start | Bad DATABASE_URL | Check PostgreSQL connection string |
| Migrations fail | PostgreSQL not running | Start PostgreSQL, verify connection |
| Bot can't connect | Wrong server URL | Set `PLAYGROUND_URL=http://localhost:8080` |
| UI not loading | Dev server not running | Run both Vite dev server and control plane |
| DB pool exhausted | Too many connections | Increase `PLAYGROUND_STORAGE_POSTGRES_MAX_CONNECTIONS` |

## Related Skills

- `hanzo/hanzo-agent.md` - Multi-agent SDK (complementary)
- `hanzo/hanzo-bot.md` - Bot gateway (related infrastructure)
- `hanzo/hanzo-operative.md` - Computer use agent (sidecar in cloud agents)
- `hanzo/hanzo-tunnel.md` - Tunnel for remote agent control

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: bots, orchestration, control-plane, workflows, did, go, python, react
**Prerequisites**: Go 1.23+, Python 3.8+ (for SDK), Node.js 20+ (for UI)
