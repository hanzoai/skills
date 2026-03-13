# Hanzo Runtime - Secure AI Code Execution Sandbox

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-operative.md`, `hanzo/hanzo-agent.md`

## Overview

Hanzo Runtime is a **secure, isolated sandbox environment for executing AI-generated code**. Sub-90ms sandbox creation, OCI/Docker-compatible, with programmatic File, Git, LSP, and Execute APIs. Multi-language monorepo (Go + TypeScript + Python) with Nx workspace orchestration. SDKs published as `hanzo-runtime` (PyPI) and `@hanzo/runtime` (npm).

### Why Hanzo Runtime?

- **Sub-90ms sandbox creation**: From code to execution in under 90ms
- **Isolated execution**: AI-generated code runs in sandboxed containers with zero risk
- **Programmatic control**: File, Git, LSP, and Execute APIs
- **OCI/Docker compatible**: Use any OCI/Docker image as sandbox base
- **Unlimited persistence**: Sandboxes can live indefinitely
- **Computer use**: Built-in computer-use library for browser/desktop automation
- **Multi-language SDKs**: Python, TypeScript, Go clients

### Tech Stack

- **API**: NestJS (TypeScript) on port 8000
- **Backend Services**: Go (cli, daemon, proxy, runner)
- **Dashboard**: React + Vite + Radix UI + Tailwind
- **Docs**: Astro + Starlight
- **Build**: Nx 20.6 monorepo + pnpm workspace + Go workspace
- **Database**: PostgreSQL (TypeORM)
- **Cache**: Redis (ioredis)
- **Auth**: Hanzo IAM via OIDC (express-openid-connect, passport-jwt)
- **Telemetry**: OpenTelemetry (OTLP traces, instrumentation for HTTP/ioredis/NestJS/pg)
- **Container Management**: Dockerode

### OSS Base

Repo: `hanzoai/runtime`. License: AGPL-3.0. Fork of Daytona.

## When to use

- Executing untrusted or AI-generated code safely
- Building AI coding assistants that need code execution
- Running code in isolated sandboxes for testing/evaluation
- Computer-use automation (browser, desktop interaction)
- Agent workflows requiring programmatic code execution

## Hard requirements

1. **Docker** for sandbox container management
2. **Node.js 20+** and **pnpm** for TypeScript services
3. **Go 1.23+** for backend services (cli, daemon, proxy, runner)
4. **PostgreSQL** for persistent state
5. **Redis** for caching and messaging
6. **Hanzo IAM** for authentication (OIDC)

## Quick reference

| Item | Value |
|------|-------|
| API | port 8000 |
| Dashboard | React + Vite |
| Python SDK | `pip install hanzo-runtime` (v0.7.0) |
| TypeScript SDK | `npm install @hanzo/runtime` |
| Go Module | `github.com/hanzoai/runtime` |
| License | AGPL-3.0 |
| Repo | `github.com/hanzoai/runtime` |
| IAM Client | `runtime-dashboard` |

## One-file quickstart

### Python SDK

```python
from hanzo_runtime import HanzoRuntime, HanzoRuntimeConfig, CreateSandboxParams

# Initialize the client
runtime = HanzoRuntime(HanzoRuntimeConfig(api_key="YOUR_API_KEY"))

# Create a sandbox
sandbox = runtime.create(CreateSandboxParams(language="python"))

# Run code securely inside the sandbox
response = sandbox.process.code_run('print("Sum of 3 and 4 is " + str(3 + 4))')
if response.exit_code != 0:
    print(f"Error: {response.exit_code} {response.result}")
else:
    print(response.result)

# Clean up
runtime.remove(sandbox)
```

### TypeScript SDK

```typescript
import { HanzoRuntime } from '@hanzo/runtime'

const runtime = new HanzoRuntime({ apiKey: 'YOUR_API_KEY' })

const sandbox = await runtime.create({ language: 'python' })
const response = await sandbox.process.codeRun('print("Hello from sandbox")')
console.log(response.result)

await runtime.remove(sandbox)
```

## Core Concepts

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Hanzo IAM  │----+│  Dashboard  │----+│ Runtime API  │
│  (Casdoor)  │     │   (React)   │     │  (NestJS)    │
└─────────────┘     └─────────────┘     └──────────────┘
       │                                         │
       │                                         v
       │                                 ┌──────────────┐
       └--------------------------------+│   Sandbox    │
                                         │  Execution   │
                                         └──────────────┘
```

### Monorepo Structure

```
hanzoai/runtime
+-- apps/
|   +-- api/            # NestJS API server (TypeScript, port 8000)
|   +-- cli/            # CLI tool (Go)
|   +-- daemon/         # Background daemon (Go)
|   +-- proxy/          # Network proxy (Go)
|   +-- runner/         # Sandbox runner (Go)
|   +-- dashboard/      # Web dashboard (React + Vite)
|   +-- docs/           # Documentation site (Astro + Starlight)
|   +-- daytona-e2e/    # End-to-end tests
+-- libs/
|   +-- sdk-python/     # Python SDK (hanzo-runtime on PyPI)
|   +-- sdk-typescript/ # TypeScript SDK (@hanzo/runtime on npm)
|   +-- api-client/     # TypeScript API client (@hanzo/api-client)
|   +-- api-client-go/  # Go API client
|   +-- api-client-python/       # Generated Python API client
|   +-- api-client-python-async/ # Async Python API client
|   +-- runner-api-client/       # Runner API client
|   +-- common-go/      # Shared Go utilities
|   +-- computer-use/   # Computer-use library (Go, browser/desktop automation)
+-- functions/          # Serverless functions
+-- examples/           # Usage examples
+-- images/
|   +-- sandbox/        # Sandbox container images
+-- hack/              # Development scripts
```

### Go Workspace (go.work)

```
apps/cli
apps/daemon
apps/proxy
apps/runner
libs/api-client-go
libs/common-go
libs/computer-use
```

### Published Packages

| Package | Registry | Version | Language |
|---------|----------|---------|----------|
| `hanzo-runtime` | PyPI | 0.7.0 | Python |
| `hanzo_runtime_api_client` | PyPI | 0.7.0 | Python |
| `hanzo_runtime_api_client_async` | PyPI | 0.7.0 | Python |
| `@hanzo/runtime` | npm | dev | TypeScript |
| `@hanzo/api-client` | npm | 0.7.0 | TypeScript |
| `@hanzo/runner-api-client` | npm | 0.7.0 | TypeScript |

### IAM Integration

Dashboard uses OIDC SSO via Hanzo IAM:

```bash
# Dashboard environment
VITE_OIDC_DOMAIN=https://iam.hanzo.ai
VITE_OIDC_CLIENT_ID=runtime-dashboard
VITE_API_URL=https://api.runtime.hanzo.ai
```

IAM app configuration:
```json
{
  "name": "runtime-dashboard",
  "client_id": "runtime-dashboard",
  "redirect_uris": [
    "https://runtime.hanzo.ai",
    "https://runtime.hanzo.ai/callback"
  ],
  "grant_types": ["authorization_code", "client_credentials"]
}
```

### Sandbox Features

- **Code execution**: Run code in any language inside isolated containers
- **File API**: Read, write, and manage files in sandbox filesystem
- **Git API**: Clone repos, commit, push from within sandboxes
- **LSP API**: Language Server Protocol for IDE-like features
- **Process API**: Execute commands and scripts
- **Filesystem forking**: Fork sandbox filesystem and memory state (planned)

## Development

```bash
# Install dependencies
pnpm install

# Build all
pnpm build

# Serve all (development)
pnpm serve

# Build Go services
cd apps/runner && go build ./...
cd apps/daemon && go build ./...

# Run tests
pnpm lint
pnpm lint:py

# Generate OpenAPI clients
pnpm generate:api-client

# Publish SDKs
make publish
```

### Docker

```bash
# Development
docker build --target development -t hanzo-runtime:dev .
docker run -p 8000:8000 hanzo-runtime:dev

# Production
docker build -f Dockerfile.runtime --target production -t hanzo-runtime:prod .
docker run -p 8000:8000 hanzo-runtime:prod
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Sandbox creation slow | Docker not running | Start Docker daemon |
| Auth failure | IAM not configured | Set OIDC env vars correctly |
| SDK import error | Wrong package name | Python: `hanzo_runtime`, TS: `@hanzo/runtime` |
| Build fails | Missing Go workspace | Run from repo root with `go.work` |
| TypeORM migration fail | Missing PostgreSQL | Start PostgreSQL, set DATABASE_URL |

## Related Skills

- `hanzo/hanzo-operative.md` - Computer use for Claude (uses Runtime for sandboxing)
- `hanzo/hanzo-agent.md` - Multi-agent SDK (can use Runtime for code execution)
- `hanzo/hanzo-platform.md` - PaaS for deploying applications
- `hanzo/hanzo-id.md` - IAM and authentication

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: runtime, sandbox, code-execution, ai-agent, computer-use
**Prerequisites**: Docker, Node.js 20+, Go 1.23+, PostgreSQL, Redis
