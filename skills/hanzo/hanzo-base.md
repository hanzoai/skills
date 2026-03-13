# Hanzo Base - Open Source Backend for Any App

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-orm.md`, `hanzo/hanzo-database.md`, `hanzo/js-sdk.md`

## Overview

Hanzo Base is an **open source Go backend** that provides a complete application backend in a single binary. It includes embedded SQLite and PostgreSQL databases with realtime subscriptions, built-in file and user management, an Admin dashboard UI, REST and GraphQL APIs, CRDT-based collaboration, and a JavaScript plugin VM. Go module: `github.com/hanzoai/base`.

### Why Hanzo Base?

- **Single binary**: Entire backend compiles to one portable executable (CGO_ENABLED=0)
- **Embedded databases**: In-memory, SQL (SQLite via modernc, PostgreSQL via pgx), and vector DB with realtime subscriptions
- **Built-in auth**: Users, OAuth2, OTP, MFA, password reset, email verification, superuser management
- **Admin dashboard**: Svelte/Vite SPA embedded in the Go binary
- **JavaScript hooks**: Extend with JS via embedded Goja VM (no separate Node.js process)
- **CRDT support**: Conflict-free replicated data types for collaborative editing
- **Plugin system**: cloudsql, functions, jsvm, migratecmd, platform, scheduler, zap
- **SDK clients**: `@hanzoai/base` (JS/TS), `@hanzoai/base-react` (React hooks)

### Tech Stack

- **Backend**: Go 1.26, cobra CLI, embedded SQLite (modernc.org/sqlite), PostgreSQL (pgx/v5)
- **Admin UI**: Svelte 4, Vite 5, CodeMirror, Chart.js
- **JS SDK**: `@hanzoai/base` v0.1.0 (TypeScript, tsup)
- **React SDK**: `@hanzoai/base-react` v0.1.0 (React hooks)
- **Releases**: GoReleaser (linux/darwin/windows, amd64/arm64)
- **Container**: `ghcr.io/hanzoai/base`, Alpine-based, port 8080

### OSS Base

Repo: `hanzoai/base`. License: MIT.

## When to use

- Building app backends that need auth, file storage, and a database out of the box
- Prototyping APIs quickly with automatic CRUD, realtime, and admin UI
- Extending a Go application with a full backend toolkit (use as a library)
- Self-hosted BaaS (Backend as a Service) alternative
- Apps needing CRDT-based realtime collaboration
- Multi-tenant platforms with IAM and KMS integration (via platform plugin)

## Hard requirements

1. **Go 1.24+** for building from source
2. **Node 18+** for Admin UI development only
3. No external database required (SQLite embedded); PostgreSQL optional via plugins

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/hanzoai/base` |
| Default port | `8080` (Docker), `8090` (dev) |
| Admin UI | `/_/` path on running instance |
| Data dir | `./hz_data` (default) |
| Health check | `GET /api/health` |
| JS SDK | `@hanzoai/base` |
| React SDK | `@hanzoai/base-react` |
| Docker image | `ghcr.io/hanzoai/base` |
| License | MIT |
| Repo | `github.com/hanzoai/base` |

## One-file quickstart

### Use as a Go library

```go
package main

import (
    "log"

    "github.com/hanzoai/base"
    "github.com/hanzoai/base/core"
)

func main() {
    app := base.New()

    app.OnServe().BindFunc(func(se *core.ServeEvent) error {
        // registers new "GET /hello" route
        se.Router.GET("/hello", func(re *core.RequestEvent) error {
            return re.String(200, "Hello world!")
        })

        return se.Next()
    })

    if err := app.Start(); err != nil {
        log.Fatal(err)
    }
}
```

Then run:

```bash
go mod init myapp && go mod tidy
go run main.go serve
```

### Run as standalone binary

```bash
# Download from releases or build from source
CGO_ENABLED=0 go build -o base ./examples/base
./base serve --http=0.0.0.0:8080 --dir=./hz_data
```

### Docker

```bash
docker run -d \
  --name base \
  -p 8080:8080 \
  -v ./hz_data:/pb_data \
  ghcr.io/hanzoai/base \
  serve --http=0.0.0.0:8080 --dir=/pb_data
```

## Core Concepts

### Architecture

```
┌──────────────────────────────────────────────┐
│              Hanzo Base Binary                │
├──────────────┬───────────────┬───────────────┤
│   REST API   │  Realtime WS  │  Admin UI     │
│   /api/*     │  /api/realtime│  /_/          │
├──────────────┴───────────────┴───────────────┤
│                  core.App                     │
│  ┌─────────┐ ┌──────────┐ ┌───────────────┐ │
│  │ Records │ │ Auth     │ │ File Storage  │ │
│  │ (CRUD)  │ │ (OAuth2) │ │ (S3/local)    │ │
│  └─────────┘ └──────────┘ └───────────────┘ │
├──────────────────────────────────────────────┤
│  SQLite (default) │ PostgreSQL (cloudsql)    │
└──────────────────────────────────────────────┘
```

### Directory Structure

```
base/
  base.go              # Main Base struct, implements core.App
  core/                # Core app logic, records, collections, events, router
  apis/                # HTTP API handlers (records, auth, collections, realtime, batch, etc.)
  cmd/                 # CLI commands (serve, superuser)
  crdt/                # CRDT types (document, text, sync)
  forms/               # Form validation (record upsert, email, S3 test)
  migrations/          # Database migration scripts
  mails/               # Email templates
  plugins/
    jsvm/              # JavaScript VM (Goja) for hooks and migrations
    migratecmd/        # Migration CLI command
    cloudsql/          # Serverless PostgreSQL (per-tenant)
    functions/         # OpenFaaS serverless functions
    platform/          # Multi-tenant IAM + KMS integration
    scheduler/         # Cron/scheduled tasks
    ghupdate/          # GitHub self-update
    zap/               # ZAP binary protocol transport (port 9652)
  sdk/
    base-js/           # @hanzoai/base TypeScript SDK
    base-react/        # @hanzoai/base-react React hooks
  tools/               # Internal utility packages
    archive/           # Archive utilities
    auth/              # Auth helpers
    cron/              # Cron scheduler
    dbutils/           # Database utilities
    filesystem/        # File system abstraction
    hook/              # Event hook system
    inflector/         # String inflection
    list/              # List utilities
    logger/            # Structured logging
    mailer/            # Email sending
    osutils/           # OS utilities
    picker/            # Data picker
    router/            # HTTP router
    routine/           # Goroutine helpers
    search/            # Search utilities
    security/          # Security helpers
    store/             # In-memory store
    subscriptions/     # Realtime subscriptions
    template/          # Template engine
    tokenizer/         # Token utilities
    types/             # Custom types
  ui/                  # Admin dashboard (Svelte 4 + Vite 5)
  examples/base/       # Standalone example with all plugins
  tests/               # Integration tests
  docs/                # Documentation
```

### Plugin System

Plugins extend Base via hooks on the `core.App` interface:

| Plugin | Purpose |
|--------|---------|
| `jsvm` | JavaScript hooks and migrations via Goja VM |
| `migratecmd` | Database migration CLI commands |
| `cloudsql` | Serverless PostgreSQL per-tenant provisioning |
| `functions` | OpenFaaS serverless function integration |
| `platform` | Multi-tenant IAM (hanzo.id) + KMS (kms.hanzo.ai) |
| `scheduler` | Cron-based scheduled tasks |
| `ghupdate` | GitHub self-update for releases |
| `zap` | Binary protocol transport on port 9652 |

### CLI Commands

```bash
./base serve              # Start the HTTP server
./base serve --dev        # Dev mode (verbose logging, SQL output)
./base serve --http=0.0.0.0:8080
./base serve --dir=./hz_data
./base superuser create   # Create a superuser
./base superuser upsert   # Create or update a superuser
```

### Global Flags

```
--dir              Data directory (default: ./hz_data)
--dev              Enable dev mode
--encryptionEnv    Env var name for 32-char encryption key
--queryTimeout     Default SELECT query timeout in seconds
```

### Auth Methods

Base supports multiple authentication methods out of the box:

- **Password**: Email/password login
- **OAuth2**: External providers (Google, GitHub, Apple, etc.)
- **OTP**: One-time password via email
- **MFA**: Multi-factor authentication
- **Email verification**: Confirm email addresses
- **Password reset**: Token-based password recovery
- **Impersonation**: Superuser can act as any user
- **Token refresh**: JWT refresh flow

### Realtime

WebSocket-based realtime subscriptions at `/api/realtime`:
- Subscribe to record changes per collection
- Presence tracking for connected clients
- Event types: create, update, delete

### CRDT

The `crdt/` package provides conflict-free replicated data types:
- `Document`: JSON document with merge semantics
- `Text`: Collaborative text editing
- `Sync`: Synchronization protocol between peers

## Development

### Build and test

```bash
# Run tests
go test ./...
make test

# Lint
golangci-lint run -c ./golangci.yml ./...
make lint

# Build standalone binary
cd examples/base
CGO_ENABLED=0 go build

# Generate JS types for JSVM plugin
make jstypes

# Test coverage report
make test-report
```

### Admin UI development

```bash
cd ui
npm install
npm run dev     # Dev server at localhost:3000
npm run build   # Production build to ui/dist/
```

The Admin UI expects the backend at `http://localhost:8090` by default. Override with `HZ_BACKEND_URL` in `ui/.env.development.local`.

### Docker build

```dockerfile
# Multi-stage build (from repo Dockerfile)
FROM golang:1.26-alpine AS builder
WORKDIR /build
COPY . .
CGO_ENABLED=0 go build \
  -ldflags="-s -w -X github.com/hanzoai/base.Version=$(git describe --tags)" \
  -o /build/base ./examples/base/main.go

FROM alpine:3.21
COPY --from=builder /build/base /app/base
EXPOSE 8080
ENTRYPOINT ["/app/base"]
CMD ["serve", "--http=0.0.0.0:8080", "--dir=/pb_data"]
```

## SDK Clients

### JavaScript SDK (`@hanzoai/base`)

```typescript
import { BaseClient } from '@hanzoai/base';

const client = new BaseClient('http://localhost:8080');

// Reactive queries, query deduplication, optimistic updates
// Exports: core, react hooks, crdt
```

Exports:
- `@hanzoai/base` -- core client
- `@hanzoai/base/react` -- React hooks
- `@hanzoai/base/crdt` -- CRDT collaboration

### React SDK (`@hanzoai/base-react`)

```typescript
import { useBaseQuery } from '@hanzoai/base-react';

// Reactive queries, optimistic mutations, realtime, CRDT collaboration
```

### Official external SDKs

- **JavaScript**: `@hanzoai/js-sdk` (github.com/hanzoai/js-sdk)
- **Dart**: `@hanzoai/dart-sdk` (github.com/hanzoai/dart-sdk)

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 8090 already in use | Another Base instance running | Kill process or use `--http=:8091` |
| SQLite build errors | CGO required by default | Use `CGO_ENABLED=0` for pure Go SQLite |
| Admin UI not loading | UI dist not embedded | Run `cd ui && npm run build` first |
| JS hooks not executing | Wrong hooks directory | Use `--hooksDir=./hz_hooks` flag |
| Modernc version mismatch | Dependency conflict | Check `modernc_versions_check.go` logs |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS platform (uses Base platform plugin)
- `hanzo/hanzo-orm.md` - ORM package (complementary Go data layer)
- `hanzo/hanzo-database.md` - Database infrastructure
- `hanzo/js-sdk.md` - JavaScript SDK ecosystem
- `hanzo/hanzo-id.md` - IAM (used by platform plugin)
- `hanzo/hanzo-kms.md` - Secret management (used by platform plugin)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: backend, baas, go, realtime, crdt, auth, admin
**Prerequisites**: Go 1.24+, basic REST API knowledge
