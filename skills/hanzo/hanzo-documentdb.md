# Hanzo DocDB - MongoDB-Compatible Document Database

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kv-go.md`, `hanzo/go-sdk.md`

## Overview

DocDB is a MongoDB-compatible document database server. It is a proxy that converts MongoDB 5.0+ wire protocol queries to SQL, using PostgreSQL with the DocumentDB extension as the storage engine. This is a full server, not a client library. Any MongoDB driver or tool can connect to it directly.

Fork of **FerretDB**. Repo: `hanzoai/documentdb`, branch: `main`.

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/docdb` |
| Binary | `docdb` |
| Go | 1.26+ |
| Repo | `github.com/hanzoai/documentdb` |
| Branch | `main` |
| License | Apache-2.0 |
| Wire protocol | MongoDB 5.0+ |
| Backend | PostgreSQL + DocumentDB extension |
| Default port | 27017 |
| Docker image | `ghcr.io/hanzoai/docdb-eval:2` |

## Architecture

```
Application (any MongoDB driver)
    |
    | MongoDB wire protocol (BSON)
    v
  DocDB proxy (this project)
    |
    | PostgreSQL protocol (SQL)
    v
  PostgreSQL + DocumentDB extension
```

## Quick start (Docker)

```bash
docker run -d --rm --name docdb -p 27017:27017 \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypass \
  ghcr.io/hanzoai/docdb-eval:2
```

Connect with any MongoDB client:
```bash
mongosh "mongodb://myuser:mypass@127.0.0.1:27017/"
```

Or connect to the embedded PostgreSQL:
```bash
docker exec -it docdb psql -U myuser postgres
```

## Building from source

```bash
go build -o docdb ./cmd/docdb
./docdb --listen-addr=:27017 --postgresql-url=postgres://user:pass@localhost:5432/dbname
```

## Go library embedding

DocDB can be embedded in Go applications:

```go
import "github.com/hanzoai/docdb/v2/docdb"
```

See [pkg.go.dev/github.com/hanzoai/docdb/v2/docdb](https://pkg.go.dev/github.com/hanzoai/docdb/v2/docdb) for the embeddable API.

## Key features

- Drop-in MongoDB replacement for most 5.0+ workloads
- Uses PostgreSQL as the storage backend (ACID, proven reliability)
- Any MongoDB driver or tool works (mongosh, Compass, pymongo, etc.)
- SCRAM authentication
- Prometheus metrics endpoint
- OpenTelemetry tracing
- MCP (Model Context Protocol) integration
- Health/readiness endpoints

## Project structure

| Path | Purpose |
|------|---------|
| `cmd/docdb/` | Main binary entry point |
| `cmd/envtool/` | Development environment tool |
| `docdb/` | Embeddable library package |
| `internal/` | Core implementation (handlers, backends, wire protocol) |
| `integration/` | Integration test suite |
| `build/` | Build configurations, certs, Docker deps |

## Build and test

```bash
# Build
go build ./cmd/docdb

# Unit tests
go test ./...

# Integration tests (requires PostgreSQL)
go test ./integration/...

# Task runner (alternative to make)
task test
task build
```

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: documentdb, mongodb, postgresql, database, server
**Prerequisites**: Go 1.26+, PostgreSQL with DocumentDB extension
