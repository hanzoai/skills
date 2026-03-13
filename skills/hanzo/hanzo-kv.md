# Hanzo KV - High-Performance Key-Value Store

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-sql.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-orm.md`

## Overview

Hanzo KV is a **Redis-compatible in-memory key-value store** used as the caching, streaming, and message broker layer across the Hanzo ecosystem. C codebase built with CMake, ships as `ghcr.io/hanzoai/kv`. Includes a native ZAP binary protocol module for high-throughput access. Default port 6379 (RESP), ZAP port 9653. License: BSD-3-Clause.

### Why Hanzo KV?

- **Redis drop-in replacement**: Any Redis client works out of the box
- **ZAP binary protocol**: Native module on port 9653 (17x faster than JSON-RPC)
- **Persistence**: RDB snapshots and AOF for durability
- **Replication**: Primary-replica with Sentinel automatic failover
- **Cluster mode**: Horizontal scaling with automatic sharding
- **Module system**: Extensible via C module API (`kvmodule.h`)

### Tech Stack

- **Language**: C
- **Build**: Make (top-level delegates to `src/Makefile`), CMake also available
- **Image**: `ghcr.io/hanzoai/kv` (based on `kv/kv:9-alpine`)
- **Modules**: ZAP binary protocol (`modules/zap/`)

### OSS Base

Repo: `hanzoai/kv` (Valkey/Redis fork). Default branch: `main`.

## When to use

- Caching layer for any Hanzo service
- Session storage and rate limiting
- Pub/Sub messaging between services
- Streams for event sourcing
- Queue backend (via lists or streams)
- High-throughput data access via ZAP protocol

## Hard requirements

1. **Port 6379** available for RESP protocol
2. **Port 9653** if using ZAP binary protocol module
3. **Docker** or C build toolchain (gcc, make) for building from source

## Quick reference

| Item | Value |
|------|-------|
| Default Port | 6379 (RESP) |
| ZAP Port | 9653 |
| Image | `ghcr.io/hanzoai/kv` |
| Config | `kv.conf` |
| Sentinel Config | `sentinel.conf` |
| License | BSD-3-Clause |
| Repo | `github.com/hanzoai/kv` |
| Docs | `github.com/hanzoai/kv-doc` |

## One-file quickstart

### Docker

```bash
docker run -d --name hanzo-kv -p 6379:6379 ghcr.io/hanzoai/kv
```

### Connect

```bash
docker exec -it hanzo-kv kv

127.0.0.1:6379> SET hello world
OK
127.0.0.1:6379> GET hello
"world"
```

### Build from source

```bash
make
make test
make install
```

## Core Concepts

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Application │────>│   Hanzo KV   │────>│  Persistence  │
│  (any Redis  │     │  (port 6379) │     │  (RDB / AOF)  │
│   client)    │     └──────┬───────┘     └──────────────┘
└──────────────┘            │
                     ┌──────┴───────┐
                     │  ZAP Module  │
                     │ (port 9653)  │
                     └──────────────┘
```

### CLI Tools

| Command | Description |
|---------|-------------|
| `kv` | Interactive CLI (symlink to kv-cli) |
| `kv-server` | Start KV server |
| `kv-cli` | Command-line client |
| `kv-sentinel` | High-availability sentinel |
| `kv-benchmark` | Performance benchmarking |
| `kv-check-aof` | AOF file integrity check |
| `kv-check-rdb` | RDB file integrity check |

### ZAP Binary Protocol

The ZAP module (`modules/zap/`) implements the [luxfi/zap](https://github.com/luxfi/zap) binary protocol natively:

```bash
# Load module at startup
kv-server --loadmodule /path/to/zap.so PORT 9653
```

ZAP endpoints:

| Path | Body | Description |
|------|------|-------------|
| `/get` | `{"key":"mykey"}` | GET a key |
| `/set` | `{"key":"mykey","value":"myval"}` | SET a key |
| `/del` | `{"key":"mykey"}` | DEL a key |
| `/cmd` | `{"cmd":"PING","args":[]}` | Execute any command |

### Module API

Custom modules use the KV Module API:

```c
#include "kvmodule.h"

int KVModule_OnLoad(KVModuleCtx *ctx, KVModuleString **argv, int argc) {
    if (KVModule_Init(ctx, "mymod", 1, KVMODULE_APIVER_1) == KVMODULE_ERR)
        return KVMODULE_ERR;
    // register commands...
    return KVMODULE_OK;
}
```

### Configuration

```bash
# Pass config file
kv-server /etc/kv/kv.conf

# Or command-line options
kv-server --port 6379 --maxmemory 256mb --appendonly yes
```

Docker default CMD: `--bind 0.0.0.0 --dir /data --maxmemory-policy allkeys-lru --protected-mode no`

### Client SDKs

| Language | Package | Install |
|----------|---------|---------|
| Python | hanzo-kv | `pip install hanzo-kv` |
| Go | hanzo/kv-go | `go get github.com/hanzoai/kv-go` |
| Node.js | @hanzo/kv | `npm install @hanzo/kv` |

Any Redis-compatible client library also works.

### Directory Structure

```
kv/
  CMakeLists.txt         # CMake build
  Makefile               # Top-level make (delegates to src/)
  Dockerfile             # Container build (kv/kv:9-alpine base)
  kv.conf                # Default server configuration
  sentinel.conf          # Sentinel configuration
  src/                   # C source code
  modules/
    zap/                 # ZAP binary protocol module
      zap_module.c       # Module implementation
      zap_protocol.h     # Protocol header
      Makefile           # Module build
  deps/                  # Vendored dependencies
  tests/                 # Integration tests
  utils/                 # Utility scripts
  cmake/                 # CMake modules
```

### Development Guidelines

- C style: `snake_case` variables, `camelCase` functions, `UPPER_CASE` macros
- Unit tests in `src/unit/`, integration tests in `tests/`
- Line length: keep below 90 chars when reasonable
- Use `static` for file-private functions
- License header: BSD-3-Clause (SPDX)

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | KV not running or wrong port | Check `kv-server` is running, verify port |
| OOM | maxmemory reached | Set `--maxmemory` and `--maxmemory-policy` |
| Persistence issues | AOF corruption | Run `kv-check-aof --fix` |
| ZAP module not loading | Missing .so file | Build with `cd modules/zap && make` |

## Related Skills

- `hanzo/hanzo-sql.md` - PostgreSQL database
- `hanzo/hanzo-orm.md` - ORM with KV cache backend
- `hanzo/hanzo-platform.md` - PaaS deployment
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: kv, redis, valkey, cache, pub/sub, streams, zap
**Prerequisites**: Docker or C build toolchain
