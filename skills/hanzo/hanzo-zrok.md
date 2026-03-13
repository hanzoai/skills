# Hanzo Zrok - Zero-Trust Sharing Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-zt.md`, `hanzo/hanzo-tunnel.md`, `hanzo/hanzo-network.md`

## Overview

Hanzo Zrok is a **zero-trust sharing platform** built on top of Hanzo ZT (OpenZiti). It lets users securely share web services, files, and network resources through firewalls and NAT without any network configuration changes. Single binary. Public and private sharing modes. Self-hostable. Fork of `openziti/zrok`.

### Why Hanzo Zrok?

- **Instant sharing**: `zrok share public localhost:8080` exposes a local service with a public URL
- **Zero config**: Works through firewalls, NAT, corporate networks -- no port forwarding
- **Private sharing**: Share services only with specific zrok users, no public internet exposure
- **File sharing**: Turn any directory into a shareable network drive (`--backend-mode drive`)
- **End-to-end encrypted**: All traffic encrypted via Hanzo ZT overlay, even from zrok servers
- **Self-hostable**: Run your own zrok instance from Raspberry Pi to enterprise scale
- **Multi-protocol**: HTTP/HTTPS, TCP, UDP, and WebDAV file sharing

### Tech Stack

- **Language**: Go (backend + CLI), Node.js (web console UI)
- **Go module**: `github.com/openziti/zrok/v2` (upstream module path, not yet rebranded)
- **Go version**: 1.25.3
- **Binary**: `zrok2` (symlinked to `zrok`)
- **UI**: Node.js 22 (React-based web console + agent UI)
- **Database**: SQLite (embedded, via `mattn/go-sqlite3` with CGO)
- **API**: OpenAPI-generated REST server (`go-openapi` suite, Swagger spec in `specs/`)
- **TUI**: Charm stack (`bubbletea`, `bubbles`, `lipgloss`) for terminal UI
- **Container base**: `openziti/ziti-cli:latest` (includes Ziti CLI tools)
- **Release**: GoReleaser (darwin, linux-amd64/arm64/armhf/armel, windows)

### OpenZiti Dependencies

Zrok is an application layer built on top of the OpenZiti networking fabric:

- `openziti/sdk-golang` v1.2.8 -- Go SDK for Ziti network access
- `openziti/channel/v4` v4.2.37 -- Message channels over Ziti
- `openziti/edge-api` v0.26.48 -- Edge management API client
- `openziti/identity` v1.0.116 -- Identity/certificate management
- `openziti/transport/v2` v2.0.194 -- Transport layer
- `openziti/ziti` v1.6.0 -- Core Ziti runtime
- `openziti/foundation/v2` -- Common utilities

### OSS Base

Fork of `openziti/zrok`. Repo: `hanzoai/zrok`. License: Apache 2.0. The go.mod still references the upstream module path. Branding partially applied (README, Dockerfile labels reference Hanzo, but internal module paths remain OpenZiti).

## When to use

- Sharing a local web app or API with teammates without deploying
- Exposing development servers through corporate firewalls
- Sharing files/directories as a network drive
- Private service-to-service connectivity between zrok-enabled users
- Self-hosting a sharing platform for your organization
- Replacing ngrok/Cloudflare Tunnel with a zero-trust alternative

## Hard requirements

1. **Hanzo ZT network** (or OpenZiti network) for the overlay fabric
2. **Go 1.25+** and **Node.js 22+** to build from source
3. **CGO enabled** (SQLite requires it)
4. **zrok account** or self-hosted zrok controller instance

## Quick reference

| Item | Value |
|------|-------|
| Binary | `zrok2` (aliased `zrok`) |
| Go module | `github.com/openziti/zrok/v2` |
| Container | `ghcr.io/hanzoai/zrok` |
| Base image | `openziti/ziti-cli:latest` |
| License | Apache 2.0 |
| API spec | `specs/` (OpenAPI/Swagger) |
| SDKs | Go, Node.js, Python |
| Database | SQLite (embedded) |
| Repo | `github.com/hanzoai/zrok` |

## One-file quickstart

### Share a local service publicly

```bash
# Install zrok
# (download binary or build from source)

# Create account and enable environment
zrok invite
zrok enable

# Share localhost:8080 publicly
zrok share public localhost:8080

# Share a directory as a network drive
zrok share public --backend-mode drive ~/Documents

# Share privately with other zrok users
zrok share private localhost:3000
```

### Go SDK

```go
package main

import (
    "github.com/openziti/zrok/v2/sdk/golang/sdk"
)

func main() {
    root, _ := sdk.LoadRoot()

    // Create a private share
    shr, _ := sdk.CreateShare(root, &sdk.ShareRequest{
        BackendMode: sdk.TcpTunnelBackendMode,
        ShareMode:   sdk.PrivateShareMode,
    })

    // Listen for connections
    listener, _ := sdk.NewListener(shr.Token, root)
    defer listener.Close()

    // Accept and handle connections...
}
```

### Self-host with Docker Compose

```yaml
# docker/compose/zrok-instance/
# Contains full self-hosted zrok instance setup
# See: docker/compose/zrok-instance/README.md
```

### Build from source

```bash
# Build UI
cd ui && npm ci && npm run build && cd ..
cd agent/agentUi && npm ci && npm run build && cd ..

# Build binary
CGO_ENABLED=1 go build -tags sqlite_foreign_keys \
  -o zrok2 ./cmd/zrok2/
```

## Core Concepts

### Architecture

```
+------------------+     +-------------------+     +------------------+
|  zrok Client     |     |  zrok Controller  |     |  Hanzo ZT Fabric |
|                  |     |                   |     |                  |
|  zrok share      |---->|  REST API         |---->|  Controller      |
|  zrok access     |     |  Account mgmt     |     |  Routers         |
|  zrok reserve    |     |  Share mgmt       |     |  Smart routing   |
|                  |     |  Metrics/limits    |     |  E2E encryption  |
+------------------+     +-------------------+     +------------------+
                                |
                         +------+------+
                         |   SQLite    |
                         |   (state)   |
                         +-------------+
```

### Sharing Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `public` | Anyone with the URL can access | Dev sharing, demos |
| `private` | Only zrok users with explicit access | Internal services |
| `reserved` | Persistent share with stable token | Long-running services |

### Backend Modes

| Mode | Description |
|------|-------------|
| `proxy` | HTTP reverse proxy to local service (default) |
| `web` | Static file server |
| `drive` | WebDAV file sharing (network drive) |
| `tcpTunnel` | Raw TCP tunnel |
| `udpTunnel` | Raw UDP tunnel |
| `caddy` | Caddy web server with custom config |
| `socks` | SOCKS5 proxy |

### Directory Structure

```
cmd/zrok2/              # Main binary entrypoint
controller/             # zrok controller (API server, share/access management)
agent/                  # zrok agent (daemon, background shares)
  agentUi/              # Agent web UI (Node.js)
endpoints/              # Backend mode implementations (proxy, web, drive, tunnel)
environment/            # Environment/identity management
drives/                 # WebDAV drive implementation
sdk/
  golang/               # Go SDK with examples
  nodejs/               # Node.js SDK
  python/               # Python SDK
rest_client_zrok/       # Generated REST API client
rest_model_zrok/        # Generated REST API models
rest_server_zrok/       # Generated REST API server
specs/                  # OpenAPI/Swagger specifications
tui/                    # Terminal UI (Charm stack)
ui/                     # Web console UI (Node.js/React)
docker/
  compose/
    zrok-instance/      # Self-hosted instance compose
    zrok-private-access/
    zrok-private-share/
    zrok-public-reserved/
  images/               # Docker image definitions
build/                  # Build scripts
canary/                 # Canary testing
etc/                    # Configuration templates
website/                # Documentation website
nfpm/                   # Linux package definitions (deb, rpm)
```

### Relationship to Hanzo ZT

Zrok is the **application layer** on top of Hanzo ZT (the network fabric):

- **Hanzo ZT** (`hanzoai/zt`) = network overlay fabric (controller, routers, tunnelers, SDKs)
- **Hanzo Zrok** (`hanzoai/zrok`) = sharing platform that uses ZT for secure connectivity
- Zrok controller creates ZT services, policies, and configurations automatically
- Users never need to interact with ZT directly -- zrok abstracts it away

### Relationship to Hanzo Tunnel

- **Hanzo Tunnel** (`hanzoai/tunnel`) = WebSocket-based agent bridge for bot/dev control plane
- **Hanzo Zrok** (`hanzoai/zrok`) = OpenZiti-based sharing platform for network services
- Different repos, different purposes, different protocols
- Hanzo Tunnel uses plain WSS; Zrok uses the ZT zero-trust overlay

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| CGO errors on build | Missing C compiler | Install `gcc` and `libc6-dev` |
| SQLite errors | CGO_ENABLED=0 | Must build with `CGO_ENABLED=1` |
| Share fails | No ZT network | Ensure zrok controller has ZT fabric configured |
| "not enabled" | Environment not initialized | Run `zrok enable` first |
| Docker build fails | Wrong platform | Use `--platform linux/amd64` |

## Related Skills

- `hanzo/hanzo-zt.md` - Hanzo ZT zero-trust network fabric (underlying network layer)
- `hanzo/hanzo-tunnel.md` - WebSocket-based agent bridge (different component)
- `hanzo/hanzo-network.md` - Hanzo network infrastructure
- `hanzo/hanzo-platform.md` - PaaS platform

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: zrok, openziti, zero-trust, sharing, tunnel, ngrok-alternative, network-overlay
**Prerequisites**: Go 1.25+, Node.js 22+, CGO, Hanzo ZT network
