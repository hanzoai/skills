# Hanzo ZT - Zero-Trust Network Fabric

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-zrok.md`, `hanzo/hanzo-tunnel.md`, `hanzo/hanzo-network.md`

## Overview

Hanzo ZT is the **zero-trust network overlay fabric** that powers Hanzo's secure networking infrastructure. It provides a programmable, scalable mesh network with identity-based access control, end-to-end encryption, dark services, and smart routing. Fork of OpenZiti (`openziti/ziti`), rebranded to the `hanzozt` namespace. All Hanzo ZT executables (controller, router, tunnel) are built from this single repo.

### Why Hanzo ZT?

- **Zero trust**: Every client has provisioned certificates; access is per-service, not per-network
- **Dark services**: Services have no open ports; they reach out to the fabric, not the other way around
- **End-to-end encryption**: Traffic encrypted from client application to server application
- **Smart routing**: Scalable mesh fabric with intelligent path selection
- **Peer-to-peer**: Direct connections between endpoints when possible
- **SDK embeddable**: Integrate zero-trust networking directly into applications
- **Multi-platform tunnelers**: Windows, macOS, Linux, iOS, Android, Raspberry Pi

### Tech Stack

- **Language**: Go
- **Go module**: `github.com/hanzozt/zt/v2`
- **Go version**: 1.25.3
- **Version**: 2.0
- **License**: Apache 2.0
- **API**: OpenAPI-generated REST (controller management API, edge management API)
- **Storage**: BoltDB (bbolt) for controller/router state, Raft for HA clustering
- **PKI**: Built-in certificate authority and identity management
- **Dependencies**: All core libs rebranded to `hanzozt/*` namespace

### Key Dependencies (hanzozt namespace)

| Package | Version | Purpose |
|---------|---------|---------|
| `hanzozt/foundation/v2` | v2.0.86 | Common utilities, logging, config |
| `hanzozt/channel/v4` | v4.3.4 | Secure message channels |
| `hanzozt/edge-api` | v0.26.52 | Edge management API models/client |
| `hanzozt/identity` | v1.0.125 | Identity and certificate management |
| `hanzozt/sdk-golang` | v1.4.1 | Go SDK for ZT network access |
| `hanzozt/transport/v2` | v2.0.209 | Transport layer (TCP, TLS, WS) |
| `hanzozt/storage` | v0.4.37 | BoltDB-based data storage |
| `hanzozt/metrics` | v1.4.3 | Metrics collection |
| `hanzozt/secretstream` | v0.1.47 | Encrypted streaming |
| `hanzozt/agent` | v1.0.33 | Agent protocol |
| `hanzozt/jwks` | v1.0.6 | JSON Web Key Set support |
| `hanzozt/dilithium` | v0.3.5 | Post-quantum transport (indirect) |
| `hanzozt/runzmd` | v1.0.88 | Runnable markdown documentation |
| `hanzozt/x509-claims` | v1.0.3 | X.509 certificate claim parsing |
| `hanzozt/xweb/v3` | v3.0.3 | Web server framework |
| `hanzozt/zt-db-explorer` | v1.1.3 | Database exploration tools |

### OSS Base

Fork of `openziti/ziti`. Repo: `hanzoai/zt`. Module path fully rebranded to `hanzozt/*`. The upstream OpenZiti project was developed by NetFoundry; Hanzo maintains this fork with the `hanzozt` namespace. Related ecosystem repos referenced in README: `hanzoai/edge`, `hanzoai/fabric`, `hanzoai/foundation`, plus SDKs (`zt-sdk-c`, `sdk-golang`, `zt-sdk-jvm`, `zt-sdk-swift`, `zt-sdk-nodejs`, `zt-sdk-csharp`).

## When to use

- Building zero-trust network overlays for Hanzo services
- Providing the fabric layer that Hanzo Zrok runs on top of
- Connecting distributed services without exposing ports
- Identity-based service access (not IP-based)
- Embedding secure networking into applications via SDKs
- Running tunnelers/proxies for existing applications
- HA controller clusters with Raft consensus

## Hard requirements

1. **Go 1.25+** to build from source
2. **PKI setup**: Controller requires certificate authority chain
3. **Network connectivity**: Controller and routers need to reach each other (outbound only for dark routers)
4. **BoltDB storage**: Filesystem-backed; no external database required

## Quick reference

| Item | Value |
|------|-------|
| Binary | `zt` |
| Go module | `github.com/hanzozt/zt/v2` |
| Version | 2.0 |
| License | Apache 2.0 |
| Controller API | REST (OpenAPI) |
| Storage | BoltDB (bbolt) + Raft (HA) |
| PKI | Built-in CA + x509 |
| Docs | `hanzozt.dev/docs` |
| Community | `community.hanzozt.dev` (Discourse) |
| Repo | `github.com/hanzoai/zt` |

## One-file quickstart

### Local Development

```bash
# Build from source
go build -o zt ./zt/

# See local development tutorial
cat doc/002-local-dev.md

# See local deployment tutorial
cat doc/003-local-deploy.md

# PKI setup tutorial
cat doc/004-controller-pki.md
```

### Quickstart (Docker)

```bash
# Quickstart configurations in quickstart/ directory
ls quickstart/
```

## Core Concepts

### Components

```
+------------------+     +------------------+     +------------------+
|  ZT Controller   |     |  ZT Router(s)    |     |  ZT Tunneler     |
|                  |     |                  |     |                  |
|  - PKI/CA        |<--->|  - Mesh fabric   |<--->|  - Proxy mode    |
|  - REST API      |     |  - Smart routing |     |  - TUN mode      |
|  - Identity mgmt |     |  - Link mgmt     |     |  - Host mode     |
|  - Policy engine |     |  - Metrics       |     |                  |
|  - Raft (HA)     |     |  - xgress layers |     |                  |
+------------------+     +------------------+     +------------------+
         |
  +------+------+
  |   BoltDB    |
  |   (state)   |
  +-------------+
```

### Controller

The central management plane:

- **Identity management**: Creates and manages identities with x509 certificates
- **Policy engine**: Service policies (Bind/Dial), edge router policies, service edge router policies
- **REST API**: Full management API (OpenAPI spec in `controller/specs/`)
- **Raft clustering**: HA mode with `hashicorp/raft` for controller redundancy
- **OIDC auth**: External identity provider integration (`controller/oidc_auth/`)
- **JWT signing**: Token-based auth for edge components

### Router

The data plane fabric:

- **Mesh networking**: Routers form a mesh with smart routing between them
- **xgress protocols**: Extensible ingress/egress protocol framework
  - `xgress_edge` -- Edge SDK connections
  - `xgress_edge_tunnel` -- Tunneler connections
  - `xgress_proxy` -- TCP proxy
  - `xgress_proxy_udp` -- UDP proxy
  - `xgress_transport` -- Direct transport
  - `xgress_geneve` -- Geneve encapsulation
- **Link management**: Manages inter-router links with health monitoring
- **Load balancing**: Multiple strategies (`xt_random`, `xt_smartrouting`, `xt_sticky`, `xt_weighted`)
- **Dark mode**: Routers can be dark (outbound-only connections to the fabric)
- **Metrics collection**: Performance and usage metrics

### Tunnel

The client-side connectivity:

- **DNS interception**: Resolves ZT service names via local DNS
- **Health checks**: Service health monitoring
- **Intercept**: Traffic interception for transparent proxying
- **UDP virtual connections**: UDP support over the ZT fabric
- **Router integration**: Embedded tunnel capability in routers

### Security Model

1. **Identity-based**: Every component has an x509 certificate identity
2. **Mutual TLS**: All fabric communication is mTLS
3. **Service policies**: Bind (host) and Dial (access) policies per identity per service
4. **Dark services**: No listening ports; services dial out to the fabric
5. **Dark routers**: Private network routers make only outbound connections
6. **End-to-end encryption**: Optional app-level encryption beyond transport encryption
7. **Posture checks**: Device posture validation before granting access

### Directory Structure

```
zt/                     # Main binary
  main.go               # Entrypoint
  cmd/                  # CLI commands (controller, router, tunnel subcommands)
  enroll/               # Enrollment logic
  pki/                  # PKI utilities
  run/                  # Runtime management
  tunnel/               # Tunnel command implementation
controller/             # Controller implementation
  api/                  # Controller API handlers
  config/               # Controller configuration
  db/                   # Database layer (BoltDB)
  model/                # Data models
  network/              # Network management
  raft/                 # Raft consensus for HA
  oidc_auth/            # OIDC authentication
  jwtsigner/            # JWT token signing
  handler_ctrl/         # Control channel handlers
  handler_edge_ctrl/    # Edge control handlers
  handler_mgmt/         # Management API handlers
  handler_peer_ctrl/    # Peer controller handlers
  rest_client/          # Generated REST client
  rest_model/           # Generated REST models
  rest_server/          # Generated REST server
  specs/                # OpenAPI specifications
  xt_*/                 # Load balancing strategies
  xctrl/                # Controller extension point
router/                 # Router implementation
  forwarder/            # Packet forwarding
  link/                 # Inter-router link management
  metrics/              # Router metrics
  state/                # Router state management
  xgress_*/             # Ingress/egress protocol implementations
  xlink_*/              # Inter-router link implementations
tunnel/                 # Tunnel library
  dns/                  # DNS interception
  entities/             # Tunnel entity management
  health/               # Health checking
  intercept/            # Traffic interception
  router/               # Tunnel-as-router mode
  udp_vconn/            # UDP virtual connections
common/                 # Shared types and utilities
internal/               # Internal packages
ztrest/                 # REST client utilities
zttest/                 # Test utilities
setup-cli/              # CLI setup tooling
tests/                  # Integration tests
quickstart/             # Quickstart configurations
dist/                   # Distribution packaging
doc/                    # Developer documentation
  001-overview.md
  002-local-dev.md
  003-local-deploy.md
  004-controller-pki.md
etc/                    # Configuration templates
```

### Relationship to Other Hanzo Components

```
                    +-------------------+
                    |  Hanzo ZT (zt)    |  <-- THIS REPO: network fabric
                    |  Controller       |
                    |  Router           |
                    |  Tunnel           |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
     +--------+--------+            +--------+--------+
     |  Hanzo Zrok     |            |  Hanzo Tunnel   |
     |  (zrok)         |            |  (tunnel)       |
     |                 |            |                 |
     |  Sharing        |            |  Bot agent      |
     |  platform       |            |  bridge (WSS)   |
     |  Built ON ZT    |            |  Independent    |
     +-----------------+            +-----------------+
```

- **Hanzo ZT** (`hanzoai/zt`) = The network fabric (this repo)
- **Hanzo Zrok** (`hanzoai/zrok`) = Sharing platform built on top of ZT
- **Hanzo Tunnel** (`hanzoai/tunnel`) = WebSocket agent bridge (does NOT use ZT currently; has a hook for future ZT integration)

### SDKs (Separate Repos)

| SDK | Repo | Language |
|-----|------|----------|
| C SDK | `hanzoai/zt-sdk-c` | C |
| Go SDK | `hanzoai/sdk-golang` | Go |
| JVM SDK | `hanzoai/zt-sdk-jvm` | Java/Kotlin |
| Swift SDK | `hanzoai/zt-sdk-swift` | Swift |
| Node.js SDK | `hanzoai/zt-sdk-nodejs` | JavaScript |
| C# SDK | `hanzoai/zt-sdk-csharp` | C# |

### Documentation Site

Powered by `hanzoai/zt-doc` (Docusaurus). Hosted at `hanzozt.dev`.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Controller won't start | Missing PKI certs | Follow `doc/004-controller-pki.md` |
| Router can't connect | Controller unreachable | Check controller address in router config |
| Identity enrollment fails | Expired enrollment token | Generate new enrollment token via controller API |
| Service unreachable | Missing service policy | Create Bind + Dial policies for the identity |
| Build fails | Wrong Go version | Requires Go 1.25+ |
| Raft cluster issues | Split brain | Ensure odd number of controllers, check network |

## Related Skills

- `hanzo/hanzo-zrok.md` - Zero-trust sharing platform (built on this fabric)
- `hanzo/hanzo-tunnel.md` - WebSocket agent bridge (separate component)
- `hanzo/hanzo-network.md` - Hanzo network infrastructure
- `hanzo/hanzo-node.md` - Hanzo node (Rust, different from ZT)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: zero-trust, openziti, network-overlay, mesh, controller, router, tunnel, pki, mTLS
**Prerequisites**: Go 1.25+, PKI/certificate knowledge, networking fundamentals
