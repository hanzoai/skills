# Hanzo DNS - Programmable DNS Server

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-ingress.md`, `hanzo/hanzo-universe.md`, `hanzo/hanzo-platform.md`

## Overview

Hanzo DNS is a **CoreDNS fork** providing a programmable, plugin-based DNS server for Hanzo infrastructure. Go codebase (module `github.com/coredns/coredns`), ships as a single `coredns` binary. Includes a custom `hanzoapi` plugin for API-driven DNS record management. Supports UDP/TCP, DoT (RFC 7858), DoH (RFC 8484), DoH3, DoQ (RFC 9250), and gRPC listeners. License: Apache-2.0.

### Why Hanzo DNS?

- **Plugin chain architecture**: 50+ built-in plugins, each does one thing in a middleware chain
- **Custom hanzoapi plugin**: API-driven DNS records for Hanzo services
- **Multi-protocol**: UDP, TCP, TLS (DoT), HTTPS (DoH/DoH3), QUIC (DoQ), gRPC
- **Kubernetes native**: Built-in `kubernetes` plugin for in-cluster DNS
- **Cloud provider backends**: Route53, Azure DNS, Google Cloud DNS
- **Service discovery**: etcd, Kubernetes, Nomad backends

### Tech Stack

- **Language**: Go 1.25+
- **Build**: Make + `go generate` (plugin registration from `plugin.cfg`)
- **CI**: CircleCI + GitHub Actions (CodeQL, Go tests)
- **Image**: `ghcr.io/hanzoai/dns` (distroless base, nonroot user)

### OSS Base

Repo: `hanzoai/dns` (CoreDNS fork). Default branch: `main`.

## When to use

- Internal DNS for Hanzo K8s clusters
- Programmable DNS with API-driven record management via `hanzoapi` plugin
- Service discovery across Kubernetes, etcd, or Nomad
- DNS-over-HTTPS/QUIC termination at the edge
- GeoIP-based routing and load balancing

## Hard requirements

1. **Port 53** (or alternative via `-dns.port`) available for DNS listeners
2. **Go 1.25+** for building from source
3. **Corefile** configuration file in working directory (or specified with `-conf`)

## Quick reference

| Item | Value |
|------|-------|
| Default Port | 53 (UDP/TCP) |
| Go Module | `github.com/coredns/coredns` |
| Go Version | 1.25+ |
| Binary | `coredns` |
| Config File | `Corefile` |
| Plugin Registry | `plugin.cfg` |
| License | Apache-2.0 |
| Repo | `github.com/hanzoai/dns` |
| Default Branch | `main` |

## One-file quickstart

### Build from source

```bash
git clone https://github.com/hanzoai/dns
cd dns
make
```

### Minimal Corefile

```bash
cat > Corefile <<EOF
.:53 {
    forward . 8.8.8.8
    log
}
EOF
./coredns -conf Corefile
```

### Test

```bash
dig @127.0.0.1 google.com
```

### Docker build

```bash
docker run --rm -i -t \
    -v $PWD:/go/src/github.com/coredns/coredns \
    -w /go/src/github.com/coredns/coredns \
    golang:1.25 sh -c 'GOFLAGS="-buildvcs=false" make gen && GOFLAGS="-buildvcs=false" make'
```

## Core Concepts

### Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              Hanzo DNS (CoreDNS)              │
DNS Query ────────> │                                              │
(UDP/TCP/DoT/DoH/  │  Corefile ──> Plugin Chain (ordered by cfg)  │
 DoQ/gRPC)         │                                              │
                    │  ┌──────┐ ┌──────┐ ┌─────────┐ ┌─────────┐ │
                    │  │ log  │→│cache │→│hanzoapi │→│forward  │ │
                    │  └──────┘ └──────┘ └─────────┘ └─────────┘ │
                    └──────────────────────────────────────────────┘
```

### Plugin System

Plugins are registered in `plugin.cfg` in execution order. Each plugin is a Go package under `plugin/`. To add or remove plugins, edit `plugin.cfg` and run:

```bash
go generate coredns.go
go get
go build
```

### Key Plugins (50+ total)

| Plugin | Purpose |
|--------|---------|
| `hanzoapi` | Hanzo API-driven DNS records (custom) |
| `kubernetes` | Kubernetes service discovery |
| `etcd` | etcd-backed records (SkyDNS compatible) |
| `forward` | Proxy to upstream resolvers |
| `cache` | Response caching |
| `route53` | AWS Route53 backend |
| `azure` | Azure DNS backend |
| `clouddns` | Google Cloud DNS backend |
| `prometheus` | Metrics on `:9153/metrics` |
| `log` | Query logging |
| `dnssec` | On-the-fly DNSSEC signing |
| `loadbalance` | Round-robin response shuffling |
| `rewrite` | Query rewriting |
| `acl` | Access control lists |
| `geoip` | GeoIP-based metadata |
| `tls` / `quic` / `https` | Transport protocol listeners |
| `nomad` | HashiCorp Nomad service discovery |

### Protocol Examples

```
# DNS over TLS + gRPC
tls://example.org grpc://example.org {
    whoami
}

# DNS over QUIC
quic://example.org {
    whoami
    tls mycert mykey
}

# DNS over HTTPS
https://example.org {
    whoami
    tls mycert mykey
}
```

### Directory Structure

```
dns/
  Corefile               # Runtime config (not in repo, user-created)
  Dockerfile             # Distroless container (nonroot, cap_net_bind_service)
  Makefile               # Build: make, make gen, make clean
  Makefile.docker        # Docker image builds
  Makefile.release       # Release automation
  coredns.go             # Entry point + go:generate directive
  plugin.cfg             # Plugin registration (order matters!)
  go.mod                 # Go module (github.com/coredns/coredns)
  core/                  # Core DNS server, plugin framework
  coremain/              # Main function, startup
  plugin/                # All plugins (50+ subdirectories)
    hanzoapi/            # Custom Hanzo API plugin
    kubernetes/          # K8s service discovery
    forward/             # Upstream forwarding
    cache/               # Response cache
    etcd/                # etcd backend
    ...
  pb/                    # Protobuf definitions
  request/               # DNS request helpers
  test/                  # Integration tests
  zones/                 # Example zone files
  man/                   # Man pages
  scripts/               # Build/release scripts
  .circleci/             # CircleCI config
  .github/               # GitHub Actions (CodeQL, tests)
```

### Build Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CGO_ENABLED` | 0 | Static binary by default |
| `GOTAGS` | `grpcnotrace` | Build tags |
| `COREDNS_PLUGINS` | - | Extra plugins (comma-separated, plugin.cfg format) |
| `STRIP_FLAGS` | `-s -w` | Binary stripping |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 53 in use | systemd-resolved or another DNS | Change port with `-dns.port 1053` or stop conflicting service |
| Plugin not found | Not in plugin.cfg | Add to plugin.cfg, run `go generate coredns.go && go build` |
| Zone not loading | Corefile syntax error | Check Corefile syntax, ensure zone files exist |
| K8s plugin fails | Missing RBAC permissions | Ensure ServiceAccount has list/watch on services/endpoints |
| hanzoapi not responding | Plugin not compiled in | Verify `hanzoapi:hanzoapi` is in plugin.cfg |

## Related Skills

- `hanzo/hanzo-ingress.md` - Ingress controller (routes traffic)
- `hanzo/hanzo-universe.md` - K8s infrastructure where DNS runs
- `hanzo/hanzo-platform.md` - PaaS deployment platform
- `hanzo/hanzo-pubsub.md` - NATS messaging (sibling infrastructure)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: dns, coredns, nameserver, service-discovery, kubernetes, etcd
**Prerequisites**: Go 1.25+ or Docker
