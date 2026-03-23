# Hanzo ZT - Zero-Trust Network Fabric

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-zrok.md`, `hanzo/hanzo-tunnel.md`, `hanzo/hanzo-k8s.md`

## Overview

Hanzo ZT is the **zero-trust network overlay fabric** for Hanzo's secure networking. Fork of OpenZiti. Provides a programmable mesh network with identity-based access control, end-to-end encryption, dark services, and smart routing. All executables (controller, router, tunnel) built from a single repo. Runs on hanzo-k8s.

## When to use

- Zero-trust networking between services (no open ports)
- Secure tunneling for remote access to K8s services
- Dark services that have no public endpoints
- mTLS/PKI-based identity for service-to-service auth
- End-to-end encrypted connections (not just TLS termination)

## Hard requirements

1. **ZT Controller** must be running for fabric management
2. **ZT Router** provides mesh connectivity
3. **All clients provisioned** with certificates -- no anonymous access
4. **PKI managed** by ZT's built-in certificate authority

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/hanzozt/zt/v2` |
| Go version | 1.25.3 |
| Version | 2.0 |
| License | Apache 2.0 |
| Upstream | OpenZiti |
| Repo | `github.com/hanzoai/zt` |
| K8s manifests | `universe/infra/k8s/zt/` |
| Components | controller, router, tunnel |
| Storage | BoltDB (bbolt), Raft for HA |
| PKI | Built-in CA and identity management |
| API | OpenAPI-generated REST |

## Architecture

```
                 ZT Controller
                 (management API)
                      |
             +--------+--------+
             |                 |
         ZT Router         ZT Router
         (edge)            (fabric)
             |                 |
      +------+------+   +-----+-----+
      |      |      |   |           |
   Tunnel  Tunnel  SDK  Dark      Dark
   (client)(client)     Service   Service
```

### Key concepts

| Concept | Description |
|---------|-------------|
| **Controller** | Management plane. Stores policies, identities, services. REST API. |
| **Router** | Data plane. Mesh routing between endpoints. Edge + fabric roles. |
| **Tunnel** | Client-side agent. Intercepts traffic, routes through fabric. |
| **Dark service** | No open ports. Connects outbound to fabric. Invisible to port scanners. |
| **Identity** | X.509 certificate-based. Provisioned per client/service. |
| **Service** | Named endpoint accessible through the fabric. |
| **Service policy** | Controls which identities can access which services. |

## Core packages (hanzozt namespace)

| Package | Version | Purpose |
|---------|---------|---------|
| `hanzozt/foundation/v2` | v2.0.86 | Utilities, logging, config |
| `hanzozt/channel/v4` | v4.3.4 | Secure message channels |
| `hanzozt/edge-api` | v0.26.52 | Edge management API |
| `hanzozt/identity` | v1.0.125 | Identity and cert management |
| `hanzozt/sdk-golang` | v1.4.1 | Go SDK for ZT network access |
| `hanzozt/transport/v2` | v2.0.209 | Transport layer (TCP, TLS, WS) |
| `hanzozt/storage` | v0.4.37 | BoltDB-based storage |
| `hanzozt/dilithium` | v0.3.5 | Post-quantum transport |

## Quickstart

```bash
# Initialize PKI
hanzozt pki create ca --ca-name "Hanzo ZT CA"
hanzozt pki create intermediate --ca-name "Hanzo ZT CA"

# Start controller
hanzozt controller run --config controller.yaml

# Start router
hanzozt router run --config router.yaml

# Create identity for a client
hanzozt edge create identity device my-laptop
hanzozt edge enroll --jwt my-laptop.jwt --out my-laptop.json

# Create a dark service
hanzozt edge create service my-api
hanzozt edge create terminator my-api --router fabric-router --binding sdk

# Create service policy (who can access)
hanzozt edge create service-policy my-api-access Dial \
  --identity-roles "#my-team" \
  --service-roles "@my-api"
```

## K8s deployment

```yaml
# universe/infra/k8s/zt/
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zt-controller
  namespace: hanzo
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: controller
          image: ghcr.io/hanzoai/zt-controller:latest
          ports:
            - containerPort: 1280  # Management API
            - containerPort: 6262  # Control plane
```

## Integration with zrok

Hanzo zrok (fork of OpenZiti's zrok) provides user-facing tunnel sharing on top of the ZT fabric:

```bash
# Share a local service through ZT fabric
zrok share public localhost:3000

# Access a shared service
zrok access private <share-token>
```

See `hanzo/hanzo-zrok.md` for details.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Identity enrollment fails | JWT expired | Generate new JWT with `hanzozt edge create enrollment` |
| Service unreachable | Service policy missing | Create Dial policy for the identity |
| Router not connecting | Controller unreachable | Check controller address in router config |
| Tunnel not intercepting | Wrong service config | Verify intercept.v1 config matches service name |

## Related Skills

- `hanzo/hanzo-zrok.md` -- Zero-trust sharing platform
- `hanzo/hanzo-tunnel.md` -- WebSocket tunnel bridge
- `hanzo/hanzo-k8s.md` -- K8s infrastructure
- `hanzo/hanzo-ingress.md` -- Public ingress (ZT is for private)

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: zero-trust, networking, openziti, pki, mtls, dark-services
**Prerequisites**: PKI concepts, networking basics
