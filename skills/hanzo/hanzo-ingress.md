# Hanzo Ingress - Cloud-Native L7 Reverse Proxy and Load Balancer

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-universe.md`

## Overview

Hanzo Ingress is a **Kubernetes-native L7 reverse proxy and load balancer** that serves as the front door for all Hanzo production traffic. Built in Go (Traefik fork), it watches Kubernetes Ingress resources, auto-provisions TLS certificates via Let's Encrypt, and routes traffic to internal services. Single static binary, zero runtime dependencies. Live on `hanzo-k8s` cluster as the default IngressClass (`hanzo`).

### Why Hanzo Ingress?

- **Kubernetes-native**: Watches Ingress resources, auto-configures routes
- **Automatic TLS**: Let's Encrypt provisioning and renewal with wildcard support
- **Dynamic configuration**: Zero-restart config updates as Ingress resources change
- **Full protocol support**: HTTP/2, gRPC, WebSocket
- **Built-in middleware**: Rate limiting, circuit breakers, retries, auth, compression
- **Web dashboard**: React UI for route visualization and health monitoring
- **Multi-provider**: Kubernetes Ingress (primary), Docker, File, Consul, Etcd, ECS

### Tech Stack

- **Language**: Go 1.26
- **Module**: `github.com/hanzoai/ingress/v3`
- **Binary**: `hanzo-ingress`
- **WebUI**: React dashboard (Node.js build stage)
- **Image**: `ghcr.io/hanzoai/ingress:latest`
- **Build**: Multi-stage Dockerfile (Node webui + Go binary)

### OSS Base

Repo: `hanzoai/ingress` (Traefik fork). Entry point: `cmd/traefik/`.

## When to use

- Routing external traffic to K8s services
- TLS termination for `*.hanzo.ai` and other domains
- Applying middleware (rate limiting, auth, compression) to routes
- Load balancing across service replicas
- Creating Ingress resources for new services on hanzo-k8s

## Hard requirements

1. **Kubernetes cluster** with RBAC permissions for Ingress resources
2. **Ports 80/443** available (hostNetwork mode)
3. **DNS** resolving to cluster LB (Cloudflare proxied)

## Quick reference

| Item | Value |
|------|-------|
| Image | `ghcr.io/hanzoai/ingress:latest` |
| Binary | `hanzo-ingress` |
| IngressClass | `hanzo` (default) |
| Controller | `hanzo.ai/ingress-controller` |
| Health check | `GET /ping` on port 80 |
| Replicas | 2 (production) |
| Namespace | `hanzo` |
| Network | hostNetwork (direct port binding) |
| Ports | 80 (HTTP), 443 (HTTPS) |
| Resources | 100m-1000m CPU, 128Mi-512Mi memory |
| Repo | `github.com/hanzoai/ingress` |
| K8s context | `do-sfo3-hanzo-k8s` |

## One-file quickstart

### Docker

```bash
docker run -d \
  --name hanzo-ingress \
  -p 80:80 \
  -p 443:443 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/hanzoai/ingress:latest \
  --entrypoints.web.address=:80 \
  --entrypoints.websecure.address=:443 \
  --providers.docker=true \
  --ping=true
```

### Kubernetes

```bash
kubectl apply -f https://raw.githubusercontent.com/hanzoai/ingress/master/k8s/hanzo/rbac.yaml
kubectl apply -f https://raw.githubusercontent.com/hanzoai/ingress/master/k8s/hanzo/ingressclass.yaml
kubectl apply -f https://raw.githubusercontent.com/hanzoai/ingress/master/k8s/hanzo/deployment.yaml
kubectl apply -f https://raw.githubusercontent.com/hanzoai/ingress/master/k8s/hanzo/service.yaml
```

### Build from Source

```bash
make build
./hanzo-ingress --configFile=config.toml
```

### Create an Ingress Resource

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-service
  namespace: hanzo
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  ingressClassName: hanzo
  tls:
  - hosts:
    - my-service.hanzo.ai
    secretName: my-service-tls
  rules:
  - host: my-service.hanzo.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: 8080
```

## Core Concepts

### Architecture

```
              Internet
                 |
        +--------+--------+
        | Cloudflare CDN  |
        | DNS, WAF, DDoS  |
        +--------+--------+
                 |
        +--------+--------+
        | Hanzo Ingress   |   L7 reverse proxy
        | (ports 80/443)  |   TLS termination
        | IngressClass:   |   Route matching
        |   "hanzo"       |   Load balancing
        +--+-+-+-+-+--+---+
           | | | | |  |
     +-----+ | | | |  +--------+
     |   +---+ | | +-----+     |
     |   |  +--+ +--+    |     |
     v   v  v       v    v     v
  +-----+-----+  +----+ +---+ +-------+  +-----+
  | Hanzo     |  | IAM| |KMS| | Cloud |  | PaaS|
  | Gateway   |  +----+ +---+ +-------+  +-----+
  | (API)     |
  +--+--+--+--+
     |  |  |
     v  v  v
  +------+------+------+
  |Engine|Search|  LLM |    Backend services
  +------+------+------+
```

### Request Flow

1. DNS resolves `*.hanzo.ai` to Cloudflare
2. Cloudflare proxies to hanzo-k8s cluster LB (`24.199.76.156`)
3. Hanzo Ingress terminates TLS, matches host/path rules
4. Request forwarded to the matching backend service
5. For API traffic (`api.hanzo.ai`), routes to Hanzo Gateway for endpoint-level routing

### Routed Domains (Production)

| Domain | Backend Service |
|--------|-----------------|
| `hanzo.ai` | hanzo-app |
| `api.hanzo.ai`, `llm.hanzo.ai` | Hanzo Gateway |
| `hanzo.id`, `lux.id`, `zoo.id` | IAM (Casdoor) |
| `kms.hanzo.ai` | KMS (Infisical) |
| `platform.hanzo.ai` | Platform (Dokploy) |
| `console.hanzo.ai` | Console |
| `cloud.hanzo.ai` | Cloud |

### Middleware

Middlewares are applied via Kubernetes Ingress annotations:

```yaml
metadata:
  annotations:
    hanzo.ai/ingress-ratelimit-average: "100"
    hanzo.ai/ingress-ratelimit-burst: "200"
```

Key built-in middlewares: auth, ratelimiter, circuitbreaker, retry, compress, headers, ipallowlist, buffering, inflightreq, redirect, stripprefix, grpcweb, observability (OpenTelemetry), metrics (Prometheus/Datadog/StatsD/OTLP), accesslog.

### K8s Manifests

```
k8s/hanzo/
  rbac.yaml           # ServiceAccount + ClusterRole
  ingressclass.yaml   # IngressClass "hanzo" (default)
  deployment.yaml     # 2-replica Deployment, hostNetwork
  service.yaml        # LoadBalancer Service
  middlewares.yaml    # Default middleware configurations
```

### Configuration (Production CLI)

```bash
./hanzo-ingress \
  --providers.kubernetesingress=true \
  --providers.kubernetesingress.ingressendpoint.publishedservice=hanzo/hanzo-ingress \
  --providers.kubernetesingress.allowemptyservices=true \
  --entrypoints.web.address=:80 \
  --entrypoints.websecure.address=:443 \
  --entrypoints.websecure.http.tls=true \
  --ping=true \
  --ping.entryPoint=web \
  --api.dashboard=false \
  --log.level=INFO \
  --accesslog=true
```

### Repository Structure

```
cmd/traefik/          # Binary entry point
internal/             # Core routing, middleware, provider logic
pkg/                  # Public packages, config types, version
webui/                # Built-in dashboard (React)
k8s/
  hanzo/              # Production K8s manifests (hanzo-k8s cluster)
  lux/                # Lux cluster manifests (lux-k8s cluster)
integration/          # Integration test suite
contrib/              # Community contributed configs
docs/                 # Extended documentation
Dockerfile            # Multi-stage build (Node webui + Go binary)
Makefile              # Build, test, Docker targets
```

## Build and Test

```bash
# Build binary
make build

# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Build Docker image
make build-image

# Lint
make lint
```

## Deploy / Update

```bash
kubectl --context do-sfo3-hanzo-k8s apply -f k8s/hanzo/

# Verify pods
kubectl --context do-sfo3-hanzo-k8s -n hanzo get pods -l app=hanzo-ingress

# View logs
kubectl --context do-sfo3-hanzo-k8s -n hanzo logs -l app=hanzo-ingress --tail=100 -f
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port conflict on 80/443 | Other ingress controller running | Remove conflicting controller |
| TLS cert not provisioned | cert-manager not configured | Install cert-manager with letsencrypt issuer |
| 404 on known domain | Ingress resource missing or wrong class | Check `ingressClassName: hanzo` |
| Dashboard not loading | `api.dashboard=false` in production | Enable in non-prod or use port-forward |

## Related Projects

| Project | Role |
|---------|------|
| [Hanzo Gateway](https://github.com/hanzoai/gateway) | API gateway, rate limiting, endpoint routing |
| [Hanzo Engine](https://github.com/hanzoai/engine) | GPU inference engine, model serving |
| [Hanzo Edge](https://github.com/hanzoai/edge) | On-device inference runtime |

```
Internet -> Ingress (TLS/L7) -> Gateway (API routing) -> Engine (inference) / Services
```

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS for deploying applications
- `hanzo/hanzo-cloud.md` - Cloud dashboard
- `hanzo/hanzo-universe.md` - Production K8s infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ingress, reverse-proxy, load-balancer, kubernetes, tls, traefik
**Prerequisites**: Kubernetes, DNS, TLS basics
