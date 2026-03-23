# Hanzo Ingress - L7 Reverse Proxy and Load Balancer

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-k8s.md`, `hanzo/hanzo-deploy.md`, `hanzo/hanzo-static.md`

## Overview

Hanzo Ingress is a **Kubernetes-native L7 reverse proxy and load balancer** that serves as the front door for all Hanzo production traffic. Fork of Traefik. Watches Kubernetes Ingress resources, auto-provisions TLS certificates via Let's Encrypt, routes traffic to internal services. Single static Go binary, zero runtime dependencies. Default IngressClass (`hanzo`) on hanzo-k8s cluster.

## When to use

- Routing external traffic to K8s services
- TLS termination for `*.hanzo.ai`, `*.hanzo.team`, and other domains
- Applying middleware (rate limiting, auth, compression) to routes
- Load balancing across service replicas
- Creating Ingress resources for new services

## Hard requirements

1. **No nginx, no caddy** -- Hanzo Ingress is the only reverse proxy
2. **IngressClass `hanzo`** on all Ingress resources
3. **Cloudflare DNS** with full SSL proxied to cluster LB
4. **Ports 80/443** via hostNetwork mode
5. **cert-manager** for Let's Encrypt TLS provisioning

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
| Upstream | Traefik |
| Repo | `github.com/hanzoai/ingress` |
| Module | `github.com/hanzoai/ingress/v3` |
| Go version | 1.26 |
| K8s manifests | `universe/infra/k8s/` (per-service Ingress) |
| K8s context | `do-sfo3-hanzo-k8s` |

## Architecture

```
            Internet
               |
      +--------+--------+
      | Cloudflare CDN   |
      | DNS, WAF, DDoS   |
      +--------+--------+
               |
      +--------+--------+
      | Hanzo Ingress    |   L7 reverse proxy
      | (ports 80/443)   |   TLS termination
      | IngressClass:    |   Route matching
      |   "hanzo"        |   Load balancing
      +--+-+-+-+-+--+---+
         | | | | |  |
   +-----+ | | | |  +--------+
   |   +---+ | | +-----+     |
   v   v     v v       v     v
 +---+----+ +---+ +------+ +------+
 |Gateway | |IAM| | KMS  | |Cloud |
 |(API)   | +---+ +------+ +------+
 +---+----+
     |
 Backend services
```

## Request flow

1. DNS resolves `*.hanzo.ai` to Cloudflare
2. Cloudflare proxies to hanzo-k8s cluster LB (`165.232.146.176`)
3. Hanzo Ingress terminates TLS, matches host/path rules
4. Request forwarded to matching backend service
5. For API traffic (`api.hanzo.ai`), routes to Hanzo Gateway for endpoint-level routing

## Routed domains (production)

| Domain | Backend Service | Port |
|--------|-----------------|------|
| `hanzo.ai` | hanzo-app | 3000 |
| `api.hanzo.ai`, `llm.hanzo.ai` | Hanzo Gateway | 8000 |
| `hanzo.id`, `lux.id`, `zoo.id`, `pars.id` | IAM (Casdoor) | 8000 |
| `kms.hanzo.ai` | KMS (Infisical) | 8080 |
| `platform.hanzo.ai` | Platform (Dokploy) | 3000 |
| `console.hanzo.ai` | Console (Langfuse) | 3000 |
| `cloud.hanzo.ai` | Cloud (Casibase) | 8000 |
| `chat.hanzo.ai` | Chat (LibreChat) | 3080 |
| `hanzo.team` | Team (Huly fork) | 8087 |
| `billing.hanzo.ai` | Billing | 80 |
| `o11y.hanzo.ai` | SigNoz | 3301 |

## Create an Ingress resource

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

## Middleware

Apply via Kubernetes Ingress annotations or IngressRoute CRDs:

```yaml
metadata:
  annotations:
    hanzo.ai/ingress-ratelimit-average: "100"
    hanzo.ai/ingress-ratelimit-burst: "200"
```

Built-in middleware: auth, ratelimiter, circuitbreaker, retry, compress, headers, ipallowlist, buffering, inflightreq, redirect, stripprefix, grpcweb, observability (OTEL), metrics (Prometheus), accesslog.

## K8s manifests

```
k8s/hanzo/
  rbac.yaml           # ServiceAccount + ClusterRole
  ingressclass.yaml   # IngressClass "hanzo" (default)
  deployment.yaml     # 2-replica Deployment, hostNetwork
  service.yaml        # LoadBalancer Service
  middlewares.yaml    # Default middleware configurations
```

## Production CLI flags

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

## Build and deploy

```bash
# Build from source
make build

# Deploy to production
kubectl --context do-sfo3-hanzo-k8s apply -f k8s/hanzo/

# Verify pods
kubectl --context do-sfo3-hanzo-k8s -n hanzo get pods -l app=hanzo-ingress

# View logs
kubectl --context do-sfo3-hanzo-k8s -n hanzo logs -l app=hanzo-ingress --tail=100 -f
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port conflict on 80/443 | Other ingress controller | Remove conflicting controller |
| TLS cert not provisioned | cert-manager not configured | Install cert-manager with letsencrypt issuer |
| 404 on known domain | Ingress resource missing | Check `ingressClassName: hanzo` |
| Dashboard not loading | Disabled in production | Use port-forward or enable in non-prod |
| WebSocket not working | Missing upgrade headers | Add WebSocket middleware to IngressRoute |

## Related Skills

- `hanzo/hanzo-static.md` -- Static file serving plugin
- `hanzo/hanzo-k8s.md` -- K8s infrastructure
- `hanzo/hanzo-deploy.md` -- Deployment workflow
- `hanzo/hanzo-gateway.md` -- API gateway (behind Ingress)

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: ingress, reverse-proxy, load-balancer, kubernetes, tls, traefik
**Prerequisites**: Kubernetes, DNS, TLS basics
