# Hanzo Gateway - API Gateway for Hanzo and Lux Networks

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-id.md`, `hanzo/hanzo-commerce.md`

## Overview

Hanzo Gateway is a **high-performance API gateway** built on KrakenD/Lura that routes 147+ endpoints across production clusters. It serves as the unified API entry point for all Hanzo (`api.hanzo.ai`) and Lux (`api.lux.network`) traffic with per-endpoint rate limiting, JWT authentication, billing enforcement, CORS, circuit breakers, host-based reverse proxying, and ZAP binary transport -- all driven by declarative JSON configuration.

### Why Hanzo Gateway?

- **Unified entry point**: Single binary routes to 40+ backend services across two K8s clusters
- **OpenAI-compatible**: `/v1/chat/completions`, `/v1/models`, etc. work with any OpenAI SDK
- **Host-based routing**: Transparent reverse proxy for 40+ Hanzo domains (hanzo.id, chat.hanzo.ai, platform.hanzo.ai, etc.)
- **JWT + billing middleware**: Validates IAM tokens via JWKS, checks Commerce billing balance, injects identity headers
- **Widget security**: Per-IP rate limiting and origin validation for public widget keys (`hz_*`)
- **ZAP binary transport**: TLS 1.3+PQ (post-quantum) listener for native binary protocol backends via `luxfi/zap`
- **Plugin system**: Server, client, and modifier plugins loaded at runtime

### Tech Stack

- **Language**: Go 1.26
- **Core framework**: KrakenD (Lura v2.12.1) + Gin
- **Module**: `github.com/hanzoai/gateway`
- **Binary**: `gateway` (also builds `ingress` sidecar)
- **Config**: Declarative JSON (`gateway.json` per cluster)
- **Image**: `ghcr.io/hanzoai/gateway:latest`

### OSS Base

Fork of KrakenD Community Edition with Hanzo-specific additions: auth middleware, widget security, host-based proxy routing, ZAP binary transport, and billing enforcement. Upstream telemetry reporting is disabled.

Repo: `hanzoai/gateway`.

## When to use

- Adding a new API endpoint that routes to a backend service
- Configuring rate limits for specific endpoints
- Adding a new Hanzo subdomain that needs reverse proxying
- Debugging request routing issues between gateway and backends
- Understanding how auth headers flow from client to backend
- Setting up a new cluster gateway instance

## Hard requirements

1. **Go 1.26+** for building from source
2. **K8s cluster** access (`do-sfo3-hanzo-k8s` or `do-sfo3-lux-k8s`) for deployment
3. **TLS certificates** at `/etc/tls/` for ZAP listener
4. **GHCR access** for pulling/pushing container images

## Quick reference

| Item | Value |
|------|-------|
| Hanzo domain | `api.hanzo.ai` |
| Lux domain | `api.lux.network` |
| Hanzo endpoints | 133 |
| Lux endpoints | 14 |
| Port (HTTP) | 8080 |
| Port (ZAP) | 9651 |
| Health check | `GET /__health` |
| Image (hanzo) | `ghcr.io/hanzoai/gateway:latest` |
| Image (lux) | `ghcr.io/hanzoai/gateway:lux-latest` |
| Replicas | 2 per cluster |
| Hanzo namespace | `hanzo` |
| Lux namespace | `lux-gateway` |
| K8s context (hanzo) | `do-sfo3-hanzo-k8s` |
| K8s context (lux) | `do-sfo3-lux-k8s` |
| Global rate limit (hanzo) | 5,000 req/s |
| Global rate limit (lux) | 1,000 req/s |
| Per-IP rate limit | 100 req/s |
| Auth JWKS | `https://hanzo.id/.well-known/jwks` |
| CI/CD | GitHub Actions (`deploy.yml`) |
| Repo | `github.com/hanzoai/gateway` |

## One-file quickstart

### Call the API (OpenAI-compatible)

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer hk_live_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen4-pro",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Build and run locally

```bash
cd ~/work/hanzo/gateway

# Build
make build

# Run with hanzo config
./gateway run -c configs/hanzo/gateway.json

# Run with lux config
./gateway run -c configs/lux/gateway.json

# Validate all configs
make validate

# Run tests
make test
```

### Docker Compose

```yaml
# compose.yml
services:
  gateway:
    image: ghcr.io/hanzoai/gateway:latest
    ports:
      - "8080:8080"
      - "9651:9651"
    volumes:
      - ./configs/hanzo/gateway.json:/etc/gateway/gateway.json:ro
    environment:
      AUTH_ENABLED: "true"
      AUTH_JWKS_URL: "https://hanzo.id/.well-known/jwks"
      ZAP_LISTENER_ENABLED: "false"
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8080/__health"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped
```

### Deploy

```bash
# Deploy to hanzo cluster
make deploy-hanzo

# Deploy to lux cluster
make deploy-lux

# Deploy to both
make deploy

# Check status
make status

# Tail logs
make logs-hanzo
make logs-lux
```

## Core Concepts

### Architecture

```
                    Internet
                       |
              +--------+--------+
              |                 |
     Cloudflare (hanzo)   DO LB (lux)
              |                 |
     +--------+---------+  +---+----+
     | Hanzo Ingress    |  | Lux LB |
     | (L7 TLS/routing) |  |        |
     +--------+---------+  +---+----+
              |                 |
     +--------+---------+  +---+--------+
     | Hanzo Gateway    |  | Lux Gateway |
     | 133 endpoints    |  | 14 endpoints|
     +---+----+----+----+  +---+---+----+
         |    |    |            |   |
      Cloud  IAM  Commerce   Luxd  Luxd
      API         API       (main) (test)
```

### Middleware Chain

Three gin middlewares run in order before any request reaches a backend:

1. **Auth middleware** (`auth_middleware.go`): Validates JWT via JWKS, checks billing via Commerce API (fail-open), injects `X-Hanzo-Org-Id`, `X-Hanzo-User-Id`, `X-Hanzo-User-Email` headers. API keys (`hk-*`, `sk-*`, `fw_*`, `hz_*`, `pk-*`) pass through without JWT validation.

2. **Widget security** (`widget_security.go`): Per-IP rate limiting (10 req/min default) and origin validation for widget keys (`hz_*`). Prevents abuse of public-facing widget credentials.

3. **Host proxy** (`router_engine.go`): Routes 40+ Hanzo domains to their backend K8s services via `httputil.ReverseProxy`. Supports WebSocket upgrades natively. Path-based routing for multi-backend hosts (e.g., `platform.hanzo.ai` routes `/api` to platform API, `/` to paas-ui).

### Request Flow for LLM Endpoints

```
Client --> api.hanzo.ai/v1/chat/completions
  |
  Auth middleware (JWT validate or API key passthrough)
  |
  Widget security (skip for non-hz_ tokens)
  |
  Host proxy (api.hanzo.ai matched, /v1/chat → cloud-api rewrite)
  |
  cloud-api.hanzo.svc.cluster.local:8000/api/chat/completions
  |
  Cloud API resolves model → provider (Fireworks, Together, etc.)
  |
  Streaming response back through gateway (no buffering)
```

For `api.hanzo.ai`, AI endpoints (`/v1/chat`, `/v1/completions`, `/v1/models`, etc.) are routed directly to cloud-api via reverse proxy with `/v1/` rewritten to `/api/`. Non-AI paths (`/billing/*`, `/auth/*`, `/analytics/*`, etc.) fall through to KrakenD endpoint handlers defined in `gateway.json`.

### Host-Based Routing

The gateway routes 40+ exact hostnames to backend services defined in `router_engine.go`. Examples:

| Host | Backend |
|------|---------|
| `hanzo.id` | `hanzo-login.hanzo.svc:80` |
| `chat.hanzo.ai` | `chat.hanzo.svc:80` |
| `platform.hanzo.ai` | Multi-path: `/api` to platform:4000, `/` to paas-ui:80 |
| `kms.hanzo.ai` | `kms.hanzo.svc:80` |
| `console.hanzo.ai` | `console.hanzo.svc:80` |
| `insights.hanzo.ai` | Multi-path: `/capture` to insights-capture:3000, `/` to insights-web:80 |
| `zen.hanzo.ai` | Multi-path: `/v1` to zen-gateway:4100, `/` to zen-landing:80 |

Host redirects: `app.hanzo.ai` 301-redirects to `https://hanzo.app`.

Subdomain patterns: Hosts containing `lux-test` or `lux-dev` route to testnet/devnet luxd nodes.

### Configuration Structure

All routing is defined in JSON config files (`configs/hanzo/gateway.json`, `configs/lux/gateway.json`). Each follows KrakenD v3 schema:

```json
{
  "version": 3,
  "name": "Hanzo API Gateway",
  "port": 8080,
  "timeout": "120s",
  "extra_config": {
    "router": { "return_error_msg": true },
    "qos/ratelimit/router": {
      "max_rate": 5000,
      "client_max_rate": 100,
      "strategy": "ip"
    },
    "telemetry/logging": {
      "level": "INFO",
      "prefix": "[GATEWAY]",
      "stdout": true
    }
  },
  "endpoints": [
    {
      "endpoint": "/v1/chat/completions",
      "method": "POST",
      "input_headers": ["*"],
      "output_encoding": "no-op",
      "backend": [{
        "url_pattern": "/api/chat/completions",
        "host": ["http://cloud-api.hanzo.svc.cluster.local:8000"],
        "encoding": "no-op"
      }]
    }
  ]
}
```

### ZAP Binary Transport

The gateway supports ZAP (`luxfi/zap`) as an alternative to HTTP for backend communication. ZAP is a binary protocol with mDNS service discovery.

- **ZAP listener** (`zap_listener.go`): TLS 1.3+PQ (X25519MLKEM768) listener on port 9651 for external clients. Proxies TLS connections to the internal ZAP node.
- **ZAP backend** (`zap_backend.go`): Backends with `github.com/hanzoai/gateway/zap` in their `extra_config` use ZAP binary transport instead of HTTP. Node pool with connection caching.

### Environment Variables

```bash
# Auth
AUTH_ENABLED=true              # Set false to disable all auth
AUTH_JWKS_URL=https://hanzo.id/.well-known/jwks
AUTH_ISSUER=https://hanzo.id
AUTH_PUBLIC_PATHS=/__health,/__stats,/.well-known/,/favicon
AUTH_PUBLIC_HOSTS=hanzo.id,lux.id,pars.id,id.bootno.de,id.zoo.network,iam.hanzo.ai
AUTH_REQUIRE=false             # Set true to reject unauthenticated requests

# Billing
AUTH_BILLING_URL=http://commerce.hanzo.svc.cluster.local:8001
COMMERCE_SERVICE_TOKEN=...     # S2S token for Commerce API
BILLING_ENABLED=true           # Set false to skip billing checks

# ZAP
ZAP_LISTENER_ENABLED=true
ZAP_LISTENER_PORT=9651
ZAP_TLS_CERT=/etc/tls/tls.crt
ZAP_TLS_KEY=/etc/tls/tls.key
ZAP_INTERNAL_ADDR=127.0.0.1:9652
```

### Adding a New Route

1. Edit `configs/hanzo/gateway.json` (or `configs/lux/gateway.json`)
2. Add a new entry to the `endpoints` array:
   ```json
   {
     "endpoint": "/v1/my-service/{id}",
     "method": "GET",
     "input_headers": ["*"],
     "output_encoding": "no-op",
     "backend": [{
       "url_pattern": "/api/my-service/{id}",
       "host": ["http://my-service.hanzo.svc.cluster.local:8000"],
       "encoding": "no-op"
     }]
   }
   ```
3. Validate: `make validate`
4. Deploy: `make deploy-hanzo`

### Adding a New Host Route

Edit `router_engine.go` and add to the `hostRoutes` map:

```go
"myapp.hanzo.ai": {{prefix: "/", target: mustURL("http://myapp.hanzo.svc.cluster.local:80")}},
```

For multi-path routing:

```go
"myapp.hanzo.ai": {
    {prefix: "/api", target: mustURL("http://myapp-api.hanzo.svc.cluster.local:8000")},
    {prefix: "/", target: mustURL("http://myapp-ui.hanzo.svc.cluster.local:80")},
},
```

Rebuild and deploy: `make build && make deploy-hanzo`.

### Repository Structure

```
configs/
  hanzo/
    gateway.json        # Hanzo API Gateway config (133 endpoints, 178KB)
    ingress.json        # Hanzo Ingress sidecar config
  lux/
    gateway.json        # Lux API Gateway config (14 endpoints)
k8s/
  hanzo/                # K8s manifests for hanzo-k8s cluster
    deployment.yaml     # 2 replicas, ports 8080+9651
    service.yaml        # ClusterIP
    ingress.yaml        # Ingress resource for api.hanzo.ai
  hanzo-ingress/        # K8s manifests for ingress sidecar
  lux/                  # K8s manifests for lux-k8s cluster
    deployment.yaml     # 2 replicas
    service.yaml        # LoadBalancer
cmd/
  gateway/              # Gateway binary entry point (main.go)
  gateway-integration/  # Integration test binary
  ingress/              # Ingress sidecar binary
tests/                  # Integration tests

# Core Go files (package gateway)
auth_middleware.go      # JWT validation, billing checks, header injection
auth_middleware_test.go # Auth middleware tests
widget_security.go      # Widget key rate limiting + origin validation
widget_security_test.go # Widget security tests
router_engine.go        # Host-based reverse proxy, gin engine setup
executor.go             # KrakenD executor (wires all components)
backend_factory.go      # Backend factory with CEL, Lua, metrics
handler_factory.go      # Handler factory with JOSE, rate limiting
proxy_factory.go        # Proxy factory with CEL, Lua, rate limiting
plugin.go               # Plugin loader (server, client, modifier)
encoding.go             # Custom encoding registration
zap_backend.go          # ZAP binary protocol backend transport
zap_listener.go         # TLS 1.3+PQ ZAP listener for external clients
sd.go                   # Service discovery registration
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 429 Too Many Requests | Global or per-IP rate limit exceeded | Check `qos/ratelimit/router` in gateway.json; increase limits or add per-endpoint overrides |
| 401 Unauthorized (JWT) | Invalid or expired JWT token | Verify token against `https://hanzo.id/.well-known/jwks`; check `AUTH_ISSUER` matches |
| 402 Payment Required | User has zero or negative Commerce balance | Add funds; or disable billing with `BILLING_ENABLED=false` |
| 403 Forbidden (widget) | Widget key (`hz_*`) used from unauthorized origin | Add origin to `AllowedOrigins` in `widget_security.go`; or use `hk-*` API key instead |
| API key rejected | Key prefix not recognized | Recognized prefixes: `hk-`, `sk-`, `fw_`, `hz_`, `pk-` |
| Route not found (404) | Endpoint not in gateway.json or host not in hostRoutes | Add endpoint to config or host to `router_engine.go` |
| Config validation fails | JSON syntax error or schema violation | Run `make validate`; check against KrakenD v2.12 schema |
| WebSocket not connecting | Host not in `hostRoutes` | Add host entry; `httputil.ReverseProxy` handles WS natively |
| ZAP listener fails | Missing TLS cert or wrong port | Check `ZAP_TLS_CERT`/`ZAP_TLS_KEY` paths; verify port 9651 is free |
| Billing check blocks users | Commerce API unreachable | Billing is fail-open by design; check `AUTH_BILLING_URL` connectivity |

## Related Skills

- `hanzo/hanzo-llm-gateway.md` - LLM proxy (litellm) that cloud-api uses for provider routing
- `hanzo/hanzo-cloud.md` - Cloud API backend that handles model routing and provider selection
- `hanzo/hanzo-id.md` - IAM service (Casdoor) that issues JWTs validated by gateway
- `hanzo/hanzo-commerce.md` - Commerce API used for billing balance checks
- `hanzo/hanzo-universe.md` - Production K8s infrastructure where gateway is deployed
- `hanzo/hanzo-web3-gateway.md` - Web3/blockchain API gateway (separate from this API gateway)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: gateway, api, routing, krakend, rate-limiting, authentication
**Prerequisites**: Go, Kubernetes, JSON configuration, OAuth2/JWT basics
