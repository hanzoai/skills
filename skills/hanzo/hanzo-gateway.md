# Hanzo Gateway

API gateway for all Hanzo services. KrakenD-based with Go auth middleware.

## URLs
- **Production**: api.hanzo.ai
- **Health**: api.hanzo.ai/__health (KrakenD built-in, via gateway.json)
- **Models**: api.hanzo.ai/v1/models (41 models, 14 Zen)
- **IAM Client ID**: hanzo-gateway
- **Docker Image**: ghcr.io/hanzoai/gateway:main

## Tech Stack
- **Framework**: KrakenD (Go) with Lura router (luraproject/lura/v2)
- **HTTP Engine**: Gin
- **Auth**: Custom Go middleware (auth_middleware.go)
- **Config**: configs/hanzo/gateway.json (KrakenD endpoints)

## Key Files
- `auth_middleware.go` — JWT validation, billing check, X-IAM-* header injection
- `auth_middleware_security_test.go` — 12+ security regression tests
- `router_engine.go` — Gin engine setup, health route dedup, host-based routing
- `configs/hanzo/gateway.json` — KrakenD endpoint definitions (146 endpoints)

## Security (2026-03-25)
- `stripIdentityHeaders()` — deletes ALL X-IAM-* headers before processing (prevents injection)
- JWT issuer always validated (no empty-iss skip)
- JWT audience validation (configurable AUTH_AUDIENCE)
- Error messages sanitized ("Invalid token" — no JWT internals leaked)
- 7 regression tests for header injection vectors

## Headers (2026-03-28)
- Gateway sets: X-User-Id, X-Org-Id, X-User-Email (unified, no IAM prefix)
- KrakenD propagates same headers via propagate_claims
- JWT claims: extracts `sub` (user ID), `owner` (org), `email`, with fallback to `preferred_username`/`name` when `sub` is empty (Casdoor compatibility)
- Billing check: GET /api/v1/billing/balance on commerce (fail-open)

## Health Endpoint (2026-03-27)
- `/__health` is defined in gateway.json (KrakenD backend config)
- `NewEngine()` injects `disable_health: true` into `luragin.Namespace` extra_config
- This prevents Lura from registering a duplicate `/__health` route (which panics Gin)
- Do NOT manually register `/__health` in router_engine.go — KrakenD handles it

## Telemetry (2026-03-25)
- OpenTelemetry exporter to otel-collector.hanzo.svc:4317
- Service name: gateway, sample rate: 0.1

## Build & Deploy
- Go 1.26.1, Dockerfile uses golang:1.26.1-alpine
- K8s: deployment/gateway, 2 replicas
- CI: shared docker-build.yml workflow (multi-arch)
- Makefile: `make build`, `make test`

## Upstream
- Custom (not a fork), MIT license
