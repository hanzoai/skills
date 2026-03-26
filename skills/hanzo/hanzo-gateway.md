# Hanzo Gateway

API gateway for all Hanzo services. KrakenD-based with Go auth middleware.

## URLs
- **Production**: api.hanzo.ai
- **Health**: api.hanzo.ai/health
- **Models**: api.hanzo.ai/v1/models (41 models, 14 Zen)
- **IAM Client ID**: hanzo-gateway
- **Docker Image**: ghcr.io/hanzoai/gateway:main

## Tech Stack
- **Framework**: KrakenD (Go)
- **Auth**: Custom Go middleware (auth_middleware.go)
- **Config**: configs/hanzo/gateway.json (KrakenD endpoints)

## Key Files
- `auth_middleware.go` — JWT validation, billing check, X-IAM-* header injection
- `auth_middleware_security_test.go` — 12+ security regression tests
- `router_engine.go` — Gin engine setup
- `configs/hanzo/gateway.json` — KrakenD endpoint definitions (146 endpoints)

## Security (2026-03-25)
- `stripIdentityHeaders()` — deletes ALL X-IAM-* headers before processing (prevents injection)
- JWT issuer always validated (no empty-iss skip)
- JWT audience validation (configurable AUTH_AUDIENCE)
- Error messages sanitized ("Invalid token" — no JWT internals leaked)
- 7 regression tests for header injection vectors

## Headers
- Gateway sets: X-IAM-Org-Id, X-IAM-User-Id, X-IAM-User-Email
- KrakenD propagates same headers via propagate_claims
- Billing check: GET /api/v1/billing/balance on commerce (fail-open)

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
