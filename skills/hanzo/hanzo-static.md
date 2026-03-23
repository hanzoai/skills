# Hanzo Static - Static File Server Plugin for Traefik

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-ingress.md`, `hanzo/hanzo-deploy.md`

## Overview

Hanzo Static is a **Traefik plugin** that serves static files (SPAs, assets) directly from the ingress controller. It replaces the need for nginx or caddy sidecars for static site serving. Configured as middleware on Traefik IngressRoutes, it serves files from a mounted volume with proper cache headers, SPA fallback routing, and gzip support.

## When to use

- Serving static-exported Next.js apps (billing, team front, etc.)
- SPA serving with client-side routing (fallback to index.html)
- Serving pre-built frontend assets without a Node.js runtime
- Replacing nginx/caddy containers for static content

## Hard requirements

1. **No nginx, no caddy** -- use the Traefik static plugin or Hanzo Ingress middleware
2. **Gzip awareness**: If serving pre-gzipped files (`.html.gz`), both `.html` and `.html.gz` must be updated together
3. **Cache headers**: Set proper `Cache-Control` for immutable assets vs HTML

## Quick reference

| Item | Value |
|------|-------|
| Plugin type | Traefik middleware plugin |
| Ingress controller | Hanzo Ingress (Traefik fork) |
| IngressClass | `hanzo` |
| SPA fallback | Configurable (default: `index.html`) |
| Gzip | Serves `.gz` files if `Accept-Encoding: gzip` |
| Cache (assets) | `Cache-Control: public, max-age=31536000, immutable` |
| Cache (HTML) | `Cache-Control: no-cache` |

## Architecture

```
Client Request
      |
Hanzo Ingress (Traefik)
      |
Static Middleware Plugin
      |
  +---+---+
  |       |
Asset   SPA
match   fallback
  |       |
Serve   index.html
file    (+ gzip)
```

## Configuration

### Traefik middleware (IngressRoute)

```yaml
apiVersion: traefik.io/v1alpha1
kind: Middleware
metadata:
  name: static-serve
  namespace: hanzo
spec:
  plugin:
    static:
      root: /srv/www
      index: index.html
      spa: true
      gzip: true
      headers:
        Cache-Control: "public, max-age=31536000, immutable"
      htmlHeaders:
        Cache-Control: "no-cache"
```

### IngressRoute with static middleware

```yaml
apiVersion: traefik.io/v1alpha1
kind: IngressRoute
metadata:
  name: billing-static
  namespace: hanzo
spec:
  entryPoints:
    - websecure
  routes:
    - match: Host(`billing.hanzo.ai`)
      kind: Rule
      middlewares:
        - name: static-serve
      services:
        - name: billing
          port: 80
  tls:
    certResolver: letsencrypt
```

### Static site deployment pattern

For static-exported apps (Next.js with `output: 'export'`), the typical pattern is:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: billing
  namespace: hanzo
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: billing
          image: ghcr.io/hanzoai/billing:latest
          # Image contains built static files + minimal HTTP server
          ports:
            - containerPort: 80
          livenessProbe:
            httpGet:
              path: /health
              port: 80
```

The container image typically uses a minimal Alpine-based HTTP server or the Traefik static plugin directly.

## SPA routing

For single-page applications, the plugin intercepts 404s and serves `index.html` instead, allowing client-side routing to work:

```
GET /dashboard/settings
  -> No file at /dashboard/settings
  -> SPA mode: serve /index.html
  -> Client-side router handles /dashboard/settings
```

## Gzip handling

The plugin serves pre-gzipped files when available:

1. Client sends `Accept-Encoding: gzip`
2. Plugin checks for `<file>.gz` alongside `<file>`
3. If `.gz` exists, serves it with `Content-Encoding: gzip`
4. If not, serves the original file

**Critical gotcha**: When updating static files, you MUST update both the original and `.gz` versions. If you only update `index.html` but not `index.html.gz`, clients will receive the stale gzipped version.

## Common patterns

### Next.js static export

```dockerfile
# Dockerfile for static site
FROM node:22-alpine AS builder
WORKDIR /app
COPY . .
RUN pnpm install && pnpm build

FROM ghcr.io/hanzoai/static:latest
COPY --from=builder /app/out /srv/www
```

### Security headers

The static plugin adds security headers by default:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Old content after deploy | Stale `.gz` file | Update both `.html` and `.html.gz` |
| 404 on client routes | SPA mode not enabled | Set `spa: true` in middleware config |
| Broken assets | Wrong cache headers | Check `Cache-Control` middleware config |
| CORS errors | Missing headers | Add CORS middleware before static middleware |

## Related Skills

- `hanzo/hanzo-ingress.md` -- Traefik ingress controller
- `hanzo/hanzo-deploy.md` -- Deployment workflow
- `hanzo/hanzo-billing.md` -- Example static site deployment

---

**Last Updated**: 2026-03-23
**Category**: Hanzo Ecosystem
**Related**: static, traefik, spa, nginx-replacement, cache
**Prerequisites**: Traefik, K8s IngressRoute CRDs
