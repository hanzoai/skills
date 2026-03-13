# Hanzo Analytics - Privacy-Focused Web Analytics

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-database.md`

## Overview

Hanzo Analytics is a **privacy-focused web analytics platform** -- an alternative to Google Analytics. Next.js 15 dashboard (port 3000) with Prisma ORM, PostgreSQL/ClickHouse storage, and a Go-based high-performance event collector. Includes anti-ad-blocker technology (encrypted tracker + WASM decryption). Live at `analytics.hanzo.ai`.

### Why Hanzo Analytics?

- **Privacy-first**: No cookies required, GDPR/HIPAA compliant by design
- **Anti-ad-blocker**: Encrypted tracker script + WASM decryptor defeats blockers
- **Dual database**: PostgreSQL for metadata, optional ClickHouse for event scale
- **Real-time**: Live visitor tracking, real-time dashboards
- **Multi-tenant**: Teams, users, roles, shared dashboards
- **Revenue tracking**: Built-in e-commerce revenue attribution
- **Link shortener + tracking pixels**: URL shortening and pixel tracking built in

### Tech Stack

- **Dashboard**: Next.js 15 + React 19 + Chart.js + react-intl (i18n)
- **ORM**: Prisma 6 with PostgreSQL adapter
- **State**: Zustand + TanStack React Query
- **Collector**: Go (Gin) + ClickHouse for high-throughput event ingestion
- **Tracker**: Rollup-bundled JS, encrypted delivery via WASM
- **Validation**: Zod
- **Auth**: JWT (jsonwebtoken) + bcryptjs password hashing
- **Linting**: Biome
- **Testing**: Jest + Cypress (e2e)
- **Images**: `ghcr.io/hanzoai/analytics:latest`

### OSS Base

Repo: `hanzoai/analytics` (Umami fork, v3.0.3).

## When to use

- Tracking page views, events, and sessions across websites
- Privacy-compliant analytics (GDPR, HIPAA) without third-party cookies
- Real-time visitor dashboards and traffic reports
- E-commerce revenue attribution and funnel analysis
- URL shortening with click tracking
- Tracking pixel deployment
- Shared/public analytics dashboards (share links)

## Hard requirements

1. **PostgreSQL** (minimum v12.14) for metadata and event storage
2. **Node.js 18.18+** for the dashboard application
3. **Go 1.26+** for the collector service (optional, for ClickHouse pipeline)
4. **ClickHouse** (optional) for high-volume event analytics

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://analytics.hanzo.ai` |
| Demo | `https://analytics.hanzo.ai/share/LGazGOecbDtaIwDr/hanzo.ai` |
| Dev port | `3001` (Next.js dev) / `3000` (production) |
| Image | `ghcr.io/hanzoai/analytics:latest` |
| Package | `@hanzo/analytics` (v3.0.3) |
| Database | PostgreSQL (required), ClickHouse (optional) |
| ORM | Prisma 6 |
| Collector | Go + Gin + ClickHouse (`collector/`) |
| Health check | `/api/heartbeat` |
| Repo | `github.com/hanzoai/analytics` |
| License | MIT |

## One-file quickstart

### Docker Compose

```yaml
# compose.yml
services:
  analytics:
    image: ghcr.io/hanzoai/analytics:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://analytics:analytics@db:5432/analytics
      APP_SECRET: replace-me-with-a-random-string
    depends_on:
      db:
        condition: service_healthy
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "curl http://localhost:3000/api/heartbeat"]
      interval: 5s
      timeout: 5s
      retries: 5

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: analytics
      POSTGRES_USER: analytics
      POSTGRES_PASSWORD: analytics
    volumes:
      - analytics-db-data:/var/lib/postgresql/data
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  analytics-db-data:
```

### From source

```bash
git clone https://github.com/hanzoai/analytics.git
cd analytics
cp .env.example .env  # Set DATABASE_URL
pnpm install
pnpm build            # Creates tables on first run, default: admin/analytics
pnpm start            # http://localhost:3000
```

### Add tracking to a website

```html
<script defer src="https://analytics.hanzo.ai/script.js"
        data-website-id="YOUR-WEBSITE-ID"></script>
```

## Core Concepts

### Architecture

```
                   ┌──────────────────────┐
                   │   analytics.hanzo.ai │
                   │   (Next.js 15)       │
                   │                      │
                   │  ┌────────────────┐  │
 Browser ─────────▶│  │ /api/send      │  │──── PostgreSQL
 (tracker.js)      │  │ /api/batch     │  │     (sessions, events,
                   │  │ /api/realtime  │  │      users, teams, etc.)
                   │  │ /api/websites  │  │
                   │  │ /api/reports   │  │
                   │  │ /api/auth      │  │
                   │  └────────────────┘  │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │   collector (Go)     │
 High-volume ─────▶│   Gin HTTP server    │──── ClickHouse
 events            │   event.go           │     (event_data,
                   │   forward/           │      session_data)
                   │   writer/            │
                   └──────────────────────┘
```

### Data Models (Prisma)

- **User**: Admin users with bcrypt-hashed passwords, roles
- **Website**: Tracked sites with optional share links
- **Session**: Browser/OS/device/geo per visitor
- **WebsiteEvent**: Page views, custom events with UTM params, click IDs
- **EventData / SessionData**: Key-value custom properties
- **Team / TeamUser**: Multi-tenant team management
- **Report / Segment**: Saved analytics queries
- **Revenue**: E-commerce revenue tracking per event
- **Link**: URL shortener with slug-based routing
- **Pixel**: Tracking pixel management

### Anti-Ad-Blocker Pipeline

1. Tracker JS encrypted into `tracker.bin` (binary blob)
2. `decrypt.wasm` (WebAssembly) decrypts in-browser memory
3. Lightweight loader fetches both, decrypts, executes
4. API calls use obfuscated function names (not `track()`)
5. Payloads encrypted before transmission

### API Routes

| Route | Purpose |
|-------|---------|
| `/api/send` | Receive tracking events |
| `/api/batch` | Batch event ingestion |
| `/api/realtime` | Real-time visitor data |
| `/api/websites` | Website CRUD |
| `/api/reports` | Report CRUD |
| `/api/teams` | Team management |
| `/api/users` | User management |
| `/api/auth` | Authentication |
| `/api/links` | URL shortener |
| `/api/pixels` | Tracking pixels |
| `/api/share` | Public dashboard sharing |
| `/api/heartbeat` | Health check |

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/analytics   # Required
APP_SECRET=random-secret-string                                # Required
CLICKHOUSE_URL=http://localhost:8123                            # Optional
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails on first run | Missing DATABASE_URL | Set `DATABASE_URL` in `.env` before `pnpm build` |
| Default login not working | Changed by script | Default is `admin` / `analytics`, use `npm run change-password` |
| Ad blocker blocks events | Tracker detected | Use encrypted tracker pipeline (enabled by default) |
| High event volume slow | PostgreSQL bottleneck | Deploy Go collector with ClickHouse backend |
| Geo data missing | No GeoIP database | Run `pnpm build-geo` to download MaxMind DB |
| Docker health check fails | Port not ready | Increase `retries` or wait for DB initialization |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS for deploying analytics
- `hanzo/hanzo-cloud.md` - Cloud dashboard
- `hanzo/hanzo-database.md` - PostgreSQL infrastructure
- `hanzo/hanzo-kms.md` - Secret management for APP_SECRET

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: analytics, tracking, privacy, web-analytics, events
**Prerequisites**: PostgreSQL, Node.js 18.18+, Docker (optional)
