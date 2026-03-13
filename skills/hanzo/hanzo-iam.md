# Hanzo IAM - Identity and Access Management Server

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-platform.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-cloud.md`

## Overview

Hanzo IAM is the **server-side identity and access management service** for the Hanzo ecosystem. A Casdoor fork written in Go (Beego framework) with a React admin UI, providing OAuth 2.0/OIDC/SAML/CAS/LDAP/SCIM/WebAuthn/TOTP/RADIUS authentication and authorization. Serves SSO across hanzo.id, lux.id, zoo.id, and pars.id. Includes built-in MCP (Model Context Protocol) server for AI agent identity management.

### Why Hanzo IAM?

- **Multi-protocol**: OAuth 2.0, OIDC, SAML, CAS, LDAP, SCIM, WebAuthn, TOTP, RADIUS -- all in one binary
- **40+ social providers**: GitHub, Google, Apple, Microsoft, Discord, MetaMask, Web3Onboard, and more
- **Multi-tenant**: Multiple organizations and applications in a single deployment
- **RBAC + Casbin**: Role-based access control with fine-grained permissions via Casbin policy engine
- **MCP-native**: Built-in MCP server for AI agents to manage users, apps, and permissions
- **User balance tracking**: Credit balance system for billing integration with Hanzo Commerce/Cloud
- **Web3 auth**: MetaMask and Web3Onboard wallet login natively supported
- **WAF**: Built-in Coraza web application firewall

### Tech Stack

- **Backend**: Go 1.26, Beego v2 web framework, xorm ORM
- **Frontend (admin UI)**: React 18, Ant Design 5, craco, pnpm
- **Database**: PostgreSQL (primary), MySQL, SQLite (embedded)
- **Session store**: Redis/Valkey (auto-discovers `hanzo-kv` or `redis` in K8s)
- **Auth libraries**: casbin/casbin v2, go-webauthn, go-jose, gosaml2, go-ldap
- **Identity providers**: markbates/goth + custom implementations in `idp/` directory
- **Image**: `ghcr.io/hanzoai/iam:latest`

### OSS Base

Fork of [Casdoor](https://github.com/casdoor/casdoor) by the Casbin community. Apache-2.0 licensed.

Repo: `github.com/hanzoai/iam`

## When to use

- Deploying or managing the IAM backend server itself
- Configuring organizations, applications, or OAuth providers
- Managing users, roles, and permissions via the admin API
- Setting up LDAP/SAML/SCIM enterprise federation
- Integrating billing/credits with the IAM user balance system
- Debugging server-side auth issues (token generation, provider callbacks)
- Using MCP tools for AI agent identity operations

For **client-side OAuth flows** (login UI, token extraction, redirect handling), see `hanzo/hanzo-id.md` instead.

## Hard requirements

1. **PostgreSQL** (or MySQL/SQLite) database
2. **Redis/Valkey** for session storage (optional -- falls back to file sessions)
3. **init_data.json** for bootstrap organizations, applications, and admin user
4. **TLS termination** via Caddy, ingress, or reverse proxy for production

## Quick reference

| Item | Value |
|------|-------|
| Admin UI | `https://iam.hanzo.ai` (internal) |
| Login UI | `https://hanzo.id` (separate repo: hanzoai/id) |
| API port | 8000 |
| LDAP port | 389 (LDAPS: 636) |
| RADIUS port | 1812 |
| Image | `ghcr.io/hanzoai/iam:latest` |
| Go module | `github.com/hanzoai/iam` |
| Go version | 1.26 |
| Frontend | React 18 + Ant Design 5 (pnpm) |
| Config | `conf/app.conf` (Beego INI format) |
| Init data | `init_data.json` |
| Repo | `github.com/hanzoai/iam` |
| License | Apache-2.0 |

## Domains served

| Domain | Purpose |
|--------|---------|
| `hanzo.id` | Hanzo AI accounts |
| `lux.id` | Lux Network accounts |
| `zoo.id` | Zoo Labs accounts |
| `pars.id` | Pars accounts |
| `id.ad.nexus` | Ad Nexus accounts |
| `id.bootno.de` | Bootnode accounts |
| `iam.hanzo.ai` | Core IAM API |

## One-file quickstart

### Docker (quickest)

```bash
docker run -d --name hanzo-iam -p 8000:8000 ghcr.io/hanzoai/iam:latest
# Open http://localhost:8000 for admin UI
```

### Docker Compose (full stack)

```yaml
# compose.yml
services:
  iam:
    image: ghcr.io/hanzoai/iam:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://hanzo:hanzo@postgres:5432/iam?sslmode=disable
      - REDIS_URL=redis://redis:6379
      - IAM_ORIGIN=https://hanzo.id
      - ENABLE_MULTI_TENANT=true
      - ALLOWED_ORIGINS=hanzo.id,zoo.id,lux.id,pars.id,iam.hanzo.ai
    volumes:
      - ./init_data.json:/app/init_data.json:ro
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: ghcr.io/hanzoai/sql:latest
    environment:
      POSTGRES_USER: hanzo
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
      POSTGRES_DB: iam
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hanzo -d iam"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: ghcr.io/hanzoai/kv:latest
    command: kv-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### From source

```bash
git clone https://github.com/hanzoai/iam.git
cd iam

# Backend
go build -o server .

# Frontend
cd web && pnpm install && pnpm run build && cd ..

# Configure
cp conf/app.dev.conf conf/app.conf
# Edit conf/app.conf with your database credentials

# Run
./server
```

### Admin API examples

```bash
# Get all users in an organization
curl -s "https://iam.hanzo.ai/api/get-users?owner=hanzo" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

# Create a new application
curl -X POST "https://iam.hanzo.ai/api/add-application" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "admin",
    "name": "app-myservice",
    "organization": "hanzo",
    "clientId": "my-client-id",
    "clientSecret": "my-client-secret",
    "redirectUris": ["https://myservice.hanzo.ai/callback"],
    "expireInHours": 168,
    "grantTypes": ["authorization_code", "implicit"]
  }'

# Update user balance (billing integration)
curl -X POST "https://iam.hanzo.ai/api/update-user" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "hanzo",
    "name": "username",
    "balance": 10000
  }'
```

## Core Concepts

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Hanzo IAM Server (Go)                      │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  OAuth2/OIDC │  SAML/CAS    │  LDAP/SCIM   │  WebAuthn/MFA  │
├──────────────┴──────────────┴──────────────┴─────────────────┤
│  Beego Router → Controllers → Object (business logic)        │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  40+ IdPs    │  Casbin RBAC │  MCP Server  │  WAF (Coraza)  │
├──────────────┴──────────────┴──────────────┴─────────────────┤
│  PostgreSQL/MySQL/SQLite  │  Redis/Valkey  │  File Storage   │
└───────────────────────────┴────────────────┴─────────────────┘
         ↑                        ↑                    ↑
  ┌──────┴──────┐          ┌──────┴──────┐     ┌──────┴──────┐
  │  hanzo.id   │          │  cloud.     │     │  platform.  │
  │  (login UI) │          │  hanzo.ai   │     │  hanzo.ai   │
  └─────────────┘          └─────────────┘     └─────────────┘
```

### Directory Structure

```
iam/
├── main.go              # Entry point (Beego bootstrap, filter chain)
├── conf/                # Configuration files (INI format)
│   ├── app.conf         # Active config (gitignored in prod)
│   ├── app.dev.conf     # Docker dev (PostgreSQL)
│   ├── app.prod.conf    # Production (hanzo.id)
│   ├── app.staging.conf # Staging (stg.hanzo.id)
│   └── waf.conf         # Coraza WAF rules
├── controllers/         # HTTP handlers (Beego controllers)
│   ├── auth.go          # OAuth2 authorization flows
│   ├── token.go         # Token issuance and validation
│   ├── account.go       # User account management
│   ├── user.go          # User CRUD
│   ├── application.go   # Application management
│   ├── organization.go  # Organization management
│   ├── permission.go    # Permission management
│   ├── role.go          # Role management
│   ├── mfa.go           # Multi-factor authentication
│   ├── webauthn.go      # WebAuthn/Passkeys
│   ├── saml.go          # SAML SSO
│   ├── cas.go           # CAS protocol
│   ├── ldap.go          # LDAP management
│   ├── scim.go          # SCIM provisioning
│   ├── webhook.go       # Webhook management
│   ├── wellknown_oidc_discovery.go  # .well-known endpoints
│   └── wellknown_oauth_prm.go      # OAuth metadata
├── object/              # Core business logic (models + ORM)
│   ├── adapter.go       # Database adapter
│   ├── ormer.go         # xorm ORM setup
│   └── transaction.go   # Billing transactions
├── routers/             # Beego filters and route definitions
│   ├── router.go        # All API route registrations
│   ├── authz_filter.go  # Authorization filter
│   ├── cors_filter.go   # CORS handling
│   ├── secure_cookie_filter.go  # Secure cookie enforcement
│   ├── static_filter.go # SPA static file serving
│   └── mcp_util.go      # MCP protocol utilities
├── idp/                 # Identity provider implementations
│   ├── github.go, google.go, facebook.go, ...  # 30+ providers
│   ├── metamask.go, web3onboard.go              # Web3 providers
│   ├── goth.go          # Goth library integration (40+ providers)
│   └── provider.go      # Provider interface
├── mcp/                 # MCP (Model Context Protocol) server
│   ├── base.go          # MCP server setup and tool registry
│   ├── auth.go          # MCP auth tools
│   ├── application.go   # MCP app management tools
│   └── permission.go    # MCP permission tools
├── authz/               # Casbin authorization engine
├── ldap/                # LDAP server implementation
├── radius/              # RADIUS server implementation
├── scim/                # SCIM protocol implementation
├── captcha/             # CAPTCHA generation
├── certificate/         # X.509 certificate management
├── email/               # Email templates and sending
├── notification/        # Push notifications
├── proxy/               # HTTP client proxy
├── storage/             # File storage backends
├── web/                 # React admin UI (Ant Design)
│   ├── src/             # React source
│   └── package.json     # pnpm, React 18, Ant Design 5
├── init_data.json       # Bootstrap data (orgs, apps, users)
├── compose.yml          # Production Docker Compose
├── Dockerfile           # Multi-stage (frontend + backend + alpine)
├── Makefile             # Build, test, deploy commands
└── k8s.yaml             # Kubernetes deployment manifest
```

### Filter Chain (request lifecycle)

The `main.go` registers Beego filters in this order:

1. `SecureCookieFilter` (BeforeStatic) -- enforce secure cookies behind TLS
2. `StaticFilter` (BeforeRouter) -- serve React SPA static files
3. `AutoSigninFilter` (BeforeRouter) -- auto-sign-in from cookies
4. `CorsFilter` (BeforeRouter) -- CORS headers
5. `TimeoutFilter` (BeforeRouter) -- request timeout enforcement
6. `ApiFilter` (BeforeRouter) -- API auth and rate limiting
7. `PrometheusFilter` (BeforeRouter) -- metrics collection
8. `RecordMessage` (BeforeRouter) -- audit logging
9. `FieldValidationFilter` (BeforeRouter) -- input validation

### Redis Auto-Discovery

The server auto-discovers Redis in Kubernetes by trying DNS lookups for `hanzo-kv` and `redis` service names. Falls back to file-based sessions if no Redis is available.

### Init Data

`init_data.json` bootstraps the database on first run:

| Entity | Defaults |
|--------|----------|
| Organization | `hanzo` (dark theme, primary color #fd4444) |
| Applications | `app-hanzo`, `app-cloud`, `app-commerce` |
| Admin user | `admin@hanzo.ai` with configurable balance |
| Certificates | RSA certs for JWT signing |

Set `initDataNewOnly = true` in conf to avoid overwriting existing data on restart.

### Billing Integration

IAM tracks user credit balances. The flow is:
1. **Commerce** processes payments and calls IAM API to add credits
2. **Cloud** consumes AI tokens and debits balance via IAM transactions
3. **IAM** is the source of truth for user credit balances

### MCP Server

The built-in MCP server (in `mcp/` directory) exposes IAM operations as MCP tools for AI agents:

- **Auth tools**: Validate tokens, get user info, check permissions
- **Application tools**: List/create/update OAuth applications
- **Permission tools**: Manage roles, permissions, and Casbin policies

### Supported Identity Providers

Native implementations in `idp/` directory:

| Category | Providers |
|----------|-----------|
| Code hosting | GitHub, GitLab, Gitee, Bitbucket |
| Social | Google, Facebook, Twitter, LinkedIn, Discord, Reddit |
| Enterprise | ADFS, Okta, Azure AD B2C, SAML (any) |
| Messaging | Telegram, Lark, DingTalk, WeChat, WeCom, Line, Slack |
| Web3 | MetaMask, Web3Onboard |
| Regional | Alipay, Baidu, Bilibili, Douyin, QQ, Weibo, Kwai |
| Via Goth | 40+ additional providers |

## Configuration

### Beego INI config (`conf/app.conf`)

```ini
appname = hanzo-iam
httpport = 8000
runmode = dev
driverName = postgres
dataSourceName = user=hanzo password=xxx host=postgres port=5432 sslmode=disable dbname=hanzo_iam
dbName = hanzo_iam
redisEndpoint = redis:6379
authState = "hanzo"
origin = https://hanzo.id
staticBaseUrl = "https://cdn.hanzo.ai"
ldapServerPort = 389
ldapsServerPort = 636
radiusServerPort = 1812
radiusDefaultOrganization = "hanzo"
initDataNewOnly = true
initDataFile = "./init_data.json"
logConfig = {"adapter":"file", "filename": "logs/hanzo-iam.log", "maxdays":99999}
quota = {"organization": -1, "user": -1, "application": -1, "provider": -1}
```

### Environment Variables

```bash
# Database
POSTGRES_USER=hanzo
POSTGRES_PASSWORD=<generate-secure-password>
POSTGRES_DB=iam

# IAM server
IAM_ORIGIN=https://iam.hanzo.ai
ENCRYPTION_KEY=<32-byte-hex-key>
ENABLE_MULTI_TENANT=true
ALLOWED_ORIGINS=hanzo.id,zoo.id,lux.id,pars.id,iam.hanzo.ai

# Per-org client secrets (override init_data.json placeholders)
HANZO_CLIENT_SECRET=<generate-secret>
ZOO_CLIENT_SECRET=<generate-secret>
LUX_CLIENT_SECRET=<generate-secret>
PARS_CLIENT_SECRET=<generate-secret>

# Email (optional)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=noreply@hanzo.ai
SMTP_PASSWORD=<email-password>
```

### MySQL vs PostgreSQL

**PostgreSQL** (recommended):
```ini
driverName = postgres
dataSourceName = user=hanzo password=xxx host=postgres port=5432 sslmode=disable dbname=hanzo_iam
```

**MySQL** (dataSourceName must NOT include database name -- it is appended from `dbName`):
```ini
driverName = mysql
dataSourceName = hanzo:pass@tcp(localhost:3306)/
dbName = hanzo_iam
```

## Makefile Commands

```bash
make dev          # Start local dev with Docker Compose (PostgreSQL)
make dev-down     # Stop local dev
make run          # Run Go server locally (go run)
make backend      # Build Go binary to bin/manager
make frontend     # Build React admin UI (pnpm)
make ut           # Run Go unit tests with coverage
make docker-build # Build Docker image
make docker-push  # Push to ghcr.io/hanzoai/iam
make deploy       # Helm deploy to K8s
make staging      # Start staging compose
make prod         # Start production compose
make build-prod   # Build and push production image
```

## API Endpoints

### OAuth2/OIDC (RFC 6749 compliant)

| Endpoint | Path | Method |
|----------|------|--------|
| Authorize | `/oauth/authorize` | GET |
| Token | `/oauth/token` | POST |
| Introspect | `/oauth/introspect` | POST |
| Revoke | `/oauth/revoke` | POST |
| UserInfo | `/oauth/userinfo` | GET |
| Device | `/oauth/device` | POST |
| Logout | `/oauth/logout` | GET |
| OIDC Discovery | `/.well-known/openid-configuration` | GET |
| OAuth Metadata | `/.well-known/oauth-authorization-server` | GET |
| JWKS | `/.well-known/jwks` | GET |

### Admin REST API (`/api/`)

| Resource | Endpoints |
|----------|-----------|
| Users | `get-users`, `get-user`, `add-user`, `update-user`, `delete-user` |
| Organizations | `get-organizations`, `get-organization`, `add-organization`, `update-organization` |
| Applications | `get-applications`, `get-application`, `add-application`, `update-application` |
| Providers | `get-providers`, `get-provider`, `add-provider`, `update-provider` |
| Roles | `get-roles`, `get-role`, `add-role`, `update-role`, `delete-role` |
| Permissions | `get-permissions`, `get-permission`, `add-permission`, `update-permission` |
| Tokens | `get-tokens`, `get-token`, `delete-token` |
| Sessions | `get-sessions`, `delete-session` |
| Records | `get-records` (audit log) |
| Webhooks | `get-webhooks`, `add-webhook`, `update-webhook` |
| Account | `get-account`, `signup`, `login`, `logout` |
| Verification | `send-verification-code`, `verify-code` |
| MFA | `mfa-setup-initiate`, `mfa-setup-verify`, `mfa-setup-enable` |

### Protocol Endpoints

| Protocol | Path |
|----------|------|
| SAML | `/api/saml/redirect`, `/api/saml/metadata` |
| CAS | `/cas/login`, `/cas/logout`, `/cas/serviceValidate`, `/cas/proxyValidate` |
| LDAP | Port 389 (LDAPS 636) |
| RADIUS | Port 1812 |
| SCIM | `/scim/v2/Users`, `/scim/v2/Groups` |
| Registry token | `/v2/token` (Docker registry auth) |

### Metrics

Prometheus metrics exposed at `/api/metrics`.

## SDK Integration

### Go SDK

```go
import "github.com/hanzoid/go-sdk/iamsdk"

iamsdk.InitConfig(
    "https://hanzo.id",
    "hanzo-app-client-id",
    "hanzo-app-client-secret",
    "cert-hanzo",
    "hanzo",
    "app-hanzo",
)

// Validate a token
claims, err := iamsdk.ParseJwtToken(token)

// Get user
user, err := iamsdk.GetUser("admin/username")
```

### JavaScript SDK

```javascript
import { SDK } from 'iam-js-sdk'

const sdk = new SDK({
  serverUrl: 'https://hanzo.id',
  clientId: 'hanzo-app-client-id',
  appName: 'app-hanzo',
  organizationName: 'hanzo',
})

// Start auth
sdk.signin_redirect()

// Parse callback
const token = sdk.exchangeForAccessToken()
```

## Production Deployment

### K8s (hanzo-k8s cluster)

IAM runs on the hanzo-k8s cluster at `24.199.76.156`:
- Namespace: `hanzo`
- Service: `iam`
- Database: `postgres.hanzo.svc` (db: `iam`)
- Secrets: via KMS (`kms.hanzo.ai`)

### CI/CD

- **Build**: Go tests with PostgreSQL, frontend build (pnpm), Go build with race detection, gofumpt lint, Cypress E2E
- **Release**: Semantic versioning, GitHub Release, Docker push to `ghcr.io/hanzoai/iam`
- **Deploy**: Push to main triggers multi-arch Docker build (amd64/arm64) and SSH deploy

### Health Check

```bash
curl -f http://localhost:8000/api/health
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Access denied to database hanzo_iamhanzo_iam" | MySQL dataSourceName includes dbname | End with `/` not `/dbname` |
| "could not open pg_filenode.map" | PostgreSQL volume corruption | `docker compose down -v && docker volume prune` |
| Tokens expire instantly | `expireInHours=0` on app | Set `expireInHours=168` in Casdoor app config |
| Code flow rejected | Empty grant_type xorm bug | Use implicit flow (`response_type=token`) |
| initData overwrites on restart | `initDataNewOnly=false` | Set `initDataNewOnly=true` in conf |
| App owner queries fail | Apps under `owner=admin` not org | WHERE `owner='admin'` not org name |
| Redis not found | No Redis endpoint configured | Auto-discovers `hanzo-kv` or `redis` DNS; set `redisEndpoint` explicitly |
| LDAP not starting | Port 389 in use | Check `ldapServerPort` in conf |
| WAF blocking requests | Coraza rules too strict | Adjust `conf/waf.conf` |

## Related Skills

- `hanzo/hanzo-id.md` - Client-side login UI and OAuth flows (Next.js)
- `hanzo/hanzo-platform.md` - PaaS (uses IAM for auth)
- `hanzo/hanzo-kms.md` - Secret management (also uses IAM)
- `hanzo/hanzo-cloud.md` - Cloud dashboard (uses IAM for auth + billing)
- `hanzo/hanzo-console.md` - Admin console (uses IAM for auth)
- `hanzo/hanzo-commerce.md` - Commerce (writes user balances to IAM)
- `hanzo/hanzo-universe.md` - K8s infrastructure manifests

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: iam, casdoor, oauth2, oidc, saml, ldap, scim, webauthn, identity, authentication, authorization, sso
**Prerequisites**: Go, Docker, PostgreSQL, OAuth2/OIDC concepts
