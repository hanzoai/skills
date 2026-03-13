# Hanzo Commerce - Multi-Tenant E-Commerce and Billing Platform

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-id.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-console.md`

## Overview

Hanzo Commerce is the **core product** of Hanzo AI -- a multi-tenant e-commerce platform that powers billing, payments, subscriptions, storefronts, and API usage metering for the entire Hanzo ecosystem and external customers. It is a standalone Go binary (Cobra CLI + Gin HTTP) with 100+ data models, 4 database backends, 6 payment providers, 25+ third-party integrations, and a complete OpenAPI 3.0 specification.

**Live at**: https://commerce.hanzo.ai | **API**: https://api.hanzo.ai/v1 | **Version**: v1.36.5

### Why Hanzo Commerce?

- **Full e-commerce platform**: Products, variants, collections, carts, orders, checkout, fulfillment, shipping, tax, inventory, returns, disputes
- **Multi-tenant**: Namespace-isolated per organization with per-org databases
- **4 database backends**: SQLite (embedded, dev), PostgreSQL (production), MongoDB/FerretDB, ClickHouse (analytics)
- **6 payment providers**: Stripe, Square, PayPal, Authorize.net, Braintree, Adyen + crypto (Bitcoin, Ethereum)
- **Usage-based billing**: Ledger, credit grants, metering, tiered pricing, Temporal workflows
- **AI-powered**: Product recommendations via vector search (sqlite-vec / pgvector / Qdrant)
- **Event-driven**: NATS/JetStream pub/sub + ClickHouse analytics pipeline
- **KMS-first secrets**: All payment credentials via KMS (kms.hanzo.ai), zero plaintext
- **IAM integrated**: Dual auth -- legacy access tokens + hanzo.id OIDC JWT
- **OpenAPI spec**: 88KB spec at `/openapi.yaml` covering all endpoints

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Go 1.26, single binary |
| HTTP | Gin web framework |
| CLI | Cobra command framework |
| Databases | SQLite + sqlite-vec, PostgreSQL + pgvector, MongoDB, ClickHouse |
| KV Cache | Redis/Valkey (hanzoai/kv-go) |
| Search | Meilisearch (hanzoai/search-go) |
| Vector | Qdrant (hanzoai/vector-go) |
| Object Storage | MinIO/S3 (hanzoai/storage-go) |
| Pub/Sub | NATS + JetStream (hanzoai/pubsub-go) |
| Tasks | Temporal SDK (billing workflows) |
| ORM | hanzoai/orm v0.3.1 (generic CRUD, mixin.Model[T]) |
| Analytics | ClickHouse (hanzoai/datastore-go) |
| Auth | hanzo.id OIDC + legacy access tokens |
| Secrets | KMS (Infisical-compatible REST) |
| Crypto | luxfi/crypto, luxfi/geth, btcsuite |
| Image | `ghcr.io/hanzoai/commerce:latest` |
| Testing | Ginkgo v2 + Gomega |

### Repo

`github.com/hanzoai/commerce` -- Go module `github.com/hanzoai/commerce`

## When to use

Use this skill when:
- Working with the commerce Go codebase or API
- Configuring payment providers (Stripe, Square, PayPal, etc.)
- Setting up billing, subscriptions, or usage metering
- Understanding the multi-tenant data model
- Deploying commerce to K8s
- Integrating commerce with other Hanzo services (IAM, KMS, Console)
- Working with the checkout flow or Stripe Connect

## Hard requirements

1. **CGO_ENABLED=1** for all builds (SQLite requires it)
2. **KMS credentials** for payment provider access in production
3. **IAM** (hanzo.id) for JWT-based authentication
4. **PVC is ReadWriteOnce** -- deployment MUST use `strategy: Recreate`, not RollingUpdate
5. **Never store payment credentials in plaintext** -- KMS is the single source of truth

## Quick reference

| Item | Value |
|------|-------|
| API | `https://api.hanzo.ai/v1` (via gateway) |
| Direct | `https://commerce.hanzo.ai` (port 8001) |
| Auth | Bearer token (org API key or IAM JWT) |
| Image | `ghcr.io/hanzoai/commerce:latest` |
| K8s | `hanzo` namespace, `commerce` deployment, Recreate strategy |
| Port | 8001 (container), 8090 (default local) |
| Data | PVC `commerce-data` (10Gi, do-block-storage) |
| KMS secret | `commerce-kms-auth` (client-id, client-secret, project-id) |
| Repo | `github.com/hanzoai/commerce` |
| OpenAPI | `/openapi.yaml` (88KB, 3.0.3) |
| CI | Test on PR + Docker build/deploy on push to main |

## Architecture

```
                    +-----------------------+
                    |    API Gateway         |
                    |    (api.hanzo.ai)      |
                    +----------+------------+
                               |
                    +----------v------------+
                    |   Commerce App         |
                    |   (Cobra + Gin)        |
                    +------------------------+
                    | HTTP Routes (/api/v1)  |
                    | Middleware Chain        |
                    | Hook System            |
                    +----+------+------+----+
                         |      |      |
              +----------+  +---+---+  +----------+
              |             |       |              |
     +--------v---+  +-----v--+  +-v--------+  +--v--------+
     | SQLite/PG  |  | Redis  |  | NATS     |  | ClickHouse|
     | (per-org)  |  | (cache)|  | (events) |  | (analytics|
     +------------+  +--------+  +----------+  +-----------+
              |
     +--------v-----------+
     | KMS    | Meilisearch|
     | (creds)| (search)   |
     +--------+------------+
```

### Multi-Tenancy Model

- **Namespace-based**: `Organization.Name` IS the namespace
- Per-org SQLite databases: `{COMMERCE_DIR}/orgs/{orgName}/data.db`
- Per-org PostgreSQL: tenant_id column isolation
- `middleware.Namespace()` sets context namespace for all downstream queries
- `"platform"` org returns empty namespace (admin bypass, documented)
- Dual auth: legacy access tokens (org-bound) + IAM JWT (OIDC/JWKS via hanzo.id)

### Hook System

Extensible lifecycle hooks (PocketBase-compatible pattern):
- `OnBootstrap` -- app initialization
- `OnServe` -- before HTTP server starts
- `OnRouteSetup` -- add custom routes
- `OnTerminate` -- graceful shutdown
- `Hook[T]`, `TaggedHook[T]`, `Resolver` -- typed hook registry

## Directory Structure

```
commerce/
  commerce.go          Main App framework (Config, Bootstrap, Routes, Serve, Shutdown)
  cmd/commerce/        CLI entry point (main.go)

  api/                 HTTP Handlers (30+ resource groups)
    accesstoken/       API token management
    account/           User account (login, register, profile)
    affiliate/         Affiliate program
    auth/              OAuth2 token endpoint
    billing/           Balance, usage, deposits, credits
    cart/              Shopping cart
    cdn/               CDN/asset serving
    checkout/          Checkout sessions (Stripe Checkout)
    contributor/       Contributor management
    counter/           Atomic counters
    coupon/            Coupon codes
    customergroup/     Customer segmentation
    dashv2/            Dashboard v2 API
    data/              Generic data API
    deploy/            Deploy management
    form/              Form submissions
    fulfillment/       Order fulfillment
    inventory/         Inventory tracking
    library/           Content library
    namespace/         Namespace management (admin-only)
    notification/      Notifications
    order/             Order CRUD
    organization/      Org management
    pricing/           Price lists and rules
    promotion/         Promotions
    referrer/          Referral tracking
    region/            Geographic regions
    review/            Product reviews
    search/            Search proxy
    store/             Storefront
    subscription/      Subscriptions
    tax/               Tax calculation
    transaction/       Transaction records
    user/              User management
    xd/                Cross-domain helpers

  models/              Data Models (100+ entities)
    product/           Product, variant, collection
    order/             Order, lineitem, fulfillmentmodel
    cart/              Shopping cart
    subscription/      Subscription, subscriptionitem, subscriptionschedule
    plan/              Subscription plans
    payment/           Payments, paymentintent, paymentmethod
    invoice/           Invoices
    billing*/          billingevent, billinginvoice
    credit/            Credits, creditgrant
    coupon/            Coupons, discount, promotion, promotionrule, redemption
    user/              User accounts
    organization/      Multi-tenant orgs
    token/             API tokens, oauthtoken, publishableapikey
    wallet/            Crypto wallets
    cryptobalance/     Crypto balances
    cryptopaymentintent/ Crypto payment intents
    networktoken/      Network tokens
    tokensale/         Token sales
    affiliate/         Affiliates
    referral/          Referral programs
    inventory/         Inventory, inventorylevel, stocklocation
    shippingoption/    Shipping options, profiles, rates
    fulfillment*/      Fulfillment providers, sets
    region/            Regions, geozones, servicezones
    taxrate/           Tax rates, rules, providers, regions
    price/             Price, pricelist, priceset, pricepreference, pricerule, pricingrule
    meter/             Usage metering
    usagewatermark/    Usage watermarks
    spendalert/        Spend alerts
    balancetransaction/ Balance transactions
    dispute/           Payment disputes
    refund/            Refunds
    fee/               Fees
    payout/            Payouts
    transfer/          Transfers
    review/            Product reviews
    form/              Form submissions
    webhook/           Webhooks, webhookendpoint
    notification/      Notifications
    types/             Shared types (currency, etc.)
    mixin/             Generic Model[T] mixin with CRUD
    migrations/        Schema migrations
    fixtures/          Test fixtures

  billing/             Billing Engine
    credit/            Credit system
    engine/            Billing calculation engine
    ledger/            Double-entry ledger
    tier/              Tiered pricing
    workflows/         Temporal billing workflows

  payment/             Payment Processing
    processor/         Payment processor interface
    providers/         Provider implementations
      adyen/           Adyen
      braintree/       Braintree
      lemonsqueezy/    Lemon Squeezy
      paypal/          PayPal
      recurly/         Recurly
    router/            Payment routing logic

  thirdparty/          Third-Party Integrations (25+)
    authorizenet/      Authorize.net
    bitcoin/           Bitcoin payments
    ethereum/          Ethereum payments
    square/            Square payments
    paypal/            PayPal (legacy)
    kms/               KMS client (secret management)
    cardconnect/       CardConnect
    cloudflare/        Cloudflare CDN
    facebook/          Facebook integration
    indiegogo/         Indiegogo crowdfunding
    mailchimp/         Mailchimp email
    mandrill/          Mandrill transactional email
    mercury/           Mercury banking
    mpc/               Multi-party computation
    netlify/           Netlify deployment
    reamaze/           Reamaze support
    recaptcha/         reCAPTCHA
    sendgrid/          SendGrid email
    shipstation/       ShipStation fulfillment
    shipwire/          Shipwire fulfillment
    smtprelay/         SMTP relay
    wire/              Wire transfers
    woopra/            Woopra analytics
    bigquery/          BigQuery analytics
    paymentmethods/    Payment method registry

  db/                  Database Backends
    sqlite.go          SQLite + sqlite-vec (embedded, per-org)
    postgres.go        PostgreSQL + pgvector (production)
    mongo.go           MongoDB (alternative)
    datastore.go       ClickHouse via hanzoai/datastore-go (analytics)
    db.go              Manager, interface, pool
    query.go           Query builder (PascalCase -> camelCase auto-conversion)
    model.go           Generic model interface
    serialize.go       JSON serialization

  datastore/           Cloud Datastore abstraction layer
    datastore.go       Core datastore interface
    get.go, put.go     CRUD operations
    delete.go          Delete operations
    key.go, key/       Key generation
    query.go, query/   Query builder
    parallel/          Parallel query execution

  infra/               Infrastructure Clients
    infra.go           Manager (connect, health, close)
    kv.go              Redis/Valkey client
    search.go          Meilisearch client
    vector.go          Qdrant vector client
    storage.go         MinIO/S3 storage client
    pubsub.go          NATS/JetStream client
    tasks.go           Temporal task client
    locking.go         Distributed locking

  hooks/               Hook System
    hooks.go           Registry, Hook[T] generic
    tagged.go          TaggedHook[T]
    event.go           Event types

  events/              Event System
    publisher.go       NATS/JetStream publisher
    client.go          HTTP analytics client
    bootstrap.go       Stream bootstrap
    schema.go          Event schema

  auth/                Authentication
    auth.go            Auth helpers
    iam.go             IAM (hanzo.id) OIDC/JWKS validation
    password/          Password hashing

  middleware/           HTTP Middleware
    accesstoken.go     API token auth
    iammiddleware/     IAM JWT validation
    oauthmiddleware/   OAuth middleware
    organization.go    Org resolution
    cache.go           Cache headers
    cors.go            CORS
    error.go           Error handler
    logger.go          Request logging

  ai/                  AI Features
    recommendations.go Vector-based product recommendations

  email/               Email System
    email.go           Email sending
    emails.go          Template rendering
    subscribe.go       Mailing list
    tasks/             Async email tasks

  cron/                Background Jobs
    payout/            Payout processing
    tasks/             Scheduled tasks

  k8s/                 Kubernetes Manifests
    deployment.yaml    Deployment (Recreate strategy, 1 replica)
    service.yaml       ClusterIP service (port 8001)
    api-ingress.yaml   Ingress for commerce.hanzo.ai
    configmap.yaml     Environment config
    pvc.yaml           PersistentVolumeClaim (10Gi)
    kms-auth-secret.yaml KMS auth credentials
    frontend-admin.yaml  Admin UI deployment
    frontend-site.yaml   Marketing site deployment
    frontend-store.yaml  Storefront deployment

  config/              Environment configs (dev, staging, production, sandbox)
  templates/           Email and page templates
  test/                Ginkgo test suites
  test-integration/    Integration tests
  demo/                Demo data
  deploy/              Deployment scripts
  docs/                Documentation
  scripts/             Utility scripts
  openapi.yaml         OpenAPI 3.0.3 specification (88KB)
  bench_test.go        Benchmarks
```

## Running

```bash
# Development (SQLite, local)
go run cmd/commerce/main.go serve --dev

# Production (PostgreSQL)
SQL_URL=postgresql://user:pass@host:5432/commerce \
  KMS_ENABLED=true KMS_URL=http://kms:8080 \
  ./commerce serve 0.0.0.0:8001

# Seed an organization with plans and API tokens
./commerce seed bootnode

# Docker
docker build --platform linux/amd64 -t ghcr.io/hanzoai/commerce:latest .
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMMERCE_DIR` | `./commerce_data` | Data directory for SQLite DBs |
| `COMMERCE_SECRET` | `change-me-in-production` | Encryption secret |
| `COMMERCE_HTTP` | `127.0.0.1:8090` | Listen address |
| `COMMERCE_DEV` | `false` | Development mode |
| `SQL_URL` | - | PostgreSQL DSN (production) |
| `KV_URL` | - | Redis/Valkey URL |
| `SEARCH_URL` | - | Meilisearch URL |
| `VECTOR_URL` | - | Qdrant URL |
| `S3_URL` | - | S3/MinIO URL |
| `PUBSUB_URL` | - | NATS URL |
| `TASKS_URL` | - | Temporal URL |
| `DATASTORE_URL` | - | ClickHouse DSN |
| `DOC_URL` | - | MongoDB URL |
| `KMS_ENABLED` | `false` | Enable KMS secret management |
| `KMS_URL` | - | KMS base URL |
| `KMS_CLIENT_ID` | - | KMS Universal Auth client ID |
| `KMS_CLIENT_SECRET` | - | KMS Universal Auth client secret |
| `KMS_PROJECT_ID` | - | KMS project ID |
| `KMS_ENVIRONMENT` | `prod` | KMS environment |
| `IAM_ENABLED` | `true` | Enable IAM JWT validation |
| `IAM_ISSUER` | `https://hanzo.id` | IAM issuer URL |
| `IAM_CLIENT_ID` | - | IAM client ID |
| `IAM_CLIENT_SECRET` | - | IAM client secret |
| `ANALYTICS_ENDPOINT` | - | Analytics collector HTTP URL |
| `INSIGHTS_ENABLED` | `false` | PostHog product analytics |
| `ANALYTICS_ENABLED` | `false` | Umami-like web analytics |

## API Endpoints

### Core Commerce

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/store` | GET/POST | Storefront listing/create |
| `/api/v1/product` | CRUD | Products |
| `/api/v1/variant` | CRUD | Product variants |
| `/api/v1/collection` | CRUD | Product collections |
| `/api/v1/cart` | CRUD | Shopping carts |
| `/api/v1/order` | CRUD | Orders |
| `/api/v1/checkout` | POST | Payment processing |
| `/api/v1/checkout/sessions` | POST | Stripe Checkout sessions |
| `/api/v1/subscription` | CRUD | Subscriptions |
| `/api/v1/coupon` | CRUD | Coupon codes |
| `/api/v1/promotion` | CRUD | Promotions |
| `/api/v1/inventory` | CRUD | Inventory tracking |
| `/api/v1/fulfillment` | CRUD | Order fulfillment |
| `/api/v1/review` | CRUD | Product reviews |

### Billing

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/billing/balance` | GET | Balance by user + currency (cents) |
| `/api/v1/billing/balance/all` | GET | All currency balances |
| `/api/v1/billing/usage` | POST | Record API usage (withdraw) |
| `/api/v1/billing/deposit` | POST | Create deposit transaction |
| `/api/v1/billing/credit` | POST | Grant starter credit ($5, 30-day expiry) |
| `/api/v1/billing/zap` | POST | Clear balance |

### Analytics

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/analytics/event` | POST | Single event |
| `/api/v1/analytics/events` | POST | Batch events |
| `/api/v1/analytics/identify` | POST | User identification |
| `/api/v1/analytics/pixel.gif` | GET | Pixel tracking |
| `/api/v1/analytics/ai/message` | POST | AI message event |
| `/api/v1/analytics/ai/completion` | POST | AI completion event |

### Auth & Admin

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/auth` | POST | OAuth2 password grant |
| `/api/v1/account` | GET/POST | User account |
| `/api/v1/user` | CRUD | User management (admin) |
| `/api/v1/organization` | CRUD | Organization management |
| `/api/v1/namespace` | CRUD | Namespace management (admin-only, authed) |
| `/api/v1/accesstoken` | CRUD | API token management |
| `/health` | GET | Health check |

## KMS Secret Management

All payment credentials are stored in KMS. No fallback to org-stored fields.

**Secret path convention**:
```
/tenants/{orgName}/stripe/STRIPE_LIVE_ACCESS_TOKEN
/tenants/{orgName}/stripe/STRIPE_TEST_ACCESS_TOKEN
/tenants/{orgName}/stripe/STRIPE_PUBLISHABLE_KEY
/tenants/{orgName}/square/SQUARE_PRODUCTION_ACCESS_TOKEN
/tenants/{orgName}/square/SQUARE_PRODUCTION_LOCATION_ID
/tenants/{orgName}/square/SQUARE_PRODUCTION_APPLICATION_ID
/tenants/{orgName}/authorizenet/AUTHORIZENET_LIVE_LOGIN_ID
/tenants/{orgName}/paypal/PAYPAL_LIVE_EMAIL
```

**Hydration**: `kms.Hydrate(ctx, org)` populates all 25 provider credential fields. Called once after org resolution. Cache: 5min TTL, extends to 30min on KMS failure.

**K8s**: Single "secret zero" (`commerce-kms-auth`) holds KMS Universal Auth credentials.

## Checkout Sessions

`POST /api/v1/checkout/sessions` -- public endpoint (no token auth).

**Request**:
```json
{
  "company": "acme",
  "providerHint": "stripe",
  "currency": "usd",
  "org": "acme",
  "customer": { "email": "user@example.com" },
  "items": [{ "productId": "prod_xyz", "quantity": 1 }],
  "successUrl": "https://acme.com/success",
  "cancelUrl": "https://acme.com/cancel"
}
```

**Response**: `{ "checkoutUrl": "https://checkout.stripe.com/...", "sessionId": "cs_..." }`

Org resolved from `X-Hanzo-Org` header or request body. Per-request Stripe client (multi-tenant safe).

## Cross-Compilation

```bash
# Build for linux/amd64 from macOS (zig cross-compile for CGO)
go mod vendor
CC="zig cc -target x86_64-linux-musl" CXX="zig c++ -target x86_64-linux-musl" \
  CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
  go build -mod=vendor -ldflags="-s -w -extldflags '-static'" \
  -o commerce ./cmd/commerce/main.go

# Push to GHCR
docker build --platform linux/amd64 -t ghcr.io/hanzoai/commerce:latest .
docker push ghcr.io/hanzoai/commerce:latest
```

## CI/CD

- **Test**: On PR and push to main -- `go vet`, unit tests (billing, models, payment), Ginkgo suites
- **Deploy**: On push to main or tag -- reusable `hanzoai/.github` Docker build workflow, deploys to `hanzo` namespace

## Gotchas

| Issue | Detail |
|-------|--------|
| SQLite requires CGO | Always `CGO_ENABLED=1`, needs gcc/musl-dev |
| PVC ReadWriteOnce | Deployment MUST use `Recreate` strategy |
| Filter field names | PascalCase (Go struct) auto-converts to camelCase JSON in queries |
| Boolean omitempty | `false` with omitempty may be NULL in JSON -- COALESCE handles it |
| Healthcheck | Use `curl -f`, not `wget --spider` (Gin only handles GET) |
| Colima cross-compile | QEMU crashes HTTP/2 on ARM Mac -- use zig cc instead |
| Platform org bypass | `"platform"` org returns empty namespace (admin bypass, intentional) |
| Global entities | Organization, User, Token use `DefaultNamespace = true` by design |

## Related Skills

- `hanzo/hanzo-id.md` -- IAM and OIDC authentication (hanzo.id)
- `hanzo/hanzo-kms.md` -- Secret management (kms.hanzo.ai)
- `hanzo/hanzo-console.md` -- Admin console
- `hanzo/hanzo-platform.md` -- PaaS deployment platform
- `hanzo/hanzo-cloud.md` -- Cloud dashboard

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: e-commerce, billing, payments, subscriptions, multi-tenant, API
**Prerequisites**: Go, REST APIs, Kubernetes basics
