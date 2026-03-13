# Hanzo Dataroom - Document Sharing & Analytics

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-iam.md`

## Overview

Hanzo Dataroom is the **open-source DocSend alternative** -- a document sharing platform with built-in analytics, custom domains, and secure link-based access. Papermark fork rebranded for the Hanzo ecosystem. Built with Next.js 15, Prisma ORM on PostgreSQL, and Tinybird for analytics. Supports S3/Vercel Blob storage, Stripe billing, SAML SSO (BoxyHQ), and Hanzo IAM OAuth. Enterprise features live in `ee/` under a separate license.

### What it actually is

- Next.js 15 full-stack app (pages router + app router)
- Prisma ORM with PostgreSQL (multi-schema: `prisma/schema/`)
- Document storage: AWS S3 or Vercel Blob (configurable)
- Analytics: Tinybird (page-by-page view tracking)
- Auth: NextAuth.js with Hanzo IAM OAuth, passkeys (Hanko)
- Email: Resend (transactional) + react-email templates
- Payments: Stripe (in `ee/stripe/`)
- Background jobs: Trigger.dev v3
- Rich text: Tiptap editor
- PDF rendering: react-pdf, mupdf, pdf-lib
- AI features: OpenAI + Google Vertex AI SDKs for document AI
- Notion import: react-notion-x
- Enterprise: SAML SSO, custom branding, datarooms (in `ee/`)
- Deployed at `dataroom.hanzo.ai`

### Upstream

Fork of [Papermark](https://github.com/mfts/papermark). Description still references Papermark. License: AGPLv3 (core) + enterprise license (`ee/LICENSE.md`).

## When to use

- Sharing documents with per-page view analytics
- Building a branded document portal with custom domains
- Investor datarooms with NDA/email verification gates
- Self-hosting a DocSend alternative

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/dataroom` |
| Package | `hanzo-dataroom` (private) |
| Version | 0.1.0 |
| Branch | `main` |
| Stack | Next.js 15, React 18, TypeScript, Tailwind CSS |
| ORM | Prisma 6.5 on PostgreSQL |
| Auth | NextAuth.js + Hanzo IAM OAuth + passkeys (Hanko) |
| Analytics | Tinybird |
| Storage | AWS S3 or Vercel Blob |
| Payments | Stripe |
| Email | Resend + react-email |
| Jobs | Trigger.dev v3 |
| AI | OpenAI, Google Vertex AI |
| License | AGPLv3 (core), enterprise license (ee/) |
| Live | `dataroom.hanzo.ai` |
| Node | >= 22 |

## Quickstart

```bash
git clone https://github.com/hanzoai/dataroom.git
cd dataroom
npm install

# Configure
cp .env.example .env
# Required: POSTGRES_PRISMA_URL, NEXTAUTH_SECRET, BLOB_READ_WRITE_TOKEN or S3 creds
# Auth: IAM_URL, IAM_CLIENT_ID, IAM_CLIENT_SECRET
# Email: RESEND_API_KEY
# Analytics: TINYBIRD_TOKEN

# Initialize database
npm run dev:prisma

# Development
npm run dev     # http://localhost:3000

# Production
npm run build
npm start
```

## Architecture

```
dataroom.hanzo.ai
       |
  Next.js 15 App
       |
  ┌────┼────────┬──────────┬──────────┐
  │    │        │          │          │
PostgreSQL  S3/Blob   Tinybird    Stripe
(Prisma)   (docs)   (analytics) (billing)
  │
  ├── Hanzo IAM (OAuth)
  ├── Resend (email)
  ├── Trigger.dev (jobs)
  └── Hanko (passkeys)
```

## Directory structure

```
hanzoai/dataroom/
  package.json          # hanzo-dataroom v0.1.0
  Dockerfile            # Container build
  middleware.ts         # Auth + custom domain routing
  next.config.mjs       # Image domains, rewrites, redirects
  app/                  # Next.js app router pages
  pages/                # Next.js pages router (API routes)
  components/           # React UI (shadcn/ui based)
    emails/             # react-email templates
  lib/                  # Shared utilities
    tinybird/           # Analytics datasources and endpoints
  prisma/
    schema/             # Multi-file Prisma schema
    migrations/         # Database migrations
    add-migration.sh    # Migration helper script
  ee/                   # Enterprise features
    features/           # Datarooms, SAML SSO, advanced analytics
    emails/             # Enterprise email templates
    stripe/             # Billing integration
    limits/             # Plan-based feature gates
    LICENSE.md          # Enterprise license
  context/              # React context providers
  styles/               # Global CSS
  public/               # Static assets
  components.json       # shadcn/ui config
  trigger.config.ts     # Trigger.dev v3 config
  vercel.json           # Vercel deployment config
```

## Key environment variables

| Variable | Purpose |
|----------|---------|
| `POSTGRES_PRISMA_URL` | PostgreSQL connection (pooled) |
| `POSTGRES_PRISMA_URL_NON_POOLING` | PostgreSQL connection (migrations) |
| `NEXTAUTH_SECRET` | NextAuth session secret |
| `IAM_URL` | Hanzo IAM URL (`https://hanzo.id`) |
| `IAM_CLIENT_ID` | Hanzo IAM OAuth client ID |
| `IAM_CLIENT_SECRET` | Hanzo IAM OAuth client secret |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob token (if using Vercel storage) |
| `NEXT_PRIVATE_UPLOAD_BUCKET` | S3 bucket name (if using S3) |
| `RESEND_API_KEY` | Resend email API key |
| `TINYBIRD_TOKEN` | Tinybird analytics token |
| `STRIPE_*` | Stripe keys (enterprise billing) |
| `TRIGGER_SECRET_KEY` | Trigger.dev API key |
| `NEXT_PRIVATE_DOCUMENT_PASSWORD_KEY` | Encryption key for document passwords |

## Database

Prisma multi-schema setup in `prisma/schema/`. Run migrations:

```bash
npm run dev:prisma              # Generate client + deploy migrations
npx prisma migrate dev          # Create new migration
bash prisma/add-migration.sh    # Helper for adding migrations
```

## Tinybird analytics

```bash
cd lib/tinybird
tb push datasources/*
tb push endpoints/get_*
```

## Docker

```bash
docker build -t hanzo-dataroom .
docker run -p 3000:3000 --env-file .env hanzo-dataroom
```

## Related Skills

- `hanzo/hanzo-iam.md` -- OAuth provider (Hanzo IAM at hanzo.id)
- `hanzo/hanzo-platform.md` -- PaaS deployment
- `hanzo/hanzo-sql.md` -- PostgreSQL database

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: dataroom, documents, analytics, docsend, papermark, sharing
**Prerequisites**: Node.js 22+, PostgreSQL, S3 or Vercel Blob, Resend API key
