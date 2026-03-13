# Hanzo Sign - Open Source Document Signing

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-kms.md`, `hanzo/hanzo-platform.md`

## Overview

Hanzo Sign is an **open-source DocuSign alternative** for electronic document signing. Turborepo monorepo with a React Router (Remix) + Hono server frontend, tRPC + ts-rest APIs, Prisma ORM, and PDF signing via P12 certificates or Google Cloud KMS. Live at `sign.hanzo.ai`. Version 2.7.1. Licensed AGPL-3.0.

### Why Hanzo Sign?

- **Self-hostable**: Full control over your signing infrastructure
- **Open source**: Inspect and audit the entire signing pipeline
- **Multiple signing providers**: Local P12 certificates or Google Cloud HSM
- **Swappable storage**: Database or S3-compatible object storage
- **API-first**: REST (V1) and tRPC (V2) APIs with OpenAPI spec
- **Enterprise features**: Teams, custom branding, webhooks, bulk sending
- **i18n**: Multi-language via Lingui
- **AI integration**: Google Vertex AI (Gemini) for document analysis

### Tech Stack

- **Frontend**: React 18, React Router v7 (Remix), Tailwind CSS, shadcn/ui, Radix UI
- **Server**: Hono (on Node.js)
- **Database**: PostgreSQL 15 (Prisma ORM + Kysely)
- **API**: tRPC (V2, current), ts-rest (V1, deprecated)
- **Auth**: Arctic (OAuth), WebAuthn/Passkeys, Hanzo IAM (hanzo.id)
- **Email**: React Email, Nodemailer (SMTP, Resend, MailChannels)
- **PDF**: @libpdf/core, pdfjs-dist, PDF-Lib
- **Jobs**: Inngest or local database-backed queue
- **Build**: Turborepo + npm workspaces, Vite
- **Testing**: Playwright (E2E)

### OSS Base

Repo: `hanzoai/sign` (fork of `documenso/documenso`).

## When to use

- Sending documents for electronic signature
- Building document signing workflows
- Self-hosting a DocuSign-like service
- Creating document templates for recurring signatures
- Bulk sending documents via API
- Embedding signing into existing applications

## Hard requirements

1. **Node.js** v22 or above
2. **PostgreSQL** database
3. **SMTP server** or email provider (Resend, MailChannels)
4. **P12 signing certificate** (or Google Cloud KMS)

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://sign.hanzo.ai` (port 3000) |
| API V2 | `/api/v2/*` (tRPC + OpenAPI) |
| API V1 | `/api/v1/*` (ts-rest, deprecated) |
| Auth | Arctic OAuth + WebAuthn + Hanzo IAM |
| DB | PostgreSQL (Prisma + Kysely) |
| Root package | `@hanzo/sign-root` |
| Version | 2.7.1 |
| License | AGPL-3.0 |
| Repo | `github.com/hanzoai/sign` |

## One-file quickstart

### Developer Quickstart (Docker)

```bash
# Clone and setup
git clone https://github.com/hanzoai/sign.git
cd sign
cp .env.example .env

# One command: install, start db/mail containers, migrate, seed, dev
npm run d
```

### Manual Setup

```bash
npm install
cp .env.example .env

# Set required env vars:
# NEXTAUTH_SECRET, NEXT_PUBLIC_WEBAPP_URL
# NEXT_PRIVATE_DATABASE_URL, NEXT_PRIVATE_DIRECT_DATABASE_URL
# NEXT_PRIVATE_SMTP_FROM_NAME, NEXT_PRIVATE_SMTP_FROM_ADDRESS

npm run prisma:migrate-dev
npm run translate:compile
npm run dev
```

### Docker (Production)

```bash
docker run -d --restart=unless-stopped \
  -p 3000:3000 \
  -v hanzo-sign:/app/data \
  --name hanzo-sign \
  hanzo-sign:latest
```

## Core Concepts

### Architecture

```
                         Remix App (Hono Server)
                            apps/remix
  /api/v1/*    /api/v2/*    /api/trpc/*    /api/jobs/*    React Router UI
  (ts-rest)     (tRPC)       (tRPC)       (Jobs API)
       |            |            |             |
  @hanzo/sign-api  @hanzo/sign-trpc  @hanzo/sign-lib  @hanzo/sign-email
                              |
               +--------------+--------------+
               |              |              |
          Storage         Jobs           PDF Signing
          Provider       Provider         Provider
               |              |              |
         DB or S3      Inngest/Local   Local P12 or GCloud HSM
```

### Monorepo Packages

**Applications** (`apps/`):

| Package | Description | Port |
|---------|-------------|------|
| `@hanzo/sign-remix` | Main app (React Router + Hono) | 3000 |
| `@hanzo/sign-documentation` | Documentation site (Nextra) | 3002 |
| `@hanzo/sign-openpage-api` | Public analytics API | 3003 |

**Core Packages** (`packages/`):

| Package | Description |
|---------|-------------|
| `@hanzo/sign-lib` | Core business logic (server, client, universal) |
| `@hanzo/sign-trpc` | tRPC API layer with OpenAPI (V2) |
| `@hanzo/sign-api` | REST API layer via ts-rest (V1) |
| `@hanzo/sign-prisma` | Database layer (Prisma ORM + Kysely) |
| `@hanzo/sign-ui` | UI components (shadcn + Radix + Tailwind) |
| `@hanzo/sign-email` | Email templates (React Email) |
| `@hanzo/sign-auth` | Auth (Arctic OAuth, WebAuthn/Passkeys) |
| `@hanzo/sign-signing` | PDF signing (Local P12, Google Cloud KMS) |
| `@hanzo/sign-ee` | Enterprise Edition features |
| `@hanzo/sign-assets` | Static assets |

### Swappable Providers

**Storage** (`NEXT_PUBLIC_UPLOAD_TRANSPORT`):
- `database` -- Store files as Base64 in PostgreSQL (default)
- `s3` -- S3-compatible storage (AWS, MinIO, etc.)

**PDF Signing** (`NEXT_PRIVATE_SIGNING_TRANSPORT`):
- `local` -- P12 certificate file (default)
- `gcloud-hsm` -- Google Cloud KMS Hardware Security Module

**Email** (`NEXT_PRIVATE_SMTP_TRANSPORT`):
- `smtp-auth` -- Standard SMTP with credentials (default)
- `smtp-api` -- SMTP with API key
- `resend` -- Resend API
- `mailchannels` -- MailChannels API

**Background Jobs** (`NEXT_PRIVATE_JOBS_PROVIDER`):
- `local` -- Database-backed queue (default)
- `inngest` -- Managed cloud service

### Document Signing Flow

1. Upload document to storage provider (DB or S3)
2. Add recipients (signers, viewers, approvers)
3. Add signature fields (positions on PDF)
4. Send document (triggers email job)
5. Recipient signs via unique link
6. `seal-document` job finalizes the PDF
7. Signing provider cryptographically signs the PDF
8. Signed PDF stored back to storage provider

### Auth / IAM Integration

Hanzo Sign supports Hanzo IAM (hanzo.id) for production auth:

```bash
IAM_URL="https://hanzo.id"
IAM_CLIENT_ID="<your-client-id>"
IAM_CLIENT_SECRET="<your-client-secret>"
```

Also supports Google OAuth, Microsoft OAuth, and generic OIDC providers.

### Signing Certificates

For local signing, generate a P12 certificate:

```bash
openssl genrsa -out private.key 2048
openssl req -new -x509 -key private.key -out certificate.crt -days 365
read -s -p "Enter certificate password: " CERT_PASS
openssl pkcs12 -export -out certificate.p12 -inkey private.key -in certificate.crt \
    -password env:CERT_PASS -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -macalg sha1
```

Place at `apps/remix/resources/certificate.p12`.

### Environment Variables

| Variable | Purpose | Options |
|----------|---------|---------|
| `NEXTAUTH_SECRET` | Session encryption | Random string |
| `NEXT_PUBLIC_WEBAPP_URL` | Public URL | `https://sign.hanzo.ai` |
| `NEXT_PRIVATE_DATABASE_URL` | PostgreSQL connection | Connection string |
| `NEXT_PUBLIC_UPLOAD_TRANSPORT` | Storage provider | `database`, `s3` |
| `NEXT_PRIVATE_SIGNING_TRANSPORT` | Signing provider | `local`, `gcloud-hsm` |
| `NEXT_PRIVATE_SMTP_TRANSPORT` | Email provider | `smtp-auth`, `smtp-api`, `resend`, `mailchannels` |
| `NEXT_PRIVATE_JOBS_PROVIDER` | Jobs provider | `local`, `inngest` |
| `IAM_URL` | Hanzo IAM endpoint | `https://hanzo.id` |

### Development Services (Docker)

| Service | Port |
|---------|------|
| PostgreSQL | 54320 |
| Inbucket (Mail UI) | 9000 |
| Inbucket (SMTP) | 2500 |
| MinIO (S3 Dashboard) | 9001 |
| MinIO (S3 API) | 9002 |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No emails received | Using dev quickstart | Check Inbucket at http://localhost:9000 |
| "Failed to get private key bags" | P12 cert has no password | Re-create with password (min 4 chars) |
| Can't see env vars in scripts | Env not loaded | Wrap with `npm run with:env -- npm run myscript` |
| IPv6 deployment | Default binds IPv4 | Pass `npm run start -- -H ::` |
| Lingui errors | Translations not compiled | Run `npm run translate:compile` |

## Related Skills

- `hanzo/hanzo-id.md` - IAM and authentication (hanzo.id)
- `hanzo/hanzo-kms.md` - Secret management
- `hanzo/hanzo-platform.md` - PaaS deployment
- `hanzo/hanzo-commerce.md` - E-commerce (contracts, checkout)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: signing, documents, pdf, docusign, e-signature
**Prerequisites**: Node.js 22+, PostgreSQL, SMTP
