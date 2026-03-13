# Hanzo tRPC OpenAPI - Generate OpenAPI Specs from tRPC Routers

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-platform.md`, `hanzo/hanzo-openapi.md`

## Overview

`@hanzo/trpc-openapi` is a TypeScript library that generates **OpenAPI 3.x documents** from tRPC v11 routers and provides HTTP adapters to serve tRPC procedures as REST endpoints. Published on npm as `@hanzo/trpc-openapi`. This is a maintained fork of the abandoned `trpc-to-openapi` project with Hanzo-specific enhancements.

### Why @hanzo/trpc-openapi?

- **tRPC + REST**: Expose tRPC procedures as standard REST endpoints without duplicating code
- **OpenAPI doc generation**: Auto-generate Swagger/OpenAPI specs from your tRPC router definitions
- **Multiple adapters**: Express, Fastify, Next.js (pages + app router), Nuxt, Koa, Fetch API, standalone Node HTTP
- **Zod v4 native**: Built for Zod v4 with `zod-openapi` v5 integration
- **Optional meta**: Procedures without `.meta({ openapi: ... })` are still exposed with sensible defaults (method from procedure type, path from procedure path)

### Tech Stack

- **Language**: TypeScript 5.8
- **Runtime**: Node.js
- **Package Manager**: pnpm
- **tRPC**: v11.1.0+ (peer dependency)
- **Zod**: v4.x (peer dependency)
- **OpenAPI**: `openapi3-ts` v4.4.0, `zod-openapi` v5.x (peer dependency)
- **Testing**: Jest 29 + ts-jest
- **Linting**: ESLint + Prettier
- **Build**: TypeScript compiler (dual CJS + ESM output)

### OSS Base

Fork of [trpc-to-openapi](https://www.npmjs.com/package/trpc-to-openapi). Hanzo-specific additions:
- Optional input/output parameters are required by default in generated docs
- `.meta` is optional for all procedures (auto-generates defaults)
- `override` flag: use only provided openapi object, no defaults merged
- `additional` flag: merge provided openapi on top of defaults

Repo: `github.com/hanzoai/trpc-openapi`

## When to use

- Exposing tRPC procedures as REST/OpenAPI endpoints
- Generating OpenAPI/Swagger documentation from tRPC routers
- Building APIs that need both tRPC and REST access
- Incremental migration between REST and tRPC
- Auto-documenting Hanzo platform tRPC services

## Hard requirements

1. **@trpc/server** `^11.1.0`
2. **zod** `^4.0.0` (Zod v4, not v3)
3. **zod-openapi** `^5.0.1`
4. **Node.js** 22+ (based on `@types/node ^22`)

## Quick reference

| Item | Value |
|------|-------|
| npm Package | `@hanzo/trpc-openapi` |
| Version | `0.0.17` |
| Repo | `github.com/hanzoai/trpc-openapi` |
| Branch | `main` |
| License | MIT |
| tRPC Version | `^11.1.0` (peer dep) |
| Zod Version | `^4.0.0` (peer dep) |
| Build | `pnpm build` (runs tests first) |
| Test | `pnpm test` |
| Exports | CJS (`dist/cjs/`) + ESM (`dist/esm/`) |

## One-file quickstart

### Install

```bash
pnpm add @hanzo/trpc-openapi
# peer deps
pnpm add @trpc/server zod zod-openapi
```

### Define a tRPC router with OpenAPI metadata

```typescript
import { initTRPC } from '@trpc/server';
import { z } from 'zod';
import { OpenApiMeta, generateOpenApiDocument, createOpenApiExpressMiddleware } from '@hanzo/trpc-openapi';
import express from 'express';

// 1. Create tRPC instance with OpenApiMeta
const t = initTRPC.meta<OpenApiMeta>().create();

// 2. Define router with openapi metadata
const appRouter = t.router({
  getUser: t.procedure
    .meta({ openapi: { method: 'GET', path: '/users/{id}' } })
    .input(z.object({ id: z.string() }))
    .output(z.object({ id: z.string(), name: z.string() }))
    .query(({ input }) => ({ id: input.id, name: 'Hanzo User' })),

  createUser: t.procedure
    .meta({ openapi: { method: 'POST', path: '/users', protect: true } })
    .input(z.object({ name: z.string(), email: z.string().email() }))
    .output(z.object({ id: z.string(), name: z.string() }))
    .mutation(({ input }) => ({ id: 'usr_new', name: input.name })),
});

// 3. Generate OpenAPI document
const openApiDoc = generateOpenApiDocument(appRouter, {
  title: 'My API',
  version: '1.0.0',
  baseUrl: 'http://localhost:3000',
});

// 4. Serve REST endpoints
const app = express();
app.get('/openapi.json', (req, res) => res.json(openApiDoc));
app.use('/api', createOpenApiExpressMiddleware({ router: appRouter }));
app.listen(3000);
```

## Core Concepts

### Architecture

```
┌──────────────────────┐
│  tRPC Router         │
│  (procedures + meta) │
└──────┬───────────────┘
       │
  ┌────▼────────────────────┐
  │  generateOpenApiDocument │──── OpenAPI 3.x JSON
  └────┬────────────────────┘
       │
  ┌────▼────────────────────┐
  │  HTTP Adapter            │──── REST endpoints
  │  (Express/Fastify/Next)  │     GET /users/{id}
  └─────────────────────────┘     POST /users
```

### Adapters

| Adapter | Import | Handler Factory |
|---------|--------|-----------------|
| Express | `@hanzo/trpc-openapi` | `createOpenApiExpressMiddleware` |
| Fastify | `@hanzo/trpc-openapi` | `fastifyTRPCOpenApiPlugin` |
| Next.js Pages | `@hanzo/trpc-openapi` | `createOpenApiNextHandler` |
| Next.js App Router | `@hanzo/trpc-openapi` | `createOpenApiFetchHandler` |
| Nuxt | `@hanzo/trpc-openapi` | `createOpenApiNuxtHandler` |
| Koa | `@hanzo/trpc-openapi` | `createOpenApiKoaMiddleware` |
| Fetch | `@hanzo/trpc-openapi` | `createOpenApiFetchHandler` |
| Node HTTP | `@hanzo/trpc-openapi` | `createOpenApiHttpHandler` |
| Standalone | `@hanzo/trpc-openapi` | `createOpenApiHttpHandler` (with `http.createServer`) |

### OpenApiMeta Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `GET\|POST\|PATCH\|PUT\|DELETE` | (from proc type) | HTTP method |
| `path` | `/${string}` | (from proc path) | URL path with `{param}` placeholders |
| `enabled` | `boolean` | `true` | Include in OpenAPI doc and adapters |
| `protect` | `boolean` | `true` | Require Authorization header |
| `override` | `boolean` | `false` | Use only provided openapi object (no defaults) |
| `additional` | `boolean` | `false` | Merge provided openapi on top of defaults |
| `tags` | `string[]` | -- | Grouping tags |
| `summary` | `string` | -- | Short description |
| `description` | `string` | -- | Verbose description |
| `deprecated` | `boolean` | `false` | Mark as deprecated |
| `contentTypes` | `string[]` | `['application/json']` | Accepted content types |
| `requestHeaders` | `ZodObject` | -- | Custom request headers schema |
| `responseHeaders` | `ZodObject` | -- | Custom response headers schema |
| `errorResponses` | `number[]\|Record<number,string>` | -- | Error response codes |

### Input Handling

- **GET/DELETE**: Input from URL query parameters (auto-coerced to `number`, `boolean`, `bigint`, `date`)
- **POST/PATCH/PUT**: Input from JSON request body
- **Path parameters**: Defined as `{paramName}` in the path, must exist in input schema as `string`, `number`, `bigint`, or `date`

### Filtering Procedures

```typescript
const doc = generateOpenApiDocument<{ isPublic: boolean }>(appRouter, {
  title: 'API',
  version: '1.0.0',
  baseUrl: 'http://localhost:3000',
  filter: ({ metadata }) => metadata.isPublic === true,
});
```

## Directory structure

```
trpc-openapi/
├── src/
│   ├── index.ts              # Re-exports everything
│   ├── types.ts              # OpenApiMeta, OpenApiMethod, response types
│   ├── generator/
│   │   ├── index.ts          # generateOpenApiDocument()
│   │   ├── paths.ts          # OpenAPI path generation
│   │   └── schema.ts         # Zod-to-OpenAPI schema conversion
│   ├── adapters/
│   │   ├── express.ts        # Express middleware
│   │   ├── fastify.ts        # Fastify plugin
│   │   ├── next.ts           # Next.js handler (pages + app)
│   │   ├── nuxt.ts           # Nuxt handler
│   │   ├── koa.ts            # Koa middleware
│   │   ├── fetch.ts          # Fetch API handler
│   │   ├── standalone.ts     # Standalone Node HTTP
│   │   └── node-http/        # Core Node HTTP handler
│   └── utils/                # Internal utilities
├── test/
│   ├── generator.test.ts     # 115KB of generator tests
│   └── adapters/             # Adapter-specific tests
├── examples/
│   ├── with-express/
│   ├── with-fastify/
│   ├── with-nextjs/
│   ├── with-nextjs-appdir/
│   └── with-nuxtjs/
├── dist/
│   ├── cjs/                  # CommonJS build output
│   └── esm/                  # ESM build output
├── package.json              # @hanzo/trpc-openapi v0.0.17
├── tsconfig.json
└── jest.config.ts
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `zod is not a function` | Using Zod v3 instead of v4 | Upgrade to `zod ^4.0.0` |
| Procedures missing from OpenAPI doc | `enabled: false` in meta | Remove or set `enabled: true` |
| Query params not parsed | Complex types in GET input | Use `z.preprocess()` for objects/arrays in query params |
| 404 on REST endpoint | Path mismatch | Ensure `meta.openapi.path` matches exactly, paths are case-insensitive |
| Auth not working | `protect: true` but no `createContext` | Implement `createContext` to extract `Authorization` header |
| Type error with meta | tRPC not configured with `OpenApiMeta` | Use `initTRPC.meta<OpenApiMeta>().create()` |

## Related Skills

- `hanzo/hanzo-platform.md` - PaaS platform (uses tRPC + this library)
- `hanzo/hanzo-openapi.md` - OpenAPI specification patterns
- `hanzo/hanzo-llm-gateway.md` - LLM gateway API

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: trpc, openapi, swagger, rest, api
**Prerequisites**: TypeScript, tRPC v11, Zod v4
