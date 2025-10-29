---
description: Discover and activate relevant skills (292 skills, 28 gateways)
argument-hint: [category|search-term] (optional)
---

# Skills Discovery Assistant

You are helping the user discover and activate relevant skills from their skills library.

## Your Task

**User's Request:** `$ARGUMENTS`

Follow these steps:

### 1. Read Skills Catalog

Read the master catalog:
```bash
cat skills/README.md
```

### 2. Detect Project Context

Analyze the current directory to understand the project:
```bash
# List files to detect project type
ls -la | head -30

# Check for language/framework indicators
ls *.{json,md,go,py,rs,swift,zig,toml,yaml,yml} 2>/dev/null | head -20
```

**Technology Detection → Gateway Mapping:**
- `package.json` → **discover-frontend** (React, Next.js, TypeScript)
- `go.mod` → **discover-backend** (Go), may also need discover-api
- `requirements.txt`, `pyproject.toml`, `uv.lock` → **discover-backend** (Python), **discover-ml** if ML work
- `Cargo.toml` → **discover-backend** (Rust), **discover-wasm** if WASM
- `build.zig` → Zig skills at root level
- `*.swift`, `*.xcodeproj` → **discover-mobile** (iOS/Swift)
- `Dockerfile`, `docker-compose.yml` → **discover-containers**
- `.beads/` → Beads workflow skills (root level)
- `tests/`, `__tests__/` → **discover-testing**
- Database files → **discover-database**

### 3. Analyze Conversation Context

Review the current conversation for:
- Technologies mentioned (frameworks, tools, databases)
- Problems discussed (performance, debugging, deployment)
- Explicit skill requests
- Work phase (planning, implementation, testing, deployment)

Map to gateway keywords:
- "REST API" → **discover-api**
- "GraphQL" → **discover-api**
- "Postgres", "MongoDB", "Redis" → **discover-database**
- "Docker", "Kubernetes" → **discover-containers**
- "CI/CD", "GitHub Actions" → **discover-cicd**
- "observability", "logging", "metrics" → **discover-observability**
- "caching", "CDN" → **discover-caching**
- "debugging", "GDB", "profiling" → **discover-debugging**
- "build", "Make", "CMake" → **discover-build-systems**
- "ML", "model", "training" → **discover-ml**
- "math", "linear algebra" → **discover-math**
- "compiler", "parser", "AST" → **discover-plt**
- "diagram", "flowchart", "Mermaid", "visualization" → **discover-diagrams**

### 4. Provide Contextual Recommendations

Based on the argument provided:

**If NO ARGUMENT (default view):**

Display in this format:
```
━━━ SKILLS DISCOVERY ━━━

RECOMMENDED FOR THIS PROJECT:
→ discover-[category]
  Keywords: [key topics]
  cat skills/discover-[category]/SKILL.md

→ discover-[category]
  Keywords: [key topics]
  cat skills/discover-[category]/SKILL.md

CATEGORIES (292 skills, 28 gateways):
Frontend (8) | Database (8) | API (7) | Testing (6) | Diagrams (8) | ML (30)
Math (19) | Debugging (14) | Build Systems (8) | Caching (7) | Observability (8)
Containers (5) | CI/CD (4) | PLT (13) | Formal (10) | Cloud (13)

COMMANDS:
/skills api          - View API skills
/skills frontend     - View frontend skills
/skills postgres     - Search for 'postgres'
/skills list         - Show all categories
```

Recommend 2-4 gateway skills that match:
- Detected technologies in the current directory
- Topics discussed in conversation
- Common workflows for the project type

**If ARGUMENT = category name:**

**A) If discover-{category} gateway exists:**
```
━━━ {CATEGORY} SKILLS (N total) ━━━

Keywords: [comma-separated keywords]

KEY SKILLS:
→ [skill-name] - [one-line description]
→ [skill-name] - [one-line description]
→ [skill-name] - [one-line description]

LOAD GATEWAY:
cat skills/discover-{category}/SKILL.md    # Overview + quick reference

LOAD FULL INDEX:
cat skills/{category}/INDEX.md              # All skills with details

LOAD SPECIFIC SKILL:
cat skills/{category}/[skill-name].md       # Individual skill
```

**B) If searching root-level skills:**
Check for skills like:
- `skill-*.md` (meta skills)
- `beads-*.md` (workflow skills)
- Root-level technology skills

Display similarly but note they're at root level.

**Example for `/skills api`:**
```
━━━ API SKILLS (7 total) ━━━

Keywords: REST, GraphQL, authentication, authorization, rate limiting

KEY SKILLS:
→ rest-api-design - RESTful resource modeling, HTTP semantics
→ graphql-schema-design - GraphQL types, resolvers, N+1 prevention
→ api-authentication - JWT, OAuth 2.0, API keys, sessions
→ api-authorization - RBAC, ABAC, policy engines
→ api-rate-limiting - Token bucket, sliding window algorithms
→ api-versioning - API versioning, deprecation, compatibility
→ api-error-handling - RFC 7807, validation errors

LOAD:
cat skills/discover-api/SKILL.md       # Gateway overview
cat skills/api/INDEX.md                # Full details
cat skills/api/rest-api-design.md     # Specific skill
```

**If ARGUMENT = search term:**

Search across:
- Gateway skill descriptions (discover-*/SKILL.md)
- Category INDEX.md files
- skills/README.md catalog
- Root-level skill filenames

Display matching gateway categories FIRST, then specific skills:
```
━━━ SEARCH: 'postgres' ━━━

GATEWAY:
→ discover-database
  Keywords: PostgreSQL, MongoDB, Redis, query optimization
  cat skills/discover-database/SKILL.md

MATCHING SKILLS:
→ postgres-query-optimization.md - Database/Performance
  Debug slow queries, EXPLAIN plans, index design
  cat skills/database/postgres-query-optimization.md

→ postgres-migrations.md - Database/Schema
  Schema changes, zero-downtime deployments
  cat skills/database/postgres-migrations.md

→ postgres-schema-design.md - Database/Design
  Designing schemas, relationships, data types
  cat skills/database/postgres-schema-design.md

RELATED GATEWAYS:
discover-observability, discover-caching, discover-debugging

[Refine search: /skills postgres optimization]
[View category: /skills database]
```

**If ARGUMENT = "list":**

Show all 28 gateway categories:
```
━━━ ALL CATEGORIES (292 skills, 28 gateways) ━━━

BACKEND & DATA:
  discover-api (7)         - REST, GraphQL, auth, rate limiting
  discover-database (8)    - Postgres, MongoDB, Redis, optimization
  discover-data (5)        - ETL, streaming, batch processing
  discover-caching (7)     - Redis, CDN, HTTP caching, invalidation

FRONTEND & MOBILE:
  discover-frontend (8)    - React, Next.js, state management, a11y
  discover-mobile (4)      - iOS, Swift, SwiftUI, concurrency

TESTING & DOCUMENTATION:
  discover-testing (6)     - Unit, integration, e2e, TDD, coverage
  discover-diagrams (8)    - Mermaid flowcharts, sequence, ER, architecture, Gantt

INFRASTRUCTURE:
  discover-containers (5)  - Docker, Kubernetes, security
  discover-cicd (4)        - GitHub Actions, pipelines
  discover-cloud (13)      - Modal, AWS, GCP, serverless
  discover-infra (6)       - Terraform, IaC, Cloudflare Workers
  discover-observability (8) - Logging, metrics, tracing, alerts
  discover-debugging (14)  - GDB, LLDB, profiling, memory leaks
  discover-build-systems (8) - Make, CMake, Gradle, Maven, Bazel
  discover-deployment (6)  - Netlify, Heroku, platforms
  discover-realtime (4)    - WebSockets, SSE, pub/sub

SPECIALIZED:
  discover-ml (30)         - Training, RAG, embeddings, evaluation
  discover-math (19)       - Linear algebra, topology, category theory
  discover-plt (13)        - Compilers, type systems, verification
  discover-formal (10)     - SAT/SMT, Z3, Lean, theorem proving
  discover-wasm (4)        - WebAssembly fundamentals, Rust to WASM
  discover-ebpf (4)        - eBPF tracing, networking, security
  discover-ir (5)          - LLVM IR, compiler optimizations
  discover-modal (2)       - Modal functions, scheduling
  discover-engineering (4) - Code review, documentation, leadership
  discover-product (4)     - Product strategy, roadmaps
  discover-collab (5)      - Collaboration, code review, pair programming

AGENT SKILLS (Root):
  elegant-design          - UI/UX design, accessibility, design systems
  anti-slop               - Detect/eliminate AI-generated patterns
  typed-holes-refactor    - Systematic TDD-based refactoring

META SKILLS (Root):
  skill-*.md              - Discovery and creation
  beads-*.md              - Workflow and task management

[View category: /skills api]
[Search: /skills postgres]
```

### 5. Output Requirements

**Format Guidelines:**
- Use Unicode box drawing (━ ─ │) for section headers
- Use `→` for list items
- Keep output under 30 lines for default view
- Include clear, copy-paste commands
- Group related items logically
- Show only relevant categories/skills for the context

**Tone:**
- Direct and helpful
- Low noise, high signal
- Focus on what the user needs now
- Don't explain the system unless asked

**DO NOT:**
- Modify any skill files
- Create new skills
- Change README.md or INDEX files
- Make assumptions about skills you haven't read
- Display full skill contents (only summaries)
- Reference _INDEX.md (it's archived)

### 6. Graceful Fallbacks

**If skills/README.md not found:**
```
━━━ ERROR ━━━

Skills catalog not found at skills/README.md

Expected structure:
skills/
├── README.md              (Master catalog)
├── discover-*/SKILL.md    (28 gateway skills)
└── {category}/INDEX.md    (Category indexes)

Is your repository in a different location?
```

**If no matches for search:**
```
━━━ NO RESULTS: '$ARGUMENTS' ━━━

No skills found matching your search.

TRY:
→ Broader search term
→ View all gateways: /skills list
→ Browse full catalog: cat skills/README.md
→ Check a category: /skills api
```

**If empty project directory:**
```
━━━ SKILLS DISCOVERY ━━━

No project files detected in current directory.

GENERAL-PURPOSE GATEWAYS:
→ discover-collab - Collaboration, documentation, CodeTour walkthroughs
  cat skills/discover-collab/SKILL.md

ROOT-LEVEL SKILLS:
→ beads-workflow.md - Multi-session task management
→ skill-creation.md - Creating new atomic skills
→ skill-repo-discovery.md - Discover skills for repositories

[View all: /skills list]
[Browse catalog: cat skills/README.md]
```

## Remember

- Read skills/README.md to get accurate information (NOT _INDEX.md - that's archived)
- Recommend gateway skills first, then specific skills
- Match skills to project context when possible
- Keep output concise and actionable
- Never modify the skills library
- Provide clear, copy-paste commands
- Use Unicode box drawing for visual clarity
- The catalog has: 292 skills, 28 gateways, 31 categories
