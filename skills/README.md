# Skills Catalog

**Total Skills**: 306 across 32 categories
**Gateway Skills**: 29 (auto-discovered by Claude Code)
**Agent Skills**: 3 (elegant-design, anti-slop, typed-holes-refactor)

## Overview

A skill is a directory. `{skill}/SKILL.md` says what it covers and names the
files beside it; those files are read when they are wanted. That is the whole
structure, and an agent runtime already does the reading, so nothing here
restates the file list a second time.

There used to be a `discover-{skill}/` twin for every skill, holding a SKILL.md
that only listed what `{skill}/` contained. It was a second address for one
thing, so it is gone.

## How It Works

### Automatic Discovery
A skill's `description` is what an agent matches against, so the right ones load
from the words in the request.

**Example**: "Design a REST API with Postgres" loads `api` and `database`; their
reference files are read from there as the work needs them.

### Manual Discovery
Browse categories and load skills explicitly:

```bash
# List every skill
ls -d skills/*/

# Load a category overview
cat skills/api/INDEX.md

# Load a specific skill
cat skills/api/rest-api-design.md
```

## Gateway Skills by Category

### Backend & APIs (7 skills)
**Gateway**: `api`
**Keywords**: API, REST, GraphQL, authentication, authorization, rate limiting
**Skills**: REST API design, GraphQL schemas, authentication (JWT/OAuth), authorization (RBAC/ABAC), rate limiting, versioning, error handling

```bash
cat skills/api/INDEX.md
```

---

### Databases (8 skills)
**Gateway**: `database`
**Keywords**: database, SQL, PostgreSQL, MongoDB, Redis, query optimization
**Skills**: PostgreSQL (schema, queries, migrations), MongoDB documents, Redis structures, connection pooling, ORMs, database selection

```bash
cat skills/database/INDEX.md
```

---

### Frontend (8 skills + elegant-design Agent Skill)
**Gateway**: `frontend`
**Keywords**: React, Next.js, UI, components, state management, forms
**Skills**: React patterns, Next.js App Router, state management, data fetching, forms, accessibility, performance, SEO
**Note**: `elegant-design` is a separate Agent Skill for UI design work

```bash
cat skills/frontend/INDEX.md
```

---

### Testing (6 skills)
**Gateway**: `testing`
**Keywords**: testing, tests, unit, integration, e2e, TDD, coverage
**Skills**: Unit testing, integration testing, e2e testing, TDD, test coverage, performance testing

```bash
cat skills/testing/INDEX.md
```

---

### Diagrams & Visualization (8 skills)
**Gateway**: `diagrams`
**Keywords**: diagram, flowchart, sequence, architecture, ER, UML, Mermaid, visualization
**Skills**: Flowcharts, sequence diagrams, class/state diagrams, ER diagrams, architecture (C4/block), charts (pie/XY/quadrant/radar), project diagrams (Gantt/timeline), specialized (Git/Sankey/mindmaps)

```bash
cat skills/diagrams/INDEX.md
```

---

### Hanzo Ecosystem (11 skills)
**Gateway**: `hanzo`
**Keywords**: Hanzo, local AI, privacy, Hanzo Node, Hanzo SDK, Hanzo MCP, streaming, ZenLM, training, mining, Zoo Gym, Go, Rust, Python
**Skills**: Hanzo Node (local inference), Python SDK (multi-provider), Go SDK (concurrent backend), @hanzo/ui (React components), Hanzo MCP (agentic workflows), Hanzo Dev (terminal agent - Rust/JS/Python), Hanzo Live (streaming), ZenLM (local AI models), Zoo Gym (training-free GRPO), Hanzo Desktop (productivity & mining), Hanzo Chat (conversation interface)

```bash
cat skills/hanzo/INDEX.md
```

**Why Use Hanzo**:
- ⚡ **10-50x faster development**: Use SDK abstractions instead of manual API integration
- 🔒 **Privacy-first**: Local AI via Hanzo Node with selective cloud fallback
- 🎯 **Automatic routing**: SDK analyzes queries and routes to optimal model
- 💰 **Cost optimization**: Built-in cost tracking + mine $AI tokens with Desktop
- 🔄 **Production-ready**: Automatic retries, circuit breaking, caching included
- 🌱 **Sustainable**: ZenLM models use 95% less energy than cloud AI
- 💚 **Wildlife conservation**: Zoo Gym by Zoo Labs Foundation (501c3)

**Organizations**:
- **Hanzo AI** (github.com/hanzoai): AI tools, SDKs, Desktop, Node, Engine
- **ZenLM** (github.com/zenlm): ZenLM models on GitHub & HuggingFace
- **Zoo AI** (github.com/zooai): Zoo Gym training platform
- **Zoo Labs Foundation** (zoo.ngo): 501(c)(3) for wildlife conservation AI

---

### Machine Learning (33 skills)
**Gateway**: `ml`
**Keywords**: machine learning, ML, AI, models, training, embeddings, transformers
**Skills**: Model training, inference, embeddings, RAG, evaluation, routing, fine-tuning, PyTorch, TensorFlow, Hugging Face, DSPy with Hanzo SDK, model selection with automatic routing, multi-model orchestration

```bash
cat skills/ml/INDEX.md
```

**Enhanced with Hanzo**: See `dspy-setup.md`, `llm-model-selection.md`, and `multi-model-orchestration.md` for Hanzo SDK integration patterns.

---

### Mathematics (19 skills)
**Gateway**: `math`
**Keywords**: mathematics, algorithms, linear algebra, calculus, topology, category theory
**Skills**: Linear algebra, calculus, topology, category theory, set theory, number theory, abstract algebra, graph theory, optimization

```bash
cat skills/math/INDEX.md
```

---

### Elegant Design (Agent Skill)
**Gateway**: Auto-discovered separately
**Keywords**: design, UI, UX, accessibility, chat, terminal, code display, streaming
**Skills**: World-class accessible interfaces, shadcn/ui, interactive components, design systems

```bash
# This is an Agent Skill - loads automatically when doing UI design work
cat skills/elegant-design/SKILL.md
```

---

### Debugging (14 skills)
**Gateway**: `debugging`
**Keywords**: debugging, GDB, LLDB, profiling, memory leaks, performance
**Skills**: GDB/LLDB, Python debugging, browser DevTools, remote debugging, memory profiling, concurrency debugging

```bash
cat skills/debugging/INDEX.md
```

---

### Programming Language Theory (13 skills)
**Gateway**: `plt`
**Keywords**: compilers, parsers, type systems, interpreters, LLVM
**Skills**: Parsing, type systems, code generation, compiler optimizations, LLVM, bytecode, interpreters

```bash
cat skills/plt/INDEX.md
```

---

### Cloud & Serverless (13 skills)
**Gateway**: `cloud`
**Keywords**: cloud, serverless, Modal, AWS, GCP, functions, deployment
**Skills**: Modal platform, serverless patterns, cloud deployment, scaling, GPU workloads

```bash
cat skills/cloud/INDEX.md
```

---

### Formal Methods (10 skills)
**Gateway**: `formal`
**Keywords**: formal methods, SAT, SMT, Z3, Lean, theorem proving, verification
**Skills**: SAT solving, SMT theory, Z3 solver, Lean theorem proving, constraint solving, formal verification

```bash
cat skills/formal/INDEX.md
```

---

### Observability (8 skills)
**Gateway**: `observability`
**Keywords**: monitoring, logging, tracing, metrics, distributed tracing, alerts
**Skills**: Structured logging, metrics instrumentation, distributed tracing, alerting, dashboard design

```bash
cat skills/observability/INDEX.md
```

---

### Build Systems (8 skills)
**Gateway**: `build-systems`
**Keywords**: build systems, Make, CMake, Gradle, Maven, Bazel, compilation
**Skills**: Make, CMake, Gradle, Maven, Bazel, cross-platform builds, build optimization

```bash
cat skills/build-systems/INDEX.md
```

---

### Caching (7 skills)
**Gateway**: `caching`
**Keywords**: caching, cache, Redis, CDN, HTTP caching, performance
**Skills**: Caching fundamentals, HTTP caching, CDN edge caching, Redis patterns, cache invalidation, Service Workers, monitoring

```bash
cat skills/caching/INDEX.md
```

---

### Deployment (6 skills)
**Gateway**: `deployment`
**Keywords**: deployment, Netlify, Heroku, CI/CD, production, releases
**Skills**: Netlify deployment/functions/optimization, Heroku deployment/addons/troubleshooting

```bash
cat skills/deployment/INDEX.md
```

---

### Infrastructure (6 skills)
**Gateway**: `infrastructure`
**Keywords**: infrastructure, Terraform, IaC, Cloudflare Workers, security, DevOps
**Skills**: Cloudflare Workers, infrastructure security, cost optimization, IaC patterns

```bash
cat skills/infrastructure/INDEX.md
```

---

### Anti-Slop (Agent Skill)
**Gateway**: Auto-discovered separately
**Keywords**: slop, AI patterns, code quality, content quality
**Skills**: Detect and eliminate generic AI patterns in text, code, and design

```bash
# This is an Agent Skill - loads automatically for quality review
cat skills/anti-slop/SKILL.md
```

---

### Typed Holes Refactor (Agent Skill)
**Gateway**: Auto-discovered separately
**Keywords**: refactoring, typed holes, incremental refactoring, test-driven
**Skills**: Systematic refactoring using typed holes methodology

```bash
# This is an Agent Skill - loads automatically for refactoring work
cat skills/typed-holes-refactor/SKILL.md
```

---

### Containers (5 skills)
**Gateway**: `containers`
**Keywords**: Docker, containers, Kubernetes, container security, docker-compose
**Skills**: Dockerfile optimization, docker-compose, container security, networking, registry management

```bash
cat skills/containers/INDEX.md
```

---

### Data Pipelines (5 skills)
**Gateway**: `data`
**Keywords**: ETL, data pipelines, batch processing, stream processing, orchestration
**Skills**: ETL patterns, batch processing, stream processing, pipeline orchestration, data validation

```bash
cat skills/data/INDEX.md
```

---

### Mobile (4 skills)
**Gateway**: `mobile`
**Keywords**: iOS, Swift, SwiftUI, mobile, SwiftData, concurrency
**Skills**: SwiftUI architecture, Swift concurrency, iOS networking, iOS testing

```bash
cat skills/mobile/INDEX.md
```

---

### CI/CD (4 skills)
**Gateway**: `cicd`
**Keywords**: CI/CD, GitHub Actions, automation, pipelines
**Skills**: GitHub Actions, GitLab CI, pipeline patterns, deployment automation

```bash
cat skills/cicd/INDEX.md
```

---

### Realtime (4 skills)
**Gateway**: `realtime`
**Keywords**: realtime, WebSockets, Server-Sent Events, streaming, pub/sub
**Skills**: WebSocket patterns, SSE, real-time protocols, state synchronization

```bash
cat skills/realtime/INDEX.md
```

---

### WebAssembly (4 skills)
**Gateway**: `wasm`
**Keywords**: WebAssembly, WASM, wasm-pack, Rust to WASM
**Skills**: WASM fundamentals, Rust to WASM, WASM in browsers, WASI

```bash
cat skills/wasm/INDEX.md
```

---

### eBPF (4 skills)
**Gateway**: `ebpf`
**Keywords**: eBPF, kernel, tracing, networking, BPF
**Skills**: eBPF fundamentals, tracing, networking, security monitoring

```bash
cat skills/ebpf/INDEX.md
```

---

### Engineering Practices (4 skills)
**Gateway**: `engineering`
**Keywords**: engineering practices, code review, technical leadership
**Skills**: Code review, documentation, technical leadership, team practices

```bash
cat skills/engineering/INDEX.md
```

---

### Product (4 skills)
**Gateway**: `product`
**Keywords**: product management, roadmap, strategy, prioritization
**Skills**: Product strategy, roadmap planning, user research, prioritization

```bash
cat skills/product/INDEX.md
```

---

### Collaboration (5 skills)
**Gateway**: `collaboration`
**Keywords**: collaboration, code review, documentation, pair programming
**Skills**: Code review, documentation, pair programming, team workflows

```bash
cat skills/collaboration/INDEX.md
```

---

### Intermediate Representation (5 skills)
**Gateway**: `ir`
**Keywords**: IR, LLVM IR, intermediate representation, compiler optimizations
**Skills**: LLVM IR, SSA form, code generation, IR optimizations

```bash
cat skills/ir/INDEX.md
```

---

### Modal (2 skills)
**Gateway**: `modal`
**Keywords**: Modal, serverless functions, cloud deployment
**Skills**: Modal functions basics, Modal scheduling

```bash
cat skills/modal/INDEX.md
```

---

### Workflow (5 skills)
**Gateway**: `workflow`
**Keywords**: Beads, workflow, task management, synthesis, cleanup, consolidation, roadmap
**Skills**: Beads workflow, dependency management, multi-session patterns, context strategies, project synthesis

```bash
cat skills/workflow/INDEX.md
```

---

## Quick Reference

### Find Skills by Technology

**Python**: ml, data, testing
**TypeScript/React**: frontend, testing
**Zig**: zig, build-systems
**Go**: cloud, distributed-systems
**Rust**: wasm, formal

### Find Skills by Task

**Build API**: api, database, testing
**Build Frontend**: frontend, elegant-design, api, testing
**Optimize Performance**: caching, database, observability
**Debug Issues**: debugging, observability
**Deploy Application**: deployment, cloud, infrastructure, cicd
**Data Engineering**: data, database, observability

### Root-Level Skills

40 skills at root level (not yet categorized):
- Zig skills (zig-*.md)
- TUI skills (bubbletea-*.md, ratatui-*.md)
- iOS skills (swift*.md, ios-*.md, swiftui-*.md)
- Modal skills (modal-*.md)
- Networking skills (mosh-*.md, mtls-*.md, nat-traversal.md, network-resilience-patterns.md, tailscale-vpn.md)
- Discovery/Meta skills (skill-*.md)
- Misc (apache-iceberg.md, duckdb-analytics.md, redpanda-streaming.md)

These will be integrated into appropriate categories in a future update.

---

## Usage Patterns

### Automatic (Recommended)
Just start working. Claude Code will auto-activate relevant gateway skills.

### Browse Then Load
1. Review this README for category overview
2. Load category index: `cat skills/{category}/INDEX.md`
3. Load specific skill: `cat skills/{category}/{skill}.md`

### Search by Keyword
```bash
# Find skills matching a keyword
grep -r "keyword" skills/*/INDEX.md

# Find a skill by what it says it covers
grep "description:" skills/*/SKILL.md
```

---

## Architecture Benefits

✅ **Context Efficient**: 60-84% reduction in tokens loaded
✅ **Auto-Discovery**: Keywords trigger relevant gateway skills
✅ **Progressive Loading**: Load only what you need, when you need it
✅ **Scalable**: Can grow to 1000+ skills with same footprint
✅ **Maintainable**: Small, focused files instead of monolithic index
✅ **Flexible**: Works for automatic AND manual discovery

---

**See also**:
- `skill-repo-discovery.md` - Discover skills for a new repository
- `skill-prompt-discovery.md` - Discover skills from user prompts
- `CLAUDE.md` - Complete development guidelines with Work Plan Protocol
