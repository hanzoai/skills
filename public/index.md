# Hanzo Skills

Native compiled cloud skills (1472 skills across 14 categories). Progressive disclosure by default — load only what your task needs.

## Install

    curl -fsSL https://hanzo.sh | sh   # macOS, Linux, WSL

Run `hanzo` — signs in, wires tools, launches your agent. `hanzo dev --claude|--codex|--cursor|--agy` switches agents; `hanzo -- -c` resumes.

Or paste this into your agent (Claude Code · Codex · Cursor · Antigravity):

    Set up Hanzo on this machine: follow https://hanzo.sh/setup.md

The skills, over plain https:

    curl -fsSL https://hanzoskills.com/llms.txt                  # Tier 0: compact category tree
    curl -fsSL https://hanzoskills.com/categories/hanzo.md       # Tier 1: one category
    curl -fsSL https://hanzoskills.com/skills/hanzo/SKILL.md     # Tier 2: one skill
    curl -fsSL https://hips.hanzo.ai/llms.txt                    # HIP architectural specs

## Start Here

- [`skill`](skills/skill/SKILL.md) — Manifest for an AI agent working with Hanzo — what runs on your own machine and the command that starts it, what needs an account on the hosted cloud at api.hanzo.ai, and how to work either way safely. Onboarding (a c...
- [`hanzo`](skills/hanzo/SKILL.md) — Sign up free with a wallet. Get an API key free. A small free allowance to try it. Top up in USDC or USDT over the API to raise your limits and pay as you go. No human required. Onboard an agent onto Hanzo, the AI ope...
- [`hanzo-base`](skills/hanzo-base/SKILL.md) — Hanzo Base is an open source Go backend that provides a complete application backend in a single binary.
- [`hanzo-cloud`](skills/hanzo-cloud/SKILL.md) — hanzoai/cloud is ONE Go binary that embeds every Hanzo-native subsystem — IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai (the former LLM control plane), and the rest — behind a single plugin registry.
- [`hanzo-mcp`](skills/hanzo-mcp/SKILL.md) — Hanzo MCP implements the Model Context Protocol -- an open standard for exposing tools, resources, and prompts to AI agents.
- [`app-port`](skills/app-port/SKILL.md) — Port an existing application onto the Hanzo stack — Hanzo Base and Hanzo Cloud OSS for the backend, ZAP for transport, @hanzo/ui and @hanzo/gui for the interface — and move its Python or JavaScript runtime into Go, Ru...

## Categories

- [Hanzo Platform & Cloud](categories/hanzo.md) (165 skills · [JSON](categories/hanzo.json)) — Core Hanzo stack, native Go/Rust/C++ services, Base, Cloud, CLI, MCP, and migration guides.
- [AI, ML & Agent Engineering](categories/ai-agents.md) (120 skills · [JSON](categories/ai-agents.json)) — LLM APIs, agent workflows, MCP servers, skill authoring, inference, audio, vision, and research.
- [Systems & Core Engineering](categories/engineering.md) (538 skills · [JSON](categories/engineering.json)) — Languages, distributed systems, debugging, testing, code review, protocols, and build tools.
- [Cloud, Containers & DevOps](categories/cloud-infra.md) (69 skills · [JSON](categories/cloud-infra.json)) — Kubernetes, containers, CI/CD, observability, edge deploys, and site architecture.
- [Data, Databases & Analytics](categories/data-databases.md) (24 skills · [JSON](categories/data-databases.json)) — SQL, vector search, data engineering, streaming pipelines, and product analytics.
- [Security & Cryptography](categories/security.md) (33 skills · [JSON](categories/security.json)) — AppSec, threat modeling, OWASP guides, IAM, KMS, and applied cryptography.
- [Frontend, UI & Design Systems](categories/frontend-design.md) (86 skills · [JSON](categories/frontend-design.json)) — React, CSS, design systems, Figma workflows, diagrams, mobile UI, and browser automation.
- [Mathematics & Formal Methods](categories/math-formal.md) (29 skills · [JSON](categories/math-formal.json)) — Category theory, algebra, proof assistants, formal verification, and optimization.
- [Product, Growth & Operations](categories/growth-product.md) (108 skills · [JSON](categories/growth-product.json)) — SEO, CRO, pricing, copywriting, GTM, revops, email, and team operations.
- [HIPs: Core AI & Models](categories/standards-core.md) (61 skills · [JSON](categories/standards-core.json)) — Hanzo Improvement Proposals for core architecture, governance, and AI model standards.
- [HIPs: Cloud & Infrastructure](categories/standards-infra.md) (113 skills · [JSON](categories/standards-infra.json)) — Hanzo Improvement Proposals for compute, networking, storage, and cloud infrastructure.
- [HIPs: Interfaces & Protocols](categories/standards-interface.md) (54 skills · [JSON](categories/standards-interface.json)) — Hanzo Improvement Proposals for ZAP, MCP, SDKs, APIs, and realtime protocols.
- [HIPs: Security, IAM & Crypto](categories/standards-security.md) (26 skills · [JSON](categories/standards-security.json)) — Hanzo Improvement Proposals for identity, KMS, zero-trust, and post-quantum security.
- [HIPs: Platform & Applications](categories/standards-apps.md) (46 skills · [JSON](categories/standards-apps.json)) — Hanzo Improvement Proposals for developer apps, agent surfaces, and ecosystem products.
