# Hanzo Skills

The native cloud in a single binary. Go, Rust, C++. Zero interpreter bloat.
Run a local copy of the cloud, extend with /v1/*, and migrate your backend to Hanzo Base with realtime native SSE, analytics, and insights.

    curl hanzoskills.com/llms.txt          # the index, for an agent
    curl hanzoskills.com/<skill>.md        # one skill
    curl hanzoskills.com/all.md            # the whole corpus, one file

Every page is `text/markdown`. Paths work with or without the `.md`.

## Start Here: Migrate to Native Go with Hanzo Base

- [`hanzo-base`](hanzo-base.md) — Open-source single-binary Go backend with embedded SQLite/Postgres, auth, admin UI, realtime native SSE.
- [`app-port`](app-port.md) — Bring an existing app onto this stack: backend to Go, transport to ZAP, runtime out of Python or JavaScript.
- [`ui-port`](ui-port.md) — Make a rewritten interface match the one it replaces, by measurement rather than by eye.
- [`hanzo-cloud`](hanzo-cloud.md) — Unified cloud platform: host + plugins over ZAP transport.
- [`hanzo-stack`](hanzo-stack.md) — Full integrated local development environment (compose.yml).
- [`go-sdk`](go-sdk.md) — Generated Go client library for the Hanzo AI API.

## Everything

### hanzo
- [`ai-development-guide`](ai-development-guide.md) — This guide covers setting up Hanzo's full AI development stack — extensions, CLI agents, MCP tools, Zen models, and skills for cross-stack work across Hanzo AI and Lux blockchain.
- [`go-sdk`](go-sdk.md) — Hanzo Go SDK is a generated Go client library for the Hanzo AI API.
- [`hanzo-aci`](hanzo-aci.md) — Hanzo ACI (Agent Computer Interface) is a Python toolkit that gives AI agents the ability to edit files, lint code, index codebases, execute shell commands, and convert documents.
- [`hanzo-agent`](hanzo-agent.md) — Hanzo Agent SDK is a Python multi-agent framework — fork of OpenAI's Agents SDK with Hanzo AI integration.
- [`hanzo-analytics`](hanzo-analytics.md) — Hanzo Analytics is a privacy-focused web analytics platform -- an alternative to Google Analytics.
- [`hanzo-ane`](hanzo-ane.md) — Hanzo ANE enables training neural networks directly on Apple Neural Engine across ALL Apple Silicon (M1+).
- [`hanzo-app`](hanzo-app.md) — Hanzo App is an AI-powered developer platform that serves as both a web-based development environment (hanzo.app) and a macOS desktop assistant.
- [`hanzo-authorizenet`](hanzo-authorizenet.md) — AuthorizeNet-Go is a Go client library for the Authorize.net XML/JSON API.
- [`hanzo-base`](hanzo-base.md) — Hanzo Base is an open source Go backend that provides a complete application backend in a single binary.
- [`hanzo-billing`](hanzo-billing.md) — Hanzo Billing is a multi-org billing portal for managing subscriptions, payments, credits, invoices, and usage across the Hanzo ecosystem.
- [`hanzo-bot`](hanzo-bot.md) — Hanzo Bot (@hanzo/bot) is a TypeScript ESM multi-channel AI messaging gateway connecting AI agents to 50+ messaging platforms through a plugin/extension architecture.
- [`hanzo-brand`](hanzo-brand.md) — Brand assets, color palette, typography, and guidelines for the Hanzo ecosystem.
- [`hanzo-candle`](hanzo-candle.md) — Hanzo Candle is a Rust-based machine learning framework — intended fork of HuggingFace candle for high-performance ML inference and training with GPU acceleration (CUDA + Metal).
- [`hanzo-charts`](hanzo-charts.md) — Hanzo Charts is the official Helm chart repository for all Hanzo services and infrastructure.
- [`hanzo-chat`](hanzo-chat.md) — AI chat interface with multi-model support, MCP integration, agents, and RAG.
- [`hanzo-checkout`](hanzo-checkout.md) — Checkout.js is an embeddable checkout widget for taking payments and pre-orders on any website.
- [`hanzo-cli`](hanzo-cli.md) — Hanzo CLI is a hybrid Rust+Node+Python command-line tool — a single CLI that provides unified access to all Hanzo services.
- [`hanzo-cloud`](hanzo-cloud.md) — hanzoai/cloud is ONE Go binary that embeds every Hanzo-native subsystem — IAM, KMS, o11y, tasks, notify, pubsub, kafka, console, ai (the former LLM control plane), and the rest — behind a single plugin registry.
- [`hanzo-code`](hanzo-code.md) — Hanzo Code is an open-source AI-native code editor -- a fork of VS Code (microsoft/vscode) with integrated LLM chat, autocomplete, inline diffs, and agent tool use baked directly into the editor core.
- [`hanzo-commerce`](hanzo-commerce.md) — Hanzo Commerce is the core product of Hanzo AI -- a multi-tenant e-commerce platform that powers billing, payments, subscriptions, storefronts, and API usage metering for the entire Hanzo ecosystem and external customers.
- [`hanzo-commerce-api`](hanzo-commerce-api.md) — Hanzo Commerce is a Go/Gin e-commerce API with SQLite storage, providing products, orders, subscriptions, invoices, usage metering, and multi-currency support (fiat + crypto).
- [`hanzo-computer`](hanzo-computer.md) — Hanzo Computer (hanzoai/computer) is an ecommerce web application for purchasing AI compute hardware.
- [`hanzo-console`](hanzo-console.md) — Hanzo Console is the observability and prompt management layer for AI applications.
- [`hanzo-contracts`](hanzo-contracts.md) — Hanzo Contracts is the smart contract repository for the Hanzo AI ecosystem.
- [`hanzo-coursework-fullstack-go-cloud`](hanzo-coursework-fullstack-go-cloud.md) — This coursework trains engineering teams migrating from slow, interpreted, memory-heavy environments (Python, Next.js, Node.js, Ruby on Rails) to Hanzo's high-performance native stack (Go, Rust, C++).
- [`hanzo-database`](hanzo-database.md) — Hanzo Database covers PostgreSQL (with pgvector), Redis, MongoDB, and Hanzo S3 configurations used across the Hanzo ecosystem.
- [`hanzo-dataroom`](hanzo-dataroom.md) — Hanzo Dataroom is the open-source DocSend alternative -- a document sharing platform with built-in analytics, custom domains, and secure link-based access.
- [`hanzo-datastore`](hanzo-datastore.md) — Hanzo Datastore provides a unified vector database abstraction for AI applications — embedding storage, similarity search, and RAG retrieval.
- [`hanzo-deploy`](hanzo-deploy.md) — Hanzo deploys all services via Kubernetes manifests stored in the universe repo (hanzoai/universe).
- [`hanzo-desktop`](hanzo-desktop.md) — Hanzo Desktop is a cross-platform desktop application for creating and managing AI agents without code.
- [`hanzo-dev`](hanzo-dev.md) — Hanzo Dev is an AI-powered development platform -- a monorepo containing a Rust-native CLI (dev, dev-tui, dev-exec), a Node.js wrapper CLI (@hanzo/dev), a legacy TypeScript CLI (codex-cli), and supporting infrastructure including an app ser
- [`hanzo-dns`](hanzo-dns.md) — Hanzo DNS is a CoreDNS fork providing a programmable, plugin-based DNS server for Hanzo infrastructure.
- [`hanzo-docs`](hanzo-docs.md) — Hanzo Docs is a multi-brand documentation platform — fork of fumadocs with 27 packages and 15 apps.
- [`hanzo-documentdb`](hanzo-documentdb.md) — DocDB is a MongoDB-compatible document database server.
- [`hanzo-edge`](hanzo-edge.md) — Hanzo Edge is an on-device AI inference runtime for running Zen models and any GGUF model locally on macOS, Linux, Web (WASM), and embedded devices.
- [`hanzo-editor`](hanzo-editor.md) — Hanzo Editor (hanzoai/code) is an open-source AI-native code editor -- a VS Code fork (originally "Void Editor") with LLM chat, autocomplete, inline diffs, and agent tool use built directly into the editor core.
- [`hanzo-engine`](hanzo-engine.md) — Hanzo Engine is Hanzo AI's high-performance, Rust-native inference and embedding engine that powers the entire Hanzo ecosystem.
- [`hanzo-evm`](hanzo-evm.md) — Hanzo EVM is a Rust-based EVM execution engine — fork of paradigmxyz/reth v1.11.0, rebranded for the Hanzo ecosystem.
- [`hanzo-examples-native-cloud`](hanzo-examples-native-cloud.md) — This repository of code examples provides reference implementations for building, extending, and operating systems within the Hanzo ecosystem.
- [`hanzo-explorer`](hanzo-explorer.md) — Hanzo Explorer is an EVM blockchain explorer for inspecting and analyzing transactions, accounts, smart contracts, and tokens on Lux Network and other EVM-compatible chains.
- [`hanzo-extension`](hanzo-extension.md) — Hanzo Extensions is a monorepo of browser and IDE extensions providing AI-powered development and browsing tools.
- [`hanzo-extract`](hanzo-extract.md) — Hanzo Extract is a Rust crate for extracting clean text from web pages, PDFs, and Claude Code conversation logs.
- [`hanzo-faucet`](hanzo-faucet.md) — Hanzo Faucet is a token distribution service for Hanzo Network (the AI Compute L1 blockchain).
- [`hanzo-flow`](hanzo-flow.md) — Visual workflow builder for AI applications.
- [`hanzo-form`](hanzo-form.md) — Form.js is a lightweight browser library for automatic form validation and submission.
- [`hanzo-functions`](hanzo-functions.md) — Hanzo Functions is a Kubernetes-native serverless platform for event-driven workloads with GPU support.
- [`hanzo-gateway`](hanzo-gateway.md) — API gateway for all Hanzo services.
- [`hanzo-gochimp`](hanzo-gochimp.md) — GoChimp3 is a Go client library for the MailChimp API v3.0.
- [`hanzo-guard`](hanzo-guard.md) — Hanzo Guard is a Rust-based LLM I/O sanitization library and toolkit that sits between applications and LLM providers.
- [`hanzo-gym`](hanzo-gym.md) — Zoo Gym is the unified training infrastructure for all ZenLM AI models, built on LLaMA Factory.
- [`hanzo-hips`](hanzo-hips.md) — Hanzo Improvement Proposals (HIPs) are the primary mechanism for proposing new features, gathering community input, and documenting design decisions for the Hanzo AI ecosystem.
- [`hanzo-hke`](hanzo-hke.md) — HKE (Hanzo Kubernetes Engine) is a planned managed Kubernetes cluster service with built-in DevOps workflows.
- [`hanzo-iam`](hanzo-iam.md) — Hanzo IAM is the server-side identity and access management service for the Hanzo ecosystem.
- [`hanzo-id`](hanzo-id.md) — Hanzo ID is the client-side identity story for every Hanzo/Lux/Zoo/Pars service: how an app authenticates a user against the shared IAM server and scopes data to the user's org.
- [`hanzo-identity`](hanzo-identity.md) — Hanzo Identity (hanzoai/identity) is a decentralized identity (DID) registration system with Solidity smart contracts and a Next.js frontend.
- [`hanzo-ingress`](hanzo-ingress.md) — Hanzo Ingress is a Kubernetes-native L7 reverse proxy and load balancer that serves as the entry point for all Hanzo production traffic.
- [`hanzo-insights`](hanzo-insights.md) — Hanzo Insights is a full product analytics platform with product analytics, feature flags, session recording, A/B testing, heatmaps, LLM analytics, error tracking, surveys, web analytics, and a custom query language (InsightsQL).
- [`hanzo-jin`](hanzo-jin.md) — Jin is a research-stage visual self-supervised learning framework implementing Joint Embedding Predictive Architectures (JEPA).
- [`hanzo-k8s`](hanzo-k8s.md) — Hanzo runs all production services on DOKS (DigitalOcean Kubernetes Service) clusters.
- [`hanzo-kensho`](hanzo-kensho.md) — Kensho is a 17B parameter image generation foundation model using a Mixture of Experts (MoE) Diffusion Transformer architecture.
- [`hanzo-kms`](hanzo-kms.md) — Hanzo KMS is a centralized secret management platform for the Hanzo ecosystem.
- [`hanzo-koe`](hanzo-koe.md) — Koe is a 1.6B parameter text-to-speech model that directly generates realistic multi-speaker dialogue from text transcripts.
- [`hanzo-kv`](hanzo-kv.md) — Hanzo KV is a Redis-compatible in-memory key-value store used as the caching, streaming, and message broker layer across the Hanzo ecosystem.
- [`hanzo-kv-go`](hanzo-kv-go.md) — Go client library for Hanzo KV (Redis/Valkey compatible).
- [`hanzo-ledger`](hanzo-ledger.md) — Hanzo Ledger is a programmable double-entry financial ledger written in Go.
- [`hanzo-live`](hanzo-live.md) — Hanzo Live provides real-time AI streaming infrastructure for building responsive, collaborative AI applications.
- [`hanzo-llm-gateway`](hanzo-llm-gateway.md) — Hanzo LLM Gateway is a unified proxy that routes requests to 100+ LLM providers through a single OpenAI-compatible API.
- [`hanzo-log`](hanzo-log.md) — Hanzo Log is a high-performance, zero-allocation structured logging library for Go.
- [`hanzo-logs`](hanzo-logs.md) — Hanzo Logs is a horizontally-scalable, multi-tenant log aggregation system for Hanzo infrastructure.
- [`hanzo-mcp`](hanzo-mcp.md) — Hanzo MCP implements the Model Context Protocol -- an open standard for exposing tools, resources, and prompts to AI agents.
- [`hanzo-memory`](hanzo-memory.md) — Hanzo Memory (@hanzo/memory) is a TypeScript AI memory service that stores, searches, and retrieves contextual memories using vector embeddings.
- [`hanzo-metrics`](hanzo-metrics.md) — Hanzo Metrics is a fast, cost-effective time-series database for monitoring AI infrastructure.
- [`hanzo-ml`](hanzo-ml.md) — Hanzo ML is a Rust-based minimalist ML framework forked from HuggingFace Candle, optimized for edge AI, quantization, and multimodal inference.
- [`hanzo-mpc`](hanzo-mpc.md) — Hanzo MPC is a threshold signing service for securely generating and managing cryptographic wallets across distributed nodes without ever exposing the full private key.
- [`hanzo-mugen`](hanzo-mugen.md) — Mugen is a PyTorch framework for deep learning research on audio generation.
- [`hanzo-nchain`](hanzo-nchain.md) — Nchain is a Kubernetes operator for deploying and scaling blockchain nodes.
- [`hanzo-network`](hanzo-network.md) — Hanzo Network enables distributed AI agent execution across heterogeneous devices (M1 MacBooks, NVIDIA GPUs, Linux servers, Raspberry Pis, etc.) with automatic peer discovery, device capability detection, and intelligent load balancing.
- [`hanzo-node`](hanzo-node.md) — Hanzo Node is a Rust-based AI agent infrastructure node that lets you create AI agents without writing code.
- [`hanzo-numscript`](hanzo-numscript.md) — Numscript is a domain-specific language for modeling complex financial transactions.
- [`hanzo-o11y`](hanzo-o11y.md) — Hanzo O11y is the full-stack observability platform for logs, metrics, and traces.
- [`hanzo-openapi`](hanzo-openapi.md) — Hanzo OpenAPI is the canonical OpenAPI 3.1.0 specification for all 26 Hanzo Cloud services.
- [`hanzo-operative`](hanzo-operative.md) — Hanzo Operative (hanzo-operative) is an autonomous computer use agent that enables AI to control desktop environments via screenshots, mouse, keyboard, and bash commands.
- [`hanzo-operator`](hanzo-operator.md) — Hanzo Operator is a unified Kubernetes operator that manages all Hanzo production infrastructure declaratively via 7 CRDs under the hanzo.ai/v1alpha1 API group.
- [`hanzo-orm`](hanzo-orm.md) — Hanzo ORM is a Go generics-based ORM with automatic serialization, KV caching, and zero code generation.
- [`hanzo-payments`](hanzo-payments.md) — Hanzo Payments is an open-source payment orchestration switch written in Rust.
- [`hanzo-platform`](hanzo-platform.md) — Hanzo Platform is a Kubernetes-native PaaS (Platform as a Service) for deploying AI applications, APIs, and services.
- [`hanzo-playground`](hanzo-playground.md) — Hanzo Playground is a Kubernetes-style control plane for AI bots.
- [`hanzo-pubsub`](hanzo-pubsub.md) — Hanzo PubSub is a high-performance messaging system for pub/sub, persistent streams, and exactly-once delivery.
- [`hanzo-pubsub-go`](hanzo-pubsub-go.md) — Go client library for Hanzo PubSub (NATS compatible).
- [`hanzo-reconciliation`](hanzo-reconciliation.md) — Hanzo Reconciliation is an automated transaction matching engine that compares ledger balances against payment provider data to verify financial consistency and identify discrepancies.
- [`hanzo-registry`](hanzo-registry.md) — Hanzo Registry is a private Docker container registry running Docker Distribution (registry:2) on hanzo-k8s, authenticated via Hanzo IAM token-based auth.
- [`hanzo-relay`](hanzo-relay.md) — Hanzo Relay (crate name: hanzo-relay, internally hanzo-tunnel) is a Rust library that connects any local Hanzo app (dev, node, desktop, bot, extension) to the cloud relay at api.hanzo.ai for remote management and multiplayer.
- [`hanzo-repl`](hanzo-repl.md) — Hanzo REPL is a Python interactive environment for testing Hanzo MCP tools and AI integration.
- [`hanzo-sdk`](hanzo-sdk.md) — Hanzo SDK is a unified multi-language SDK and CLI (@hanzo/cli) that wraps Python, TypeScript, Rust, and Go implementations into a single hanzo command.
- [`hanzo-search`](hanzo-search.md) — Hanzo Search is a high-performance search engine built as a Rust workspace with 22 crates.
- [`hanzo-search-go`](hanzo-search-go.md) — Go client library for Hanzo Search (Meilisearch compatible).
- [`hanzo-sentry`](hanzo-sentry.md) — Hanzo Sentry is a self-hosted Sentry 24.2.0 deployment for error tracking and performance monitoring across all Hanzo, Lux, Zoo, and Pars services.
- [`hanzo-sho`](hanzo-sho.md) — Sho is a text diffusion engine that generates text using masked diffusion rather than autoregressive token prediction.
- [`hanzo-sign`](hanzo-sign.md) — Hanzo Sign is an open-source DocuSign alternative for electronic document signing.
- [`hanzo-skill`](hanzo-skill.md) — Hanzo Skill (@hanzo/skill) is a TypeScript CLI tool for installing, managing, and distributing AI agent skills.
- [`hanzo-sql`](hanzo-sql.md) — Hanzo SQL is a PostgreSQL fork with pre-built extensions for AI workloads: pgvector (vector similarity), pgcron (scheduled jobs), pgdocumentdb (MongoDB wire protocol), and PostGIS (geospatial).
- [`hanzo-stack`](hanzo-stack.md) — Hanzo Stack provides the complete integrated development environment for running all Hanzo services locally.
- [`hanzo-static`](hanzo-static.md) — Hanzo Static is a Traefik plugin that serves static files (SPAs, assets) directly from the ingress controller.
- [`hanzo-storage`](hanzo-storage.md) — Hanzo Storage is a high-performance, S3-compatible object storage server for AI workloads.
- [`hanzo-store`](hanzo-store.md) — Hanzo Store is a flatfile-based marketplace for discovering and installing AI agent tools and MCP servers.
- [`hanzo-stream`](hanzo-stream.md) — Hanzo Stream is a stateless Kafka wire protocol gateway that translates standard Kafka client requests into NATS JetStream operations against Hanzo PubSub.
- [`hanzo-studio`](hanzo-studio.md) — Hanzo Studio is a visual node-based AI workflow engine for building, testing, and deploying AI pipelines.
- [`hanzo-suite`](hanzo-suite.md) — Hanzo Business Suite is a fork of Odoo 18.0 (Community Edition) providing a full-featured open-source ERP/CRM platform tightly integrated with Hanzo AI infrastructure.
- [`hanzo-team`](hanzo-team.md) — Hanzo Team is the collaboration platform at hanzo.team.
- [`hanzo-terraform`](hanzo-terraform.md) — The Hanzo Terraform Provider is a generated Terraform SDK for managing Hanzo AI resources via Infrastructure as Code.
- [`hanzo-tools`](hanzo-tools.md) — Hanzo Tools is a unified registry of 88 tools across 20 categories for all AI agent operations across the Hanzo and Zoo ecosystems.
- [`hanzo-treasury`](hanzo-treasury.md) — Hanzo Treasury is a programmable financial infrastructure stack providing a double-entry ledger, payment connectors, virtual wallets, reconciliation, and workflow orchestration.
- [`hanzo-trpc-openapi`](hanzo-trpc-openapi.md) — @hanzo/trpc-openapi is a TypeScript library that generates OpenAPI 3.x documents from tRPC v11 routers and provides HTTP adapters to serve tRPC procedures as REST endpoints.
- [`hanzo-tunnel`](hanzo-tunnel.md) — Hanzo Tunnel is a Rust library and Python agent bridge that connects local Hanzo app instances (dev machines, bots, nodes) to the cloud control plane at app.hanzo.bot.
- [`hanzo-tutorial-realtime-sse-backend`](hanzo-tutorial-realtime-sse-backend.md) — In this tutorial, you will build a production-grade, high-concurrency realtime backend in Go using Hanzo Base (github.com/hanzoai/base).
- [`hanzo-ui`](hanzo-ui.md) — Hanzo UI is a React component library forked from shadcn/ui, extended with 161+ components (3x upstream), two themes, multi-framework support, AI components, 3D components, and a visual page builder.
- [`hanzo-universe`](hanzo-universe.md) — Hanzo Universe (github.com/hanzoai/universe) is the single source of truth for all production infrastructure across the Hanzo ecosystem.
- [`hanzo-vault`](hanzo-vault.md) — Hanzo Vault is a card tokenization service that runs as an isolated cardholder data environment (CDE).
- [`hanzo-vector`](hanzo-vector.md) — Hanzo Vector is a high-performance vector search engine for the Hanzo AI platform.
- [`hanzo-vector-go`](hanzo-vector-go.md) — Go client library for Hanzo Vector (Qdrant compatible).
- [`hanzo-vllm`](hanzo-vllm.md) — Hanzo vLLM is a Rust LLM inference server with an OpenAI-compatible API, PagedAttention, continuous batching, and multi-GPU/multi-node support.
- [`hanzo-vm`](hanzo-vm.md) — Hanzo VM is an open-source cloud operating system and virtual machine management platform built with Go (Beego framework) and React.
- [`hanzo-web3`](hanzo-web3.md) — Hanzo Web3 is the blockchain infrastructure layer of the Hanzo ecosystem.
- [`hanzo-web3-gateway`](hanzo-web3-gateway.md) — Hanzo Web3 Gateway provides keyless blockchain access for autonomous AI agents.
- [`hanzo-website`](hanzo-website.md) — The Hanzo AI corporate website at hanzo.ai.
- [`hanzo-zap`](hanzo-zap.md) — ZAP is THE Hanzo transport (HIP-0114): the zero-copy protocol every Hanzo service speaks — subsystem↔subsystem inside the cloud binary (ZAP-typed Go interfaces in-process), host↔plugin-VM over the wire (ZAP RPC, HIP-0116), and service↔datab
- [`hanzo-zrok`](hanzo-zrok.md) — Hanzo Zrok is a zero-trust sharing platform built on top of Hanzo ZT (OpenZiti).
- [`hanzo-zt`](hanzo-zt.md) — Hanzo ZT is the zero-trust network overlay fabric for Hanzo's secure networking.
- [`js-sdk`](js-sdk.md) — Hanzo JS SDK is a TypeScript-first client library for the Hanzo AI API.
- [`python-sdk`](python-sdk.md) — Hanzo Python SDK is a generated Python client library for the Hanzo AI API.
- [`rust-sdk`](rust-sdk.md) — Hanzo Rust SDK is a full Rust infrastructure SDK — not just an API client.
- [`skill`](skill.md) — Onboarding manifest for an AI agent operating Hanzo — native code architecture (Go, Rust, C++), local-first cloud copy, backend migration with Hanzo Base realtime SSE, and working the AI cloud safely.
- [`zenlm`](zenlm.md) — ZenLM is a groundbreaking collaboration between Hanzo AI(https://hanzo.ai) (Techstars-backed GenAI lab) and Zoo Labs Foundation(https://zoo.ngo) (501(c)(3) environmental non-profit), building AI models that run entirely on your device—no cl

### core
- [`academy-guide`](academy-guide.md) — When a user asks a question about Claude, a Claude product, or a general "how do I use AI for X" question, check the Academy catalog (see "The catalog" below) for a strong match.
- [`algorithmic-art`](algorithmic-art.md) — Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use this when users request creating art using code, generative art, algorithmic art, flow fields, or particle systems. Create original algorithmic art rather than copying existing artists' work to avoid copyright violations.
- [`anti-slop`](anti-slop.md) — Comprehensive toolkit for detecting and eliminating "AI slop" - generic, low-quality AI-generated patterns in natural language, code, and design. Use when reviewing or improving content quality, preventing generic AI patterns, cleaning up existing content, or enforcing quality standards in writing, code, or design work.
- [`api`](api.md) — Automatically discover API design skills when working with REST APIs, GraphQL schemas, API authentication, OAuth, JWT, rate limiting, API versioning, error handling, or endpoint design. Activates for backend API development tasks.
- [`app-port`](app-port.md) — Port an existing application onto the Hanzo stack — Hanzo Base and Hanzo Cloud OSS for the backend, ZAP for transport, @hanzo/ui and @hanzo/gui for the interface — and move its Python or JavaScript runtime into Go, Rust or C++. Use when adopting a Next.js, Node, Django, Flask, FastAPI, Rails or Spring app; when a service still speaks WebSockets; when a deployment still needs an interpreter beside the binary; or when asked to modernise, migrate or absorb an app somebody else's stack currently runs.
- [`brand-guidelines`](brand-guidelines.md) — Applies Anthropic's official brand colors and typography to any sort of artifact that may benefit from having Anthropic's look-and-feel. Use it when brand colors or style guidelines, visual formatting, or company design standards apply.
- [`build-systems`](build-systems.md) — Automatically discover build system skills when working with build systems. Activates for build-systems development tasks.
- [`caching`](caching.md) — Automatically discover caching and performance skills when working with caching. Activates for caching development tasks.
- [`canvas-design`](canvas-design.md) — Create beautiful visual art in .png and .pdf documents using design philosophy. You should use this skill when the user asks to create a poster, piece of art, design, or other static piece. Create original visual designs, never copying existing artists' work to avoid copyright violations.
- [`cicd`](cicd.md) — Automatically discover CI/CD and automation skills when working with CI/CD. Activates for cicd development tasks.
- [`claude-api`](claude-api.md) — This skill helps you build LLM-powered applications with Claude.
- [`cloud`](cloud.md) — Automatically discover cloud computing and serverless skills when working with cloud. Activates for cloud development tasks.
- [`cloud-host`](cloud-host.md) — Run the Hanzo cloud on your own machine — the local API, the local cluster, ports, keys, and what already exists so you do not rebuild it. Use when self-hosting Hanzo, standing up a development environment, running the cloud from a source checkout, deciding between the local API and a local cluster, wiring KMS or IAM locally, or when someone has written their own launcher scripts because they could not find ours.
- [`cloud-kubernetes-deployment`](cloud-kubernetes-deployment.md) — Deploy and manage applications on Kubernetes with production best practices
- [`collaboration`](collaboration.md) — Automatically discover collaboration and teamwork skills when working with collaboration. Activates for collaboration development tasks.
- [`containers`](containers.md) — Automatically discover container and Docker skills when working with Docker. Activates for containers development tasks.
- [`cryptography`](cryptography.md) — Automatically discover cryptography skills when working with encryption, TLS, certificates, PKI, and security
- [`data`](data.md) — Automatically discover data pipeline and ETL skills when working with ETL. Activates for data development tasks.
- [`database`](database.md) — Automatically discover database skills when working with SQL, PostgreSQL, MongoDB, Redis, database schema design, query optimization, migrations, connection pooling, ORMs, or database selection. Activates for database design, optimization, and implementation tasks.
- [`debugging`](debugging.md) — Automatically discover debugging and profiling skills when working with debugging. Activates for debugging development tasks.
- [`deployment`](deployment.md) — Automatically discover deployment and release skills when working with deployment. Activates for deployment development tasks.
- [`diagrams`](diagrams.md) — Automatically discover diagram and visualization skills when working with flowcharts, sequence diagrams, architecture diagrams, ER diagrams, Gantt charts, mindmaps, or any visual documentation using Mermaid.js. Activates for documentation and visualization tasks.
- [`discernment-nudge`](discernment-nudge.md) — People often take an AI answer at face value, especially when it's confidently written and well-structured.
- [`distributed-systems`](distributed-systems.md) — Automatically discover distributed systems skills when working with consensus, CRDTs, replication, partitioning, and distributed algorithms
- [`ebpf`](ebpf.md) — Automatically discover eBPF and kernel skills when working with eBPF. Activates for ebpf development tasks.
- [`elegant-design`](elegant-design.md) — Create world-class, accessible, responsive interfaces with sophisticated interactive elements including chat, terminals, code display, and streaming content. Use when building user interfaces that need professional polish and developer-focused features.
- [`engineering`](engineering.md) — Automatically discover software engineering practice skills when working with engineering practices. Activates for engineering development tasks.
- [`formal`](formal.md) — Automatically discover formal methods and verification skills when working with formal methods. Activates for formal development tasks.
- [`frontend`](frontend.md) — Automatically discover frontend development skills when working with React, Next.js, UI components, state management, data fetching, forms, accessibility, performance optimization, or SEO. Activates for frontend web development tasks.
- [`frontend-design`](frontend-design.md) — Guidance for distinctive, intentional visual design when building new UI or reshaping an existing one. Helps with aesthetic direction, typography, and making choices that don't read as templated defaults.
- [`hanzo`](hanzo.md) — Automatically discover the Hanzo stack — local-first models, the MCP tooling, the SDKs and the cluster pieces — and which part answers a given need. Activates when a task names Hanzo, local AI, MCP or an agentic workflow.
- [`hanzo-api`](hanzo-api.md) — Automatically discover how to CALL Hanzo's own products over api.hanzo.ai — models, agents, sandboxes, secrets, billing, deployments, identity, telemetry. Activates when a task needs a Hanzo API rather than code written from scratch: o11y, ai, commerce, iam, git, project, admin, integration, agent, platform, marketing, visor, pricing, cloudflare, captable, provisioning, book, company, deploy, task, crm, team, framework, guide, tool, sandbox, automation, ingress.
- [`hanzo-cloud-architecture`](hanzo-cloud-architecture.md) — The canonical Hanzo Cloud architecture — one Go binary on zip, ZAP-only transport, plugin registry, embedded console, three deployment modes. Read this before touching any hanzoai service.
- [`infrastructure`](infrastructure.md) — Automatically discover infrastructure and DevOps skills when working with infrastructure. Activates for infrastructure development tasks.
- [`internal-comms`](internal-comms.md) — A set of resources to help me write all kinds of internal communications, using the formats that my company likes to use. Claude should use this skill whenever asked to write some sort of internal communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident reports, project updates, etc.).
- [`ir`](ir.md) — Automatically discover intermediate representation and compiler skills when working with intermediate representation. Activates for ir development tasks.
- [`math`](math.md) — Automatically discover mathematics and algorithm skills when working with mathematics. Activates for math development tasks.
- [`mcp-builder`](mcp-builder.md) — Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK).
- [`ml`](ml.md) — Automatically discover machine learning and AI skills when working with machine learning. Activates for ml development tasks.
- [`mobile`](mobile.md) — Automatically discover mobile development skills when working with iOS. Activates for mobile development tasks.
- [`modal`](modal.md) — Automatically discover Modal platform skills when working with Modal. Activates for modal development tasks.
- [`networking`](networking.md) — Automatically discover networking and connectivity skills when working with networking. Activates for networking development tasks.
- [`observability`](observability.md) — Automatically discover observability and monitoring skills when working with monitoring. Activates for observability development tasks.
- [`plt`](plt.md) — Automatically discover programming language theory skills when working with compilers. Activates for plt development tasks.
- [`product`](product.md) — Automatically discover product management skills when working with product management. Activates for product development tasks.
- [`protocols`](protocols.md) — Automatically discover protocol skills when working with HTTP, TCP, UDP, QUIC, and network protocols
- [`proxies`](proxies.md) — Automatically discover proxy skills when working with forward/reverse proxies, nginx, traefik, envoy, caching, or NATS messaging
- [`realtime`](realtime.md) — Automatically discover realtime communication skills when working with realtime. Activates for realtime development tasks.
- [`research`](research.md) — Gateway skill for research methodology - automatically loads relevant research skills based on task context
- [`security`](security.md) — Automatically discover security skills when working with authentication, authorization, input validation, security headers, vulnerability assessment, or secrets management. Activates for application security, OWASP, and security hardening tasks.
- [`skill-creation`](skill-creation.md) — Creating new atomic skills for the skills system
- [`skill-creator`](skill-creator.md) — Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.
- [`skill-prompt-discovery`](skill-prompt-discovery.md) — User makes a request involving specific technologies or frameworks
- [`skill-prompt-planning`](skill-prompt-planning.md) — This skill enables analyzing user prompts, requests, and conversation patterns to identify gaps in the skills catalog and plan creation of new skills that would be genuinely useful.
- [`skill-repo-discovery`](skill-repo-discovery.md) — Starting work on an unfamiliar repository or codebase
- [`skill-repo-planning`](skill-repo-planning.md) — This skill enables systematic analysis of codebases to identify skill gaps—domains, technologies, patterns, and workflows present in repositories that aren't covered by existing skills.
- [`slack-gif-creator`](slack-gif-creator.md) — Knowledge and utilities for creating animated GIFs optimized for Slack. Provides constraints, validation tools, and animation concepts. Use when users request animated GIFs for Slack like "make me a GIF of X doing Y for Slack.
- [`testing`](testing.md) — Automatically discover testing skills when working with testing. Activates for testing development tasks.
- [`theme-factory`](theme-factory.md) — Toolkit for styling artifacts with a theme. These artifacts can be slides, docs, reportings, HTML landing pages, etc. There are 10 pre-set themes with colors/fonts that you can apply to any artifact that has been creating, or can generate a new theme on-the-fly.
- [`tui`](tui.md) — Automatically discover terminal UI skills when working with terminal user interfaces. Activates for tui development tasks.
- [`typed-holes-refactor`](typed-holes-refactor.md) — Refactor codebases using Design by Typed Holes methodology - iterative, test-driven refactoring with formal hole resolution, constraint propagation, and continuous validation. Use when refactoring existing code, optimizing architecture, or consolidating technical debt through systematic hole-driven development.
- [`ui-port`](ui-port.md) — Port or migrate a UI from an old/reference implementation to a new system while preserving new-system functionality and achieving high visual parity. Use when comparing reference and target URLs, routes, repos/worktrees, Storybook stories, screenshots, PDFs, DOM/computed-style probes, or any combination; when asked to make a rewrite look like an older UI; when a visual regression is "close" but still reads as a different product; or when agents need a repeatable audit-fix-measure loop for screen-by-screen convergence.
- [`wasm`](wasm.md) — Automatically discover WebAssembly skills when working with WebAssembly. Activates for wasm development tasks.
- [`web-artifacts-builder`](web-artifacts-builder.md) — Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind CSS, shadcn/ui). Use for complex artifacts requiring state management, routing, or shadcn/ui components - not for simple single-file HTML/JSX artifacts.
- [`webapp-testing`](webapp-testing.md) — Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs.
- [`workflow`](workflow.md) — Automatically discover workflow and task management skills when working with Beads. Activates for workflow development tasks.
- [`zig`](zig.md) — Automatically discover Zig programming skills when working with Zig. Activates for zig development tasks.
- [`zoo-gym`](zoo-gym.md) — Three training methods from Zoo Gym — Training-Free GRPO (RLHF with no value network), BitDelta per-user personalisation, and DeltaSoup Byzantine-robust community learning. Use when training or personalising a ZenLM model, or choosing an RLHF method.

### claude-api
- [`agent-design`](agent-design.md) — This file covers decision heuristics for building agents on the Claude API: which primitives to reach for, how to design your tool surface, and how to manage context and cost over long runs.
- [`anthropic-cli`](anthropic-cli.md) — The ant CLI exposes every Claude API resource as a shell subcommand.
- [`batches`](batches.md) — The Batches API (POST /v1/messages/batches) processes Messages API requests asynchronously at 50% of standard prices.
- [`claude-platform-on-aws`](claude-platform-on-aws.md) — Anthropic-operated access to the Claude Developer Platform through AWS infrastructure — SigV4 authentication, AWS IAM access control, and AWS Marketplace billing.
- [`error-codes`](error-codes.md) — This file documents HTTP error codes returned by the Claude API, their common causes, and how to handle them.
- [`examples`](examples.md) — Use these examples when the user needs raw HTTP requests or is working in a language without an official SDK.
- [`files-api`](files-api.md) — Under client.Beta.Files.
- [`live-sources`](live-sources.md) — This file contains WebFetch URLs for fetching current information from platform.claude.com and Agent SDK repositories.
- [`managed-agents`](managed-agents.md) — Use these examples when the user needs raw HTTP requests or is working without an SDK.
- [`managed-agents-api-reference`](managed-agents-api-reference.md) — All endpoints require x-api-key and anthropic-version: 2023-06-01 headers.
- [`managed-agents-client-patterns`](managed-agents-client-patterns.md) — Patterns you'll write on the client side when driving a Managed Agent session, grounded in working SDK examples.
- [`managed-agents-core`](managed-agents-core.md) — Managed Agents is built around four core concepts:
- [`managed-agents-environments`](managed-agents-environments.md) — Creating a session requires an environmentid.
- [`managed-agents-events`](managed-agents-events.md) — Send events to a session via POST /v1/sessions/{id}/events.
- [`managed-agents-memory`](managed-agents-memory.md) — Sessions are ephemeral by default — when one ends, anything the agent learned is gone.
- [`managed-agents-multiagent`](managed-agents-multiagent.md) — A coordinator agent can delegate to other agents within one session.
- [`managed-agents-onboarding`](managed-agents-onboarding.md) — Claude Managed Agents is a hosted agent: Anthropic runs the agent loop and provisions a sandboxed container per session where the agent's tools execute (or your own worker, with a selfhosted environment — see shared/managed-agents-self-host
- [`managed-agents-outcomes`](managed-agents-outcomes.md) — An outcome elevates a session from conversation to work: you state what "done" looks like, and the harness runs an iterate → grade → revise loop until the artifact meets the rubric, hits maxiterations, or is interrupted.
- [`managed-agents-overview`](managed-agents-overview.md) — Managed Agents provisions a container per session as the agent's workspace.
- [`managed-agents-scheduled-deployments`](managed-agents-scheduled-deployments.md) — A scheduled deployment runs an agent on a recurring cron schedule — each firing creates a session autonomously.
- [`managed-agents-self-hosted-sandboxes`](managed-agents-self-hosted-sandboxes.md) — With config.type: "selfhosted", the agent loop stays on Anthropic's orchestration layer but tool execution moves to infrastructure you control — bash, file ops, and code run inside your container, so filesystem contents and network egress n
- [`managed-agents-tools`](managed-agents-tools.md) — Recommendation: Enable all prebuilt tools via agenttoolset20260401, then disable individually as needed.
- [`managed-agents-webhooks`](managed-agents-webhooks.md) — Anthropic can POST to your HTTPS endpoint when a Managed Agents resource changes state — an alternative to holding an SSE stream or polling.
- [`model-migration`](model-migration.md) — How to move existing code to newer Claude models.
- [`models`](models.md) — Only use exact model IDs listed in this file.
- [`platform-availability`](platform-availability.md) — Which features work on which provider platform.
- [`prompt-audit`](prompt-audit.md) — Prompts, skills, and tool descriptions accumulate instructions tuned to older models: emphasis added because an old model under-triggered, step-by-step scripts added because an old model planned poorly, format scaffolds written before the A
- [`prompt-caching`](prompt-caching.md) — This file covers how to design prompt-building code for effective caching.
- [`sdk-upgrade`](sdk-upgrade.md) — anthropic 1.x is deliberately a small step from the last 0.x release: no method was restructured and no new pattern is required.
- [`streaming`](streaming.md) — for stream.Next() { event := stream.Current() switch eventVariant := event.AsAny().(type) { case anthropic.ContentBlockDeltaEvent: switch deltaVariant := eventVariant.Delta.AsAny().(type) { case anthropic.TextDelta: fmt.Print(deltaVariant.T
- [`token-counting`](token-counting.md) — Use the counttokens endpoint (POST /v1/messages/counttokens) for accurate token counts against Claude models.
- [`tool-use`](tool-use.md) — For conceptual overview (tool definitions, tool choice, tips), see shared/tool-use-concepts.md(../../shared/tool-use-concepts.md).
- [`tool-use-concepts`](tool-use-concepts.md) — This file covers the conceptual foundations of tool use with the Claude API.

### engineering
- [`blameless-postmortem`](blameless-postmortem.md) — Date: 2025-10-27 Incident Commander: Alice Smith (@alice) Severity: Sev2 Duration: 45 minutes (14:23 - 15:08 UTC) Impact: ~15,000 users experienced errors (10% of active users) Error Budget Impact: 15% of monthly budget consumed
- [`database-connection-pool-exhausted`](database-connection-pool-exhausted.md) — Service: API Service Component: Database Layer Owner: Database Team (@database-team) Last Updated: 2025-10-27 Severity: Usually SEV-2, can escalate to SEV-1
- [`engineering-capacity-planning`](engineering-capacity-planning.md) — Comprehensive capacity planning including forecasting, resource modeling, load testing, scaling strategies, cost optimization, and disaster recovery planning for production systems
- [`engineering-ci-cd-pipelines`](engineering-ci-cd-pipelines.md) — Comprehensive CI/CD pipeline design covering build, test, security, deployment automation, artifact management, and multi-platform implementation (GitHub Actions, GitLab CI, Jenkins, CircleCI)
- [`engineering-code-quality`](engineering-code-quality.md) — Code quality metrics, maintainability principles, SOLID design, code smells detection, and quality measurement
- [`engineering-code-review`](engineering-code-review.md) — Code review practices, PR etiquette, constructive feedback, automation tools, and effective review workflows
- [`engineering-continuous-integration`](engineering-continuous-integration.md) — CI/CD pipeline design, automated testing, deployment strategies, and continuous delivery best practices
- [`engineering-deployment-strategies`](engineering-deployment-strategies.md) — Production deployment strategies including blue-green, canary, rolling updates, zero-downtime deployments, and database migrations
- [`engineering-design-patterns`](engineering-design-patterns.md) — Gang of Four design patterns, when to use each pattern, implementation examples, and anti-patterns
- [`engineering-domain-driven-design`](engineering-domain-driven-design.md) — Domain-Driven Design patterns, bounded contexts, aggregates, entities, value objects, and strategic design
- [`engineering-functional-programming`](engineering-functional-programming.md) — Functional programming principles, immutability, pure functions, higher-order functions, and FP patterns
- [`engineering-incident-response`](engineering-incident-response.md) — Comprehensive incident response practices including lifecycle management, severity classification, war rooms, runbooks, blameless postmortems, and SRE metrics (MTTR, MTTD, MTTA, MTBF)
- [`engineering-monitoring-alerts`](engineering-monitoring-alerts.md) — Production-ready monitoring and alerting strategies
- [`engineering-pair-programming`](engineering-pair-programming.md) — Pair programming techniques, driver-navigator roles, mob programming, remote pairing tools and best practices
- [`engineering-refactoring-patterns`](engineering-refactoring-patterns.md) — Refactoring techniques, when to refactor, safe refactoring strategies, and code improvement patterns
- [`engineering-rfc-consensus-building`](engineering-rfc-consensus-building.md) — Stakeholder identification, feedback collection, consensus building, and approval processes for RFCs
- [`engineering-rfc-decision-documentation`](engineering-rfc-decision-documentation.md) — Architecture Decision Records (ADRs), decision rationale, status tracking, and post-implementation review
- [`engineering-rfc-structure-format`](engineering-rfc-structure-format.md) — RFC document structure, templates, formatting conventions, and versioning for technical design documents
- [`engineering-rfc-technical-design`](engineering-rfc-technical-design.md) — Architecture proposals, technical diagrams, API design, trade-off analysis, and migration strategies for RFCs
- [`engineering-technical-debt`](engineering-technical-debt.md) — Identifying, measuring, tracking, and managing technical debt strategically and systematically
- [`engineering-test-driven-development`](engineering-test-driven-development.md) — Test-Driven Development methodology, red-green-refactor cycle, unit testing best practices, and mocking strategies
- [`error-budget-policy`](error-budget-policy.md) — Service: Payment API Owner: Payments Team Last Updated: 2025-10-27 Version: 2.0
- [`flag_retirement_workflow`](flag_retirement_workflow.md) — Production-ready workflow for safely retiring feature flags with checklists, timelines, and automation.
- [`high-memory-usage`](high-memory-usage.md) — This alert fires when memory usage exceeds 90% for more than 5 minutes on any node.
- [`on-call-runbook`](on-call-runbook.md) — Quick reference for on-call engineers supporting the Payment API.
- [`performance-review`](performance-review.md) — Use this checklist when reviewing code for performance issues and optimization opportunities.
- [`pull_request_template`](pull_request_template.md) — -   Bug fix (non-breaking change fixing an issue) -   New feature (non-breaking change adding functionality) -   Breaking change (fix or feature that breaks existing functionality) -   Refactoring (no functional changes, just code restructu
- [`security-review`](security-review.md) — Use this checklist when reviewing code for security concerns.
- [`sev1-incident-template`](sev1-incident-template.md) — This template provides a structured approach for managing critical (SEV-1) incidents.
- [`sev1-response-workflow`](sev1-response-workflow.md) — This document provides a complete walkthrough of responding to a SEV-1 (critical) incident from detection to postmortem.
- [`tdd-workflow`](tdd-workflow.md) — Complete guide to practicing Test-Driven Development from project setup through delivery.
- [`template`](template.md) — Brief description of what this alert means and why it exists.

### ml
- [`dspy-assertions`](dspy-assertions.md) — Adding constraints and validation to DSPy programs with assertions and suggestions
- [`dspy-evaluation`](dspy-evaluation.md) — Evaluating DSPy programs with metrics, Evaluate class, and performance analysis
- [`dspy-modules`](dspy-modules.md) — Building composable prediction modules with Predict, ChainOfThought, ReAct, and custom modules
- [`dspy-optimizers`](dspy-optimizers.md) — Optimizing DSPy programs with teleprompters, BootstrapFewShot, MIPROv2, and compilation
- [`dspy-rag`](dspy-rag.md) — Building Retrieval-Augmented Generation pipelines with DSPy and vector databases
- [`dspy-setup`](dspy-setup.md) — Installation, configuration, and language model setup for DSPy framework including Modal and HuggingFace
- [`dspy-signatures`](dspy-signatures.md) — Defining input/output signatures for DSPy modules and language model tasks
- [`graph-rag`](graph-rag.md) — Graph-based retrieval-augmented generation with entity extraction, community detection, and multihop reasoning
- [`hierarchical-rag`](hierarchical-rag.md) — Multi-level retrieval with recursive summarization and parent-child document structures for improved context selection
- [`hybrid-search-rag`](hybrid-search-rag.md) — Combining vector search with BM25 sparse retrieval using fusion algorithms for improved RAG accuracy
- [`ml-custom-llm-evaluation`](ml-custom-llm-evaluation.md) — Domain-specific LLM evaluation including custom metrics, RLHF evaluation, adversarial testing, bias detection, Arize Phoenix integration, and continuous evaluation patterns for production systems
- [`ml-diffusion-finetuning`](ml-diffusion-finetuning.md) — Fine-tuning diffusion models for custom styles or subjects
- [`ml-diffusion-model-basics`](ml-diffusion-model-basics.md) — Understanding diffusion model theory and mechanics (forward/reverse process)
- [`ml-huggingface-autotrain`](ml-huggingface-autotrain.md) — Need quick fine-tuning without writing training code
- [`ml-huggingface-datasets`](ml-huggingface-datasets.md) — Loading, processing, and managing datasets with HuggingFace datasets library
- [`ml-huggingface-hub`](ml-huggingface-hub.md) — Managing models, datasets, and Spaces on HuggingFace Hub
- [`ml-huggingface-spaces`](ml-huggingface-spaces.md) — Deploying Gradio and Streamlit apps on HuggingFace Spaces
- [`ml-huggingface-transformers`](ml-huggingface-transformers.md) — Loading models, pipelines, inference, and tokenization with transformers library
- [`ml-llm-as-judge`](ml-llm-as-judge.md) — Comprehensive guide to LLM-as-judge evaluation patterns including Prometheus 2 models, G-Eval framework, pairwise/pointwise/reference-guided methods, bias mitigation, and uncertainty quantification
- [`ml-llm-benchmarks-evaluation`](ml-llm-benchmarks-evaluation.md) — Comprehensive guide to evaluating LLMs using standard benchmarks including MMLU, HellaSwag, BBH, HumanEval, TruthfulQA, and GSM8K, with practical implementation using LightEval and lm-evaluation-harness
- [`ml-llm-dataset-preparation`](ml-llm-dataset-preparation.md) — Preparing training data for LLM fine-tuning
- [`ml-llm-evaluation-frameworks`](ml-llm-evaluation-frameworks.md) — Production-grade LLM evaluation using Arize Phoenix (OpenTelemetry tracing, self-hosted evals), Braintrust (86x faster search), LangSmith, and Langfuse with cost tracking and observability integration
- [`ml-llm-model-routing`](ml-llm-model-routing.md) — Intelligent LLM routing with RouteLLM, semantic routing, and cost optimization
- [`ml-llm-model-selection`](ml-llm-model-selection.md) — Choosing the right LLM based on capabilities, benchmarks, pricing, and use cases
- [`ml-lora-peft-techniques`](ml-lora-peft-techniques.md) — Fine-tuning large models with limited GPU memory
- [`ml-multi-model-orchestration`](ml-multi-model-orchestration.md) — Coordinating multiple LLMs in pipelines, ensembles, and cascading workflows
- [`ml-rag-evaluation-metrics`](ml-rag-evaluation-metrics.md) — Comprehensive guide to RAG evaluation using RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall), LLM-as-judge patterns, synthetic datasets, and integration with Arize Phoenix and Langfuse
- [`ml-stable-diffusion-deployment`](ml-stable-diffusion-deployment.md) — Deploying Stable Diffusion models to production environments
- [`ml-unsloth-finetuning`](ml-unsloth-finetuning.md) — Fine-tuning LLMs (Llama, Mistral, Qwen) efficiently
- [`rag-reranking-techniques`](rag-reranking-techniques.md) — Multi-stage retrieval pipelines with cross-encoder and LLM-based reranking for improved RAG precision

### math
- [`abstract-algebra`](abstract-algebra.md) — Abstract algebra including groups, rings, fields, Galois theory, and algebraic structures
- [`advanced-graph-algorithms`](advanced-graph-algorithms.md) — Specialized graph algorithms including graph decomposition, vertex coloring, matching algorithms, and 2024 research advances
- [`category-theory-foundations`](category-theory-foundations.md) — Category theory foundations including categories, functors, natural transformations, limits, and adjunctions
- [`differential-equations`](differential-equations.md) — Differential equations including ODEs, PDEs, analytical methods, numerical solutions
- [`graph-data-structures`](graph-data-structures.md) — Efficient graph representations including adjacency matrices, adjacency lists, edge lists, and specialized structures with space-time tradeoffs
- [`graph-theory-applications`](graph-theory-applications.md) — Real-world applications of graph algorithms including social networks, routing, dependency resolution, recommendation systems, and network analysis
- [`graph-theory-fundamentals`](graph-theory-fundamentals.md) — Core concepts, terminology, and mathematical foundations of graph theory including graph types, properties, and fundamental theorems
- [`graph-traversal-algorithms`](graph-traversal-algorithms.md) — Breadth-first search, depth-first search, topological sort, strongly connected components, and traversal-based graph analysis
- [`math-linear-algebra-computation`](math-linear-algebra-computation.md) — Solving systems of linear equations (Ax = b)
- [`math-numerical-methods`](math-numerical-methods.md) — Solving ordinary differential equations (ODEs)
- [`math-optimization-algorithms`](math-optimization-algorithms.md) — Machine learning model training (gradient descent, Adam, SGD)
- [`math-probability-statistics`](math-probability-statistics.md) — Statistical hypothesis testing (t-test, chi-square, ANOVA)
- [`minimum-spanning-tree-algorithms`](minimum-spanning-tree-algorithms.md) — Kruskal and Prim algorithms for finding minimum spanning trees in weighted undirected graphs with union-find and priority queue optimizations
- [`network-flow-algorithms`](network-flow-algorithms.md) — Maximum flow, minimum cut, Ford-Fulkerson, Edmonds-Karp, and flow-based algorithms for capacity-constrained network optimization
- [`number-theory`](number-theory.md) — Number theory including primes, modular arithmetic, Diophantine equations, and cryptographic applications
- [`set-theory`](set-theory.md) — Set theory including ZFC axioms, ordinals, cardinals, axiom of choice, and forcing
- [`shortest-path-algorithms`](shortest-path-algorithms.md) — Dijkstra, Bellman-Ford, Floyd-Warshall, and A* algorithms for finding shortest paths in weighted graphs with various constraints
- [`topology-algebraic`](topology-algebraic.md) — Algebraic topology including fundamental groups, homology, homotopy theory, and computational topology
- [`topology-point-set`](topology-point-set.md) — Point-set topology, metric spaces, open/closed sets, continuity, compactness, separation axioms

### distributed-systems
- [`distributed-systems-cap-theorem`](distributed-systems-cap-theorem.md) — CAP theorem fundamentals, consistency vs availability trade-offs, and practical implications for distributed system design
- [`distributed-systems-conflict-resolution`](distributed-systems-conflict-resolution.md) — Conflict resolution strategies including Last-Write-Wins, multi-value, semantic resolution, and application-specific merge functions
- [`distributed-systems-consensus-paxos`](distributed-systems-consensus-paxos.md) — Paxos consensus algorithm including Basic Paxos, Multi-Paxos, roles, phases, and practical implementations
- [`distributed-systems-consensus-raft`](distributed-systems-consensus-raft.md) — RAFT consensus algorithm including leader election, log replication, safety guarantees, and implementation patterns
- [`distributed-systems-crdt-fundamentals`](distributed-systems-crdt-fundamentals.md) — Conflict-free Replicated Data Types (CRDTs) fundamentals including convergence, commutativity, and basic CRDT operations
- [`distributed-systems-crdt-types`](distributed-systems-crdt-types.md) — Specific CRDT implementations including LWW-Register, OR-Set, RGA, and collaborative text editing CRDTs
- [`distributed-systems-distributed-locks`](distributed-systems-distributed-locks.md) — Distributed locking patterns including Redis Redlock, ZooKeeper locks, lease-based locking, and fencing tokens
- [`distributed-systems-dotted-version-vectors`](distributed-systems-dotted-version-vectors.md) — Dotted version vectors for efficient sibling management, compact causality tracking, reducing metadata overhead compared to pure vector clocks
- [`distributed-systems-event-sourcing`](distributed-systems-event-sourcing.md) — Event sourcing architecture, event design, CQRS, aggregates, projections, snapshots, and production implementations
- [`distributed-systems-eventual-consistency`](distributed-systems-eventual-consistency.md) — Eventual consistency models, consistency levels, read/write quorums, and practical trade-offs in distributed systems
- [`distributed-systems-gossip-protocols`](distributed-systems-gossip-protocols.md) — Gossip protocols for disseminating information, failure detection, and eventual consistency in large-scale distributed systems
- [`distributed-systems-interval-tree-clocks`](distributed-systems-interval-tree-clocks.md) — Interval tree clocks for dynamic systems, scalable causality tracking, fork/join operations, avoiding process ID exhaustion
- [`distributed-systems-leader-election`](distributed-systems-leader-election.md) — Leader election algorithms including bully algorithm, ring algorithm, and consensus-based election with RAFT/Paxos
- [`distributed-systems-logical-clocks`](distributed-systems-logical-clocks.md) — Lamport logical clocks for establishing happened-before ordering in distributed systems without synchronized physical clocks
- [`distributed-systems-partitioning-sharding`](distributed-systems-partitioning-sharding.md) — Data partitioning and sharding strategies including hash-based, range-based, consistent hashing, and rebalancing
- [`distributed-systems-probabilistic-data-structures`](distributed-systems-probabilistic-data-structures.md) — Probabilistic data structures including Bloom filters, HyperLogLog, Count-Min Sketch for space-efficient approximations
- [`distributed-systems-replication-strategies`](distributed-systems-replication-strategies.md) — Data replication strategies including primary-backup, multi-primary, chain replication, and quorum-based replication
- [`distributed-systems-vector-clocks`](distributed-systems-vector-clocks.md) — Vector clocks for tracking causality in distributed systems, detecting concurrent events, and resolving conflicts

### debugging
- [`debugging-browser-devtools`](debugging-browser-devtools.md) — Comprehensive guide to browser developer tools for debugging web applications. Covers Chrome/Firefox/Safari DevTools including Sources panel, breakpoints, performance profiling, memory analysis, network debugging, Console API, React/Vue debugging, and Lighthouse/Core Web Vitals.
- [`debugging-concurrency-debugging`](debugging-concurrency-debugging.md) — Race condition detection, deadlock debugging, and data race analysis using ThreadSanitizer and specialized tools
- [`debugging-container-debugging`](debugging-container-debugging.md) — Debugging applications inside containers using docker exec, kubectl debug, ephemeral containers, and distroless debugging techniques
- [`debugging-core-dump-analysis`](debugging-core-dump-analysis.md) — Core dump generation, GDB/LLDB analysis, crash reporting, and post-mortem debugging across platforms
- [`debugging-crash-debugging`](debugging-crash-debugging.md) — Signal handling, crash reproduction, fuzzing for crash discovery, telemetry aggregation, and post-mortem workflows
- [`debugging-distributed-systems-debugging`](debugging-distributed-systems-debugging.md) — Debugging distributed systems with trace correlation, cross-service workflows, request replay, clock skew handling, and chaos engineering
- [`debugging-gdb-fundamentals`](debugging-gdb-fundamentals.md) — Comprehensive guide to GNU Debugger (GDB) for debugging C/C++/Rust programs. Covers breakpoints, stack traces, variable inspection, TUI mode, .gdbinit customization, Python scripting, remote debugging, and core file analysis.
- [`debugging-lldb-macos-debugging`](debugging-lldb-macos-debugging.md) — Comprehensive guide to LLDB debugger for macOS/iOS development. Covers LLDB vs GDB differences, Swift/Objective-C debugging, Xcode integration, command-line usage, breakpoint expressions, watchpoints, Python scripting, and performance improvements (50x faster step-over in 2025).
- [`debugging-memory-leak-debugging`](debugging-memory-leak-debugging.md) — Comprehensive memory leak detection and debugging using heap profiling tools across multiple languages
- [`debugging-network-debugging`](debugging-network-debugging.md) — Network debugging with tcpdump, Wireshark, curl, DNS tools, SSL/TLS inspection, and network tracing utilities
- [`debugging-performance-profiling`](debugging-performance-profiling.md) — Performance profiling with CPU profilers (perf, pprof, py-spy), flame graphs, memory profiling (heaptrack, Valgrind), and profile-guided optimization
- [`debugging-production-debugging`](debugging-production-debugging.md) — Non-intrusive debugging in production environments with feature flags, dynamic logging, and observability correlation
- [`debugging-python-debugging`](debugging-python-debugging.md) — Comprehensive guide to Python debugging tools and techniques. Covers pdb (built-in debugger), ipdb (IPython integration), VSCode debugger, PyCharm debugger, pytest debugging, remote debugging with debugpy, and performance profiling with cProfile.
- [`debugging-remote-debugging`](debugging-remote-debugging.md) — Comprehensive guide to remote debugging techniques for production and development environments. Covers SSH port forwarding, VSCode Remote Development, JetBrains Gateway, container debugging (Docker), Kubernetes debugging, production-safe debugging, debug symbols, and source maps.

### cloud
- [`cloud-aws-api-gateway`](cloud-aws-api-gateway.md) — AWS API Gateway REST APIs, HTTP APIs, WebSocket APIs, authorization, and integration patterns
- [`cloud-aws-databases`](cloud-aws-databases.md) — AWS database services - RDS, DynamoDB, ElastiCache, Aurora, migration, backup, and optimization
- [`cloud-aws-ec2-compute`](cloud-aws-ec2-compute.md) — AWS EC2 instances, Auto Scaling, Load Balancing, AMIs, and instance lifecycle management
- [`cloud-aws-iam-security`](cloud-aws-iam-security.md) — AWS IAM policies, roles, Cognito authentication, Secrets Manager, KMS encryption, and security best practices
- [`cloud-aws-lambda-functions`](cloud-aws-lambda-functions.md) — AWS Lambda function development, runtime configuration, triggers, and optimization
- [`cloud-aws-networking`](cloud-aws-networking.md) — AWS networking - VPC, subnets, security groups, NACLs, Route53, CloudFront, Transit Gateway
- [`cloud-aws-storage`](cloud-aws-storage.md) — AWS storage services - S3, EBS, EFS, Glacier, lifecycle policies, encryption, and data transfer
- [`cloud-gcp-compute`](cloud-gcp-compute.md) — Google Cloud compute services including Compute Engine, Cloud Run, and GKE
- [`cloud-gcp-databases`](cloud-gcp-databases.md) — Google Cloud managed database services including Cloud SQL, Firestore, Bigtable, and Spanner
- [`cloud-gcp-iam-security`](cloud-gcp-iam-security.md) — Google Cloud IAM, service accounts, Secret Manager, and Cloud KMS security practices
- [`cloud-gcp-networking`](cloud-gcp-networking.md) — Google Cloud networking including VPC, firewall, DNS, CDN, and load balancing
- [`cloud-gcp-serverless`](cloud-gcp-serverless.md) — Google Cloud serverless services including Cloud Functions, Cloud Run, and App Engine
- [`cloud-gcp-storage`](cloud-gcp-storage.md) — Google Cloud storage services including Cloud Storage, Persistent Disk, and Filestore

### database
- [`apache-iceberg`](apache-iceberg.md) — Building data lakes with ACID transaction support and schema evolution
- [`database-database-connection-pooling`](database-database-connection-pooling.md) — Configuring database connections for applications
- [`database-database-selection`](database-database-selection.md) — Starting new projects and choosing database technology
- [`database-elasticsearch-search`](database-elasticsearch-search.md) — Full-text search and relevance optimization with Elasticsearch
- [`database-mongodb-document-design`](database-mongodb-document-design.md) — Designing MongoDB schemas
- [`database-orm-patterns`](database-orm-patterns.md) — Working with ORMs (SQLAlchemy, Prisma, GORM, Diesel)
- [`database-postgres-migrations`](database-postgres-migrations.md) — Creating database migrations for schema changes
- [`database-postgres-query-optimization`](database-postgres-query-optimization.md) — Debugging slow queries in PostgreSQL
- [`database-postgres-schema-design`](database-postgres-schema-design.md) — Designing new database schemas
- [`database-redis-data-structures`](database-redis-data-structures.md) — Implementing caching layers
- [`duckdb-analytics`](duckdb-analytics.md) — Skill: duckdb-analytics Category: Data/Analytics Related: Python, SQL, Parquet, data pipelines
- [`redpanda-streaming`](redpanda-streaming.md) — **Redpanda deployments**: Setting up Kafka-compatible streaming infrastructure
- [`reference`](reference.md) — Scope: Comprehensive guide to database migrations, schema versioning, zero-downtime deployments, and migration tooling Lines: ~1800 Last Updated: 2025-10-27

### elegant-design
- [`anti-patterns`](anti-patterns.md) — ❌ Building custom components when design systems have them - Always check shadcn/ui, daisyUI, HeroUI first - Don't reinvent the wheel
- [`chat-and-messaging`](chat-and-messaging.md) — Chat is a conversation, not a document.
- [`colors-and-spacing`](colors-and-spacing.md) — Create semantic color tokens, not just raw colors.
- [`components-and-accessibility`](components-and-accessibility.md) — Build from small to large: - Atoms: Buttons, inputs, labels - Molecules: Search bar (input + button), form field (label + input + error) - Organisms: Header, card, form - Templates: Page layouts - Pages: Specific instances
- [`design-systems`](design-systems.md) — When to use: Most projects, especially with Tailwind CSS
- [`diffs-and-logs`](diffs-and-logs.md) — Version control UIs and log viewers need clarity and scannability.
- [`layout-patterns`](layout-patterns.md) — Standard for complex layouts:
- [`performance`](performance.md) — Fast interfaces feel better.
- [`streaming-and-loading`](streaming-and-loading.md) — Content that appears instantly feels fast.
- [`terminals-and-code`](terminals-and-code.md) — Terminals and code blocks need to be readable, beautiful, and functional.
- [`testing-and-qa`](testing-and-qa.md) — -   Mobile (320px, 375px, 414px widths) -   Tablet (768px, 1024px widths) -   Desktop (1280px, 1920px widths) -   Both light and dark modes -   Different font sizes (browser zoom 100%, 125%, 150%) -   Slow 3G network simulation
- [`tools-and-libraries`](tools-and-libraries.md) — Prototyping: - Figma (https://figma.com) - Sketch (https://sketch.com) - Adobe XD
- [`typography`](typography.md) — Elegant, modern typography sets the foundation for professional interfaces.

### plt
- [`curry-howard`](curry-howard.md) — Curry-Howard correspondence relating proofs and programs, propositions and types
- [`dependent-types`](dependent-types.md) — Dependent types including Π-types, Σ-types, indexed families, and proof assistants
- [`hazelnut-calculus`](hazelnut-calculus.md) — Hazelnut structure editor calculus - bidirectionally typed lambda calculus with holes and cursor, edit actions preserving static meaning, zipper structures
- [`lambda-calculus`](lambda-calculus.md) — Lambda calculus including untyped, simply typed, Church encodings, and reduction strategies
- [`live-programming-holes`](live-programming-holes.md) — Live programming with typed holes - Hazel environment, continuous feedback, live evaluation with indeterminate results, incremental bidirectional typing (OOPSLA 2025), collaborative editing (Grove)
- [`operational-semantics`](operational-semantics.md) — Operational semantics including small-step, big-step, evaluation strategies, and reduction systems
- [`program-verification`](program-verification.md) — Program verification including Hoare logic, SMT-based verification, refinement types, and proof-carrying code
- [`structure-editors`](structure-editors.md) — Structure editor design patterns - projectional editing, syntax-directed editing, edit actions, cursor management, rendering, text workflow integration
- [`type-systems`](type-systems.md) — Type systems including type checking, inference, polymorphism, subtyping, and soundness
- [`typed-holes-foundations`](typed-holes-foundations.md) — Typed holes fundamentals - incomplete programs with static/dynamic meaning, connection to gradual typing, bidirectional type checking, and theoretical foundations
- [`typed-holes-interaction`](typed-holes-interaction.md) — IDE integration with typed holes - goal-directed programming, hole refinement, case splitting, proof search, tactics, elaborator reflection (Agda, Idris, Lean models)
- [`typed-holes-llm`](typed-holes-llm.md) — LLM integration with typed holes (OOPSLA 2024) - static context for code synthesis, type-driven prompting, validation, ranking, combining language servers with LLMs
- [`typed-holes-semantics`](typed-holes-semantics.md) — Advanced typed holes semantics - hole closures, pattern matching with holes, type error localization, polymorphism, gradual guarantees, blame tracking

### cryptography
- [`07_emergency_rotation_runbook`](07_emergency_rotation_runbook.md) — Purpose: Emergency rotation procedure for compromised secrets SLA: Complete rotation within 1 hour of detection Scope: All production secrets (databases, API keys, certificates, tokens)
- [`cryptography-certificate-management`](cryptography-certificate-management.md) — Certificate lifecycle management including rotation, renewal, monitoring, automation with Let's Encrypt and ACME
- [`cryptography-crypto-best-practices`](cryptography-crypto-best-practices.md) — Cryptography best practices, common mistakes, security patterns, and anti-patterns to avoid
- [`cryptography-cryptography-basics`](cryptography-cryptography-basics.md) — Cryptography fundamentals including symmetric/asymmetric encryption, hashing, signing, key exchange, and common algorithms
- [`cryptography-encryption-at-rest`](cryptography-encryption-at-rest.md) — Protecting stored data with encryption at rest
- [`cryptography-key-management`](cryptography-key-management.md) — Comprehensive cryptographic key lifecycle management, KMS integration, and compliance
- [`cryptography-pki-fundamentals`](cryptography-pki-fundamentals.md) — PKI fundamentals including certificate authorities, chains of trust, X.509 certificates, and certificate lifecycle
- [`cryptography-signing-verification`](cryptography-signing-verification.md) — Digital signature creation, verification, and chain of trust for documents, code, and artifacts
- [`cryptography-sni-routing`](cryptography-sni-routing.md) — Server Name Indication (SNI) for multi-domain TLS hosting, routing, and configuration patterns
- [`cryptography-ssl-legacy`](cryptography-ssl-legacy.md) — Legacy SSL/TLS protocols (SSL 2.0/3.0, TLS 1.0/1.1), vulnerabilities, deprecation, and migration strategies
- [`cryptography-tls-configuration`](cryptography-tls-configuration.md) — TLS/SSL configuration including TLS 1.2/1.3 setup, cipher suites, security best practices, and server configuration

### formal
- [`formal-backtracking-search`](formal-backtracking-search.md) — Implementing CSP solvers with systematic search
- [`formal-constraint-propagation`](formal-constraint-propagation.md) — Pruning search space before backtracking search
- [`formal-csp-modeling`](formal-csp-modeling.md) — Modeling scheduling problems (meetings, courses, jobs)
- [`formal-lean-mathlib4`](formal-lean-mathlib4.md) — Formalizing mathematics using existing results
- [`formal-lean-proof-basics`](formal-lean-proof-basics.md) — Learning Lean 4 from scratch
- [`formal-lean-tactics`](formal-lean-tactics.md) — Optimizing proof efficiency
- [`formal-lean-theorem-proving`](formal-lean-theorem-proving.md) — Formalizing complex mathematics
- [`formal-sat-solving-strategies`](formal-sat-solving-strategies.md) — Out of Scope: - SMT theories (see z3-solver-basics.md) - Theorem proving (see lean-theorem-proving.md) - Constraint programming (see csp-modeling.md) - Solver implementation (focus on usage)
- [`formal-smt-theory-applications`](formal-smt-theory-applications.md) — Out of Scope: - Pure SAT solving (see sat-solving-strategies.md) - Z3 API basics (see z3-solver-basics.md) - Interactive theorem proving (see lean-theorem-proving.md) - Constraint programming (see csp-modeling.md)
- [`formal-z3-solver-basics`](formal-z3-solver-basics.md) — Out of Scope: - Low-level C++ API (use Python bindings) - Custom theory development - Advanced proof engineering (see lean-theorem-proving.md) - SAT-only solving (see sat-solving-strategies.md)

### frontend
- [`frontend-browser-concurrency`](frontend-browser-concurrency.md) — Service Workers, SharedWorkers, Worklets, and multi-threading patterns for advanced browser concurrency
- [`frontend-frontend-performance`](frontend-frontend-performance.md) — Optimizing application performance
- [`frontend-nextjs-app-router`](frontend-nextjs-app-router.md) — Building Next.js 13+ applications
- [`frontend-nextjs-seo`](frontend-nextjs-seo.md) — Implementing SEO for Next.js applications
- [`frontend-react-component-patterns`](frontend-react-component-patterns.md) — Designing component architecture
- [`frontend-react-data-fetching`](frontend-react-data-fetching.md) — Scope: SWR, React Query, Server Actions, caching strategies, optimistic updates Lines: ~310 Last Updated: 2025-10-18
- [`frontend-react-form-handling`](frontend-react-form-handling.md) — Scope: React Hook Form, Zod validation, Server Actions, multi-step forms, form patterns Lines: ~290 Last Updated: 2025-10-18
- [`frontend-react-state-management`](frontend-react-state-management.md) — Choosing state management solution
- [`frontend-web-accessibility`](frontend-web-accessibility.md) — Building accessible web applications
- [`frontend-web-workers`](frontend-web-workers.md) — Offloading computation with Web Workers API, SharedArrayBuffer, and message passing patterns

### mobile
- [`ios-networking`](ios-networking.md) — Building network layers for iOS/macOS apps
- [`ios-testing`](ios-testing.md) — Writing unit tests for iOS/macOS apps
- [`mobile-react-native-native-modules`](mobile-react-native-native-modules.md) — Accessing iOS-specific APIs not available in React Native
- [`mobile-react-native-navigation`](mobile-react-native-navigation.md) — Implementing multi-screen navigation in React Native apps
- [`mobile-react-native-performance`](mobile-react-native-performance.md) — App animations are dropping frames (not 60fps)
- [`mobile-react-native-setup`](mobile-react-native-setup.md) — Starting a new React Native project for iOS development
- [`swift-concurrency`](swift-concurrency.md) — Working with async/await in Swift
- [`swiftdata-persistence`](swiftdata-persistence.md) — Building iOS/macOS apps with local persistence
- [`swiftui-architecture`](swiftui-architecture.md) — Building iOS/macOS apps with SwiftUI
- [`swiftui-navigation`](swiftui-navigation.md) — Implementing navigation in iOS 17+ apps

### theme-factory
- [`arctic-frost`](arctic-frost.md) — A cool and crisp winter-inspired theme that conveys clarity, precision, and professionalism.
- [`botanical-garden`](botanical-garden.md) — A fresh and organic theme featuring vibrant garden-inspired colors for lively presentations.
- [`desert-rose`](desert-rose.md) — A soft and sophisticated theme with dusty, muted tones perfect for elegant presentations.
- [`forest-canopy`](forest-canopy.md) — A natural and grounded theme featuring earth tones inspired by dense forest environments.
- [`golden-hour`](golden-hour.md) — A rich and warm autumnal palette that creates an inviting and sophisticated atmosphere.
- [`midnight-galaxy`](midnight-galaxy.md) — A dramatic and cosmic theme with deep purples and mystical tones for impactful presentations.
- [`modern-minimalist`](modern-minimalist.md) — A clean and contemporary theme with a sophisticated grayscale palette for maximum versatility.
- [`ocean-depths`](ocean-depths.md) — A professional and calming maritime theme that evokes the serenity of deep ocean waters.
- [`sunset-boulevard`](sunset-boulevard.md) — A warm and vibrant theme inspired by golden hour sunsets, perfect for energetic and creative presentations.
- [`tech-innovation`](tech-innovation.md) — A bold and modern theme with high-contrast colors perfect for cutting-edge technology presentations.

### data
- [`data-batch-processing`](data-batch-processing.md) — Orchestrating complex data pipelines with dependencies
- [`data-data-validation`](data-data-validation.md) — Validating data schema before processing
- [`data-dataflow-coordination`](data-dataflow-coordination.md) — Coordination patterns for distributed dataflow systems including barriers, epochs, and distributed snapshots
- [`data-differential-dataflow`](data-differential-dataflow.md) — Differential computation for incremental updates, maintaining indexed collections and efficient joins
- [`data-etl-patterns`](data-etl-patterns.md) — Designing data extraction from multiple sources (databases, APIs, files)
- [`data-pipeline-orchestration`](data-pipeline-orchestration.md) — Coordinating complex multi-step data workflows
- [`data-stream-processing`](data-stream-processing.md) — Processing real-time event streams (clicks, IoT, logs)
- [`data-streaming-aggregations`](data-streaming-aggregations.md) — Windowing, sessionization, time-series aggregation, and late data handling for streaming systems
- [`data-timely-dataflow`](data-timely-dataflow.md) — Timely dataflow framework for low-latency, high-throughput streaming computation with progress tracking

### observability
- [`observability-alerting-strategy`](observability-alerting-strategy.md) — Setting up production monitoring and on-call rotation
- [`observability-cost-optimization`](observability-cost-optimization.md) — Optimizing observability costs through sampling, cardinality management, and retention strategies
- [`observability-dashboard-design`](observability-dashboard-design.md) — Building Grafana dashboards for service monitoring
- [`observability-distributed-tracing`](observability-distributed-tracing.md) — Debugging latency issues in microservices
- [`observability-metrics-instrumentation`](observability-metrics-instrumentation.md) — Instrumenting applications for observability
- [`observability-opentelemetry-integration`](observability-opentelemetry-integration.md) — Integrating OpenTelemetry for unified observability (traces, metrics, logs)
- [`observability-production-incident-debugging`](observability-production-incident-debugging.md) — Debugging production incidents using observability data (logs, metrics, traces)
- [`observability-prometheus-monitoring`](observability-prometheus-monitoring.md) — Production-ready Prometheus monitoring and alerting
- [`observability-structured-logging`](observability-structured-logging.md) — Setting up application logging infrastructure

### build-systems
- [`build-systems-bazel-monorepos`](build-systems-bazel-monorepos.md) — Bazel BUILD files, WORKSPACE/MODULE.bazel, hermetic builds, remote caching, build rules, Starlark custom rules, and monorepo best practices for polyglot projects.
- [`build-systems-build-optimization`](build-systems-build-optimization.md) — Incremental builds, dependency tracking, build caching (ccache, sccache, Bazel), parallel builds, build time profiling, and CI/CD optimization strategies.
- [`build-systems-build-system-selection`](build-systems-build-system-selection.md) — Decision matrix for choosing between Make, CMake, Gradle, Maven, Bazel, and other build systems based on project requirements, language, and scale.
- [`build-systems-cmake-patterns`](build-systems-cmake-patterns.md) — Modern CMake target-based approach, find_package, FetchContent, generator expressions, cross-platform configuration, and installation rules for C/C++ projects.
- [`build-systems-cross-platform-builds`](build-systems-cross-platform-builds.md) — Platform detection, conditional compilation, preprocessor macros, CMake cross-compilation, Zig cross-compilation, toolchain configuration, and multi-platform testing strategies.
- [`build-systems-gradle-jvm-builds`](build-systems-gradle-jvm-builds.md) — Gradle Kotlin DSL and Groovy DSL, dependency management, version catalogs, build lifecycle, task configuration, multi-project builds, and performance optimization for Java/Kotlin projects.
- [`build-systems-make-fundamentals`](build-systems-make-fundamentals.md) — Makefile syntax, targets, dependencies, pattern rules, and best practices for building C/C++ and multi-language projects with Make.
- [`build-systems-maven-configuration`](build-systems-maven-configuration.md) — Maven POM structure, dependency management, plugin configuration, lifecycle phases, multi-module projects, profiles, and repository management for Java projects.

### diagrams
- [`mermaid-architecture-diagrams`](mermaid-architecture-diagrams.md) — Create C4 and block architecture diagrams with Mermaid for system design and infrastructure visualization
- [`mermaid-charts`](mermaid-charts.md) — Create data visualization charts with Mermaid including pie, XY, quadrant, and radar charts for metrics and analytics
- [`mermaid-class-state-diagrams`](mermaid-class-state-diagrams.md) — Create UML class diagrams and state machines with Mermaid for object-oriented design and state modeling
- [`mermaid-er-diagrams`](mermaid-er-diagrams.md) — Create entity-relationship diagrams with Mermaid for database schema design and data modeling
- [`mermaid-flowcharts`](mermaid-flowcharts.md) — Create process flow diagrams with Mermaid using nodes, arrows, decisions, and subgraphs for visual documentation
- [`mermaid-project-diagrams`](mermaid-project-diagrams.md) — Create Gantt charts and timeline diagrams with Mermaid for project planning and chronological visualization
- [`mermaid-sequence-diagrams`](mermaid-sequence-diagrams.md) — Create interaction and message flow diagrams with Mermaid showing communication between actors, services, and components over time
- [`mermaid-specialized`](mermaid-specialized.md) — Create specialized diagrams with Mermaid including Git graphs, Sankey flows, mindmaps, and other advanced visualization types

### modal
- [`modal-functions-basics`](modal-functions-basics.md) — Building serverless functions on Modal.com
- [`modal-gpu-workloads`](modal-gpu-workloads.md) — Running ML/AI inference on Modal
- [`modal-image-building`](modal-image-building.md) — Configuring Modal runtime environments
- [`modal-modal-common-errors`](modal-modal-common-errors.md) — Encountering Modal deployment or runtime errors
- [`modal-modal-performance-debugging`](modal-modal-performance-debugging.md) — Modal functions running slower than expected
- [`modal-scheduling`](modal-scheduling.md) — Running periodic tasks on Modal
- [`modal-volumes-secrets`](modal-volumes-secrets.md) — Persisting data across function invocations
- [`modal-web-endpoints`](modal-web-endpoints.md) — Creating HTTP APIs on Modal

### protocols
- [`migration_guide`](migration_guide.md) — Comprehensive guide for migrating Protocol Buffer schemas safely while maintaining backward/forward compatibility.
- [`protocols-amqp-rabbitmq`](protocols-amqp-rabbitmq.md) — RabbitMQ and AMQP 0-9-1 message broker implementation
- [`protocols-grpc-implementation`](protocols-grpc-implementation.md) — Implementing gRPC APIs with Protocol Buffers
- [`protocols-http2-multiplexing`](protocols-http2-multiplexing.md) — HTTP/2 protocol with multiplexing, server push, header compression, and stream prioritization
- [`protocols-kafka-streams`](protocols-kafka-streams.md) — Apache Kafka stream processing and event streaming platform
- [`protocols-mqtt-messaging`](protocols-mqtt-messaging.md) — MQTT pub-sub messaging for IoT and real-time applications
- [`protocols-protobuf-schemas`](protocols-protobuf-schemas.md) — Protocol Buffers schema design, evolution, and code generation
- [`protocols-websocket-protocols`](protocols-websocket-protocols.md) — WebSocket protocol implementation, scaling, and production deployment

### api
- [`api-api-authentication`](api-api-authentication.md) — Implementing API authentication for new services
- [`api-api-authorization`](api-api-authorization.md) — Implementing authorization logic in APIs
- [`api-api-error-handling`](api-api-error-handling.md) — Designing error responses for REST APIs
- [`api-api-rate-limiting`](api-api-rate-limiting.md) — Implementing rate limiting for APIs
- [`api-api-versioning`](api-api-versioning.md) — Planning API changes and deciding on versioning approach
- [`api-graphql-schema-design`](api-graphql-schema-design.md) — Designing GraphQL APIs from scratch
- [`api-rest-api-design`](api-rest-api-design.md) — Designing RESTful APIs from scratch

### caching
- [`caching-cache-invalidation-strategies`](caching-cache-invalidation-strategies.md) — Cache invalidation patterns and techniques - time-based, event-based, key-based, and version-based invalidation for maintaining cache consistency.
- [`caching-cache-performance-monitoring`](caching-cache-performance-monitoring.md) — Measuring and optimizing cache performance - metrics, monitoring tools, alerts, load testing, and instrumentation for cache hit rates and latency.
- [`caching-caching-fundamentals`](caching-caching-fundamentals.md) — Core caching concepts, patterns, eviction policies, and cache design principles for optimizing application performance
- [`caching-cdn-edge-caching`](caching-cdn-edge-caching.md) — CDN and edge cache optimization with Cloudflare, Fastly, and CloudFront for global performance and reduced origin load
- [`caching-http-caching`](caching-http-caching.md) — Browser and HTTP cache layer optimization using cache headers, conditional requests, and validation strategies
- [`caching-redis-caching-patterns`](caching-redis-caching-patterns.md) — Application-level caching with Redis - patterns, distributed caching, cache stampede prevention, and integration strategies for high-performance applications.
- [`caching-service-worker-caching`](caching-service-worker-caching.md) — Progressive Web App (PWA) caching with Service Workers - offline-first architecture, caching strategies, Workbox patterns, and modern browser caching.

### proxies
- [`proxies-cache-control`](proxies-cache-control.md) — HTTP caching strategies including cache headers, CDN patterns, cache invalidation, stale-while-revalidate, and edge caching optimization
- [`proxies-envoy-proxy`](proxies-envoy-proxy.md) — Envoy proxy architecture including filters, clusters, listeners, service mesh patterns, observability, and advanced load balancing
- [`proxies-forward-proxy`](proxies-forward-proxy.md) — Forward proxy fundamentals including HTTP CONNECT, SOCKS protocols, authentication, and use cases for privacy, security, and access control
- [`proxies-nats-messaging`](proxies-nats-messaging.md) — NATS messaging patterns including pub/sub, request-reply, queue groups, JetStream persistence, clustering, and high-performance messaging
- [`proxies-nginx-configuration`](proxies-nginx-configuration.md) — Nginx configuration including server blocks, locations, upstreams, caching, SSL/TLS, security headers, and performance optimization
- [`proxies-reverse-proxy`](proxies-reverse-proxy.md) — Reverse proxy patterns including load balancing, SSL termination, request routing, health checks, and high availability configurations
- [`proxies-traefik-configuration`](proxies-traefik-configuration.md) — Traefik configuration including dynamic service discovery, middleware chains, automatic Let's Encrypt, Docker/Kubernetes integration, and cloud-native routing

### research
- [`research-data-analysis`](research-data-analysis.md) — Master data analysis techniques including coding, statistical tests, visualization, interpretation, and reporting findings
- [`research-data-collection`](research-data-collection.md) — Master data collection methods including survey design, interview protocols, observation techniques, and measurement instruments
- [`research-qualitative-methods`](research-qualitative-methods.md) — Master qualitative research methods including interviews, ethnography, case studies, grounded theory, and thematic analysis
- [`research-quantitative-methods`](research-quantitative-methods.md) — Master quantitative research methods including statistical analysis, surveys, experiments, and hypothesis testing
- [`research-research-design`](research-research-design.md) — Master research study design including hypothesis formation, validity, sampling strategies, and experimental control
- [`research-research-writing`](research-research-writing.md) — Master academic and research writing including structure, citations, reporting standards, and peer review processes
- [`research-synthesis`](research-synthesis.md) — Master information synthesis, literature review, meta-analysis, and knowledge integration across sources

### security
- [`security-authentication`](security-authentication.md) — Authentication patterns including JWT, OAuth2, sessions, and multi-factor authentication for secure user identity verification
- [`security-authorization`](security-authorization.md) — Authorization models including RBAC, ABAC, policy engines, and access control patterns for securing resources
- [`security-input-validation`](security-input-validation.md) — Input validation and sanitization patterns to prevent SQL injection, XSS, command injection, and other input-based attacks
- [`security-oauth2-implementation`](security-oauth2-implementation.md) — OAuth 2.0 implementation including authorization code, client credentials, PKCE, refresh tokens, and security best practices
- [`security-secrets-management`](security-secrets-management.md) — Secrets management patterns including vaults, environment variables, key rotation, and secure credential handling for applications
- [`security-security-headers`](security-security-headers.md) — HTTP security headers including CSP, HSTS, X-Frame-Options, CORS, and other protective headers for web application security
- [`security-vulnerability-assessment`](security-vulnerability-assessment.md) — Security testing methodologies, OWASP Top 10, vulnerability scanning tools, and penetration testing practices

### ui-port
- [`acceptance`](acceptance.md) — Choose states appropriate to the app.
- [`exchange-example`](exchange-example.md) — This example captures a common failure mode: the shell is close, but the new system still reads as a different product because screen topology, typography, control geometry and information hierarchy diverge.
- [`ledger`](ledger.md) — Use one row per independently actionable issue.
- [`modes`](modes.md) — Use browser capture, DOM/style probes, source inspection, and automatic target edits.
- [`probe-contract`](probe-contract.md) — Use this when live DOM access is available.
- [`report-template`](report-template.md) — - Reference: - Target: - Viewport / DPR: - Browser: - Theme: - Fixture/state:
- [`screen-convergence`](screen-convergence.md) — Create one section per route and state.

### collaboration
- [`codetour-guided-walkthroughs`](codetour-guided-walkthroughs.md) — Creating and following CodeTour walkthroughs for codebase understanding
- [`collaboration-github-actions-workflows`](collaboration-github-actions-workflows.md) — Setting up CI/CD pipelines with GitHub Actions
- [`collaboration-github-issues-projects`](collaboration-github-issues-projects.md) — Issue management, templates, labels, milestones, GitHub Projects, and issue linking workflows
- [`collaboration-github-pull-requests`](collaboration-github-pull-requests.md) — Pull request workflow, code review, merge strategies, PR checks, and collaboration best practices
- [`collaboration-github-repository-management`](collaboration-github-repository-management.md) — Repository creation, configuration, branch protection, tags, releases, and GitHub CLI operations
- [`collaboration-github-security-features`](collaboration-github-security-features.md) — Dependabot, code scanning, secret scanning, SBOM generation, security advisories, and security best practices

### containers
- [`comparison`](comparison.md) — This directory contains examples showing the impact of Docker optimization techniques.
- [`containers-container-networking`](containers-container-networking.md) — Configuring container network connectivity
- [`containers-container-registry-management`](containers-container-registry-management.md) — Choosing a container registry (ECR/GCR/Harbor/Docker Hub)
- [`containers-container-security`](containers-container-security.md) — Hardening container images for production
- [`containers-docker-compose-development`](containers-docker-compose-development.md) — Setting up local development environments
- [`containers-dockerfile-optimization`](containers-dockerfile-optimization.md) — Writing Dockerfiles for production applications

### deployment
- [`deployment-heroku-addons`](deployment-heroku-addons.md) — Adding databases (Postgres, MySQL, MongoDB) to Heroku apps
- [`deployment-heroku-deployment`](deployment-heroku-deployment.md) — Deploying web applications, APIs, or worker processes to Heroku
- [`deployment-heroku-troubleshooting`](deployment-heroku-troubleshooting.md) — Application crashes on Heroku with H10, H12, H13, H14 errors
- [`deployment-netlify-deployment`](deployment-netlify-deployment.md) — Deploying static sites, JAMstack apps, or frontend frameworks to Netlify
- [`deployment-netlify-functions`](deployment-netlify-functions.md) — Building API endpoints for JAMstack applications
- [`deployment-netlify-optimization`](deployment-netlify-optimization.md) — Improving site load times and Core Web Vitals

### infrastructure
- [`infrastructure-aws-serverless`](infrastructure-aws-serverless.md) — Building serverless APIs and backends
- [`infrastructure-cloudflare-workers`](infrastructure-cloudflare-workers.md) — Building serverless APIs at the edge (low latency globally)
- [`infrastructure-cost-optimization`](infrastructure-cost-optimization.md) — Cloud costs are growing unexpectedly
- [`infrastructure-infrastructure-security`](infrastructure-infrastructure-security.md) — Setting up IAM roles and policies
- [`infrastructure-kubernetes-basics`](infrastructure-kubernetes-basics.md) — Deploying containerized applications at scale
- [`infrastructure-terraform-patterns`](infrastructure-terraform-patterns.md) — Provisioning cloud infrastructure (AWS, GCP, Azure, Cloudflare)

### testing
- [`testing-e2e-testing`](testing-e2e-testing.md) — Test complete user workflows from UI to backend
- [`testing-integration-testing`](testing-integration-testing.md) — Test interactions between multiple components or services
- [`testing-performance-testing`](testing-performance-testing.md) — Measure system performance under load (response time, throughput)
- [`testing-test-coverage-strategy`](testing-test-coverage-strategy.md) — Determine what code needs testing vs what to skip
- [`testing-test-driven-development`](testing-test-driven-development.md) — Design APIs and interfaces through tests
- [`testing-unit-testing-patterns`](testing-unit-testing-patterns.md) — Write isolated tests for individual functions, classes, or modules

### zig
- [`zig-build-system`](zig-build-system.md) — Use this skill when: - Configuring build.zig - Adding dependencies - Cross-compiling projects - Creating build steps - Managing build options
- [`zig-c-interop`](zig-c-interop.md) — Calling C libraries from Zig
- [`zig-memory-management`](zig-memory-management.md) — Use this skill when: - Managing memory in Zig - Using allocators correctly - Implementing defer/errdefer - Avoiding memory leaks - Understanding Zig's explicit allocation
- [`zig-package-management`](zig-package-management.md) — Adding dependencies to Zig projects
- [`zig-project-setup`](zig-project-setup.md) — Initializing new Zig projects
- [`zig-testing`](zig-testing.md) — Writing unit tests in Zig

### ir
- [`ir-query-understanding`](ir-query-understanding.md) — Query expansion, spell correction, semantic search, query classification, entity recognition, and autocomplete
- [`ir-ranking-reranking`](ir-ranking-reranking.md) — Learning to rank, cross-encoder reranking, ranking features, and evaluation metrics (nDCG, MAP, MRR)
- [`ir-recommendation-systems`](ir-recommendation-systems.md) — Collaborative filtering, content-based filtering, hybrid recommenders, matrix factorization, and cold start solutions
- [`ir-search-fundamentals`](ir-search-fundamentals.md) — Core information retrieval concepts including TF-IDF, BM25, inverted indexes, and Elasticsearch
- [`ir-vector-search`](ir-vector-search.md) — Dense retrieval using embeddings, vector databases, approximate nearest neighbors, and hybrid search

### networking
- [`mosh-resilient-ssh`](mosh-resilient-ssh.md) — Connecting over unreliable networks (mobile, WiFi)
- [`mtls-implementation`](mtls-implementation.md) — Implementing mutual TLS authentication
- [`nat-traversal`](nat-traversal.md) — Building peer-to-peer applications
- [`network-resilience-patterns`](network-resilience-patterns.md) — Building robust network applications
- [`tailscale-vpn`](tailscale-vpn.md) — Creating secure private networks (mesh VPN)

### tui
- [`bubbletea-architecture`](bubbletea-architecture.md) — Building terminal UIs in Go
- [`bubbletea-components`](bubbletea-components.md) — Building UIs with pre-made components
- [`ratatui-architecture`](ratatui-architecture.md) — Building terminal UIs in Rust
- [`ratatui-widgets`](ratatui-widgets.md) — Using Ratatui's built-in widgets
- [`tui-best-practices`](tui-best-practices.md) — Designing terminal user interfaces

### workflow
- [`beads-context-strategies`](beads-context-strategies.md) — Use this skill when: Managing Claude Code context while working with Beads, preventing context bloat, or preserving critical workflow state across sessions
- [`beads-dependency-management`](beads-dependency-management.md) — Use this skill when: Creating relationships between issues, managing blockers, or organizing work hierarchies
- [`beads-multi-session-patterns`](beads-multi-session-patterns.md) — Use this skill when: Working on complex tasks spanning multiple sessions, managing long-horizon work chains, or coordinating parallel work streams
- [`beads-workflow`](beads-workflow.md) — Use this skill when: Starting a new session with Beads, running bd commands, or managing issue workflow state
- [`workflow-project-synthesis`](workflow-project-synthesis.md) — Synthesize scattered project artifacts (plans, docs, tests, logs) into unified roadmap. Use when projects have accumulated sprawl across multiple planning documents, inconsistent naming, orphaned tests, or Beads issues out of sync with reality. Creates coherent execution plan while preserving research and maintaining referential integrity.

### anti-slop
- [`claude_md_updates`](claude_md_updates.md) — This document contains proposed updates to CLAUDE.md to integrate anti-slop awareness and prevention into Claude's core behavior.
- [`code-patterns`](code-patterns.md) — Reference guide for detecting AI slop patterns in code including generic variable names, obvious comments, and unnecessary abstractions. Use as reference when reviewing code quality.
- [`design-patterns`](design-patterns.md) — Reference guide for detecting AI slop patterns in visual and UX design including generic gradients, cookie-cutter layouts, and buzzword-heavy copy. Use as reference when reviewing design quality.
- [`text-patterns`](text-patterns.md) — Reference guide for detecting AI slop patterns in natural language text including overused phrases, buzzwords, and meta-commentary. Use as reference when reviewing text content for quality.

### cicd
- [`cicd-cd-deployment-patterns`](cicd-cd-deployment-patterns.md) — Implementing zero-downtime deployment strategies
- [`cicd-ci-optimization`](cicd-ci-optimization.md) — CI pipelines are taking too long (>10 minutes for feedback)
- [`cicd-ci-security`](cicd-ci-security.md) — Managing secrets and credentials in CI/CD pipelines
- [`cicd-ci-testing-strategy`](cicd-ci-testing-strategy.md) — Designing test execution strategies for CI pipelines

### ebpf
- [`ebpf-fundamentals`](ebpf-fundamentals.md) — eBPF basics including program types, verifier, maps, BCC vs libbpf toolchains, and loading programs
- [`ebpf-networking`](ebpf-networking.md) — Fast packet processing with XDP, TC filtering, socket programs, load balancing, and Cilium networking
- [`ebpf-security-monitoring`](ebpf-security-monitoring.md) — Security monitoring with eBPF including syscall tracking, file/network monitoring, Falco, Tetragon, and threat detection
- [`ebpf-tracing-observability`](ebpf-tracing-observability.md) — Using eBPF for tracing with bpftrace, kprobes, uprobes, tracepoints, and performance analysis

### internal-comms
- [`3p-updates`](3p-updates.md) — 3Ps can cover a team of any size, ranging all the way up to the entire company.
- [`company-newsletter`](company-newsletter.md) — Ideally it includes the following attributes: - Lots of links: pulling documents from Google Drive that are very relevant, linking to prominent Slack messages in announce channels and from executives, perhgaps referencing emails that went c
- [`faq-answers`](faq-answers.md) — - Find questions that are big sources of confusion for lots of employees at the company, generally about things that affect a large portion of the employee base - Attempt to give a nice summarized answer to that question in order to minimiz
- [`general-comms`](general-comms.md) — Before proceeding: 1.

### mcp-builder
- [`evaluation`](evaluation.md) — This document provides guidance on creating comprehensive evaluations for MCP servers.
- [`mcp_best_practices`](mcp_best_practices.md) — Follow these standardized naming patterns:
- [`node_mcp_server`](node_mcp_server.md) — This document provides Node/TypeScript-specific best practices and examples for implementing MCP servers using the MCP TypeScript SDK.
- [`python_mcp_server`](python_mcp_server.md) — This document provides Python-specific best practices and examples for implementing MCP servers using the MCP Python SDK.

### product
- [`product-prd-requirements-gathering`](product-prd-requirements-gathering.md) — Research methods, stakeholder interviews, prioritization frameworks, and requirements synthesis for PRDs
- [`product-prd-structure-templates`](product-prd-structure-templates.md) — PRD document structure, templates, and best practices for product requirements documentation
- [`product-prd-technical-specifications`](product-prd-technical-specifications.md) — API specifications, data models, architecture diagrams, and technical constraints for product requirements
- [`product-prd-user-stories-acceptance`](product-prd-user-stories-acceptance.md) — Writing effective user stories, epics, acceptance criteria, and story mapping for product requirements

### realtime
- [`realtime-pubsub-patterns`](realtime-pubsub-patterns.md) — Scope: Publish/subscribe architecture, Redis Pub/Sub, message queues, and fan-out patterns Lines: 372 Last Updated: 2025-10-18
- [`realtime-realtime-sync`](realtime-realtime-sync.md) — Scope: Conflict resolution, CRDTs, operational transformation, and eventual consistency patterns Lines: 398 Last Updated: 2025-10-18
- [`realtime-server-sent-events`](realtime-server-sent-events.md) — Scope: SSE protocol, event streams, automatic reconnection, and fallback strategies Lines: 343 Last Updated: 2025-10-18
- [`realtime-websocket-implementation`](realtime-websocket-implementation.md) — Scope: WebSocket protocol, connection management, reconnection strategies, and heartbeat patterns Lines: 387 Last Updated: 2025-10-18

### skill-creator
- [`analyzer`](analyzer.md) — Analyze blind comparison results to understand WHY the winner won and generate improvement suggestions.
- [`comparator`](comparator.md) — Compare two outputs WITHOUT knowing which skill produced them.
- [`grader`](grader.md) — Evaluate expectations against an execution transcript and outputs.
- [`schemas`](schemas.md) — This document defines the JSON schemas used by skill-creator.

### typed-holes-refactor
- [`constraint-rules`](constraint-rules.md) — Reference guide for constraint propagation rules in typed holes refactoring including dependency tracking and solution space narrowing. Use as reference when propagating constraints between resolved holes.
- [`hole-types`](hole-types.md) — Reference taxonomy of refactoring hole types including current state holes, architecture holes, and implementation holes. Use as reference during typed holes refactoring for identifying and categorizing unknowns.
- [`typed-holes-examples`](typed-holes-examples.md) — Complete worked examples of typed holes refactoring including simple file reorganization and complex API consolidation scenarios with full test suites and constraint validation.
- [`validation-patterns`](validation-patterns.md) — Reference guide for test patterns to validate hole resolutions including characterization tests, property tests, and integration tests. Use as reference when validating refactoring steps.

### wasm
- [`wasm-browser-integration`](wasm-browser-integration.md) — Loading WebAssembly modules in browsers with JavaScript interop, DOM access, WebGL, and performance optimization
- [`wasm-fundamentals`](wasm-fundamentals.md) — WebAssembly basics including binary format, modules, linear memory, WASI, and core tooling
- [`wasm-rust-toolchain`](wasm-rust-toolchain.md) — Rust to WebAssembly compilation using wasm-pack, wasm-bindgen, optimization, and JavaScript interop
- [`wasm-server-side`](wasm-server-side.md) — Server-side WebAssembly execution with Wasmtime, WASI support, edge computing, and plugin systems

### hanzo-api
- [`hanzo-api-ai`](hanzo-api-ai.md) — Hanzo AI — the model API on /v1 (/v1/chat/completions, /v1/messages, /v1/models and the rest of hanzoai/ai's surface) — mounted into a cloud binary with the money, ingest and telemetry callbacks…
- [`hanzo-api-kms`](hanzo-api-kms.md) — Secret custody: your org's secrets sealed at rest, plus threshold signing.
- [`hanzo-api-sandbox`](hanzo-api-sandbox.md) — The ONE compute primitive: a sandbox is a gVisor pod that runs somebody else's code, and every lifetime is the same object.

### zoo-gym
- [`bitdelta`](bitdelta.md) — BitDelta enables millions of personalized model variants for individual users WITHOUT LoRA.
- [`deltasoup`](deltasoup.md) — DeltaSoup enables community-driven model improvement through Byzantine-robust aggregation of personalized model deltas.
- [`training-free-grpo`](training-free-grpo.md) — Training-Free GRPO (Group Relative Policy Optimization) is Zoo Gym's default RLHF method that eliminates value networks entirely, achieving 40-60% memory reduction and 2× speedup compared to PPO.

### Standards: Core AI & Models
- [`hip-0000`](hip-0000.md) — HIP-0: Hanzo AI Architecture & Framework. Status Living. Hanzo architectural specification.
- [`hip-0001`](hip-0001.md) — HIP-1: AI Token - Hanzo's Native Currency. Status Final. Hanzo architectural specification.
- [`hip-0002`](hip-0002.md) — HIP-2: Hamiltonian Large Language Models (HLLMs) Specification. Status Final. Hanzo architectural specification.
- [`hip-0003`](hip-0003.md) — HIP-3: Jin Multimodal AI Architecture. Status Draft. Hanzo architectural specification.
- [`hip-0006`](hip-0006.md) — HIP-6: Personalized Model Tuning Architecture. Status Final. Hanzo architectural specification.
- [`hip-0007`](hip-0007.md) — HIP-7: Active Inference Integration for Hamiltonian LLMs. Status Final. Hanzo architectural specification.
- [`hip-0008`](hip-0008.md) — HIP-8: HMM (Hanzo Market Maker) - Native DEX for AI Compute Resources. Status Final. Hanzo architectural specification.
- [`hip-0009`](hip-0009.md) — HIP-9: Agent SDK — Multiagent Orchestration Framework. Status Final. Hanzo architectural specification.
- [`hip-0019`](hip-0019.md) — HIP-19: Tensor Operations Standard. Status Final. Hanzo architectural specification.
- [`hip-0020`](hip-0020.md) — HIP-20: Blockchain Node Standard. Status Final. Hanzo architectural specification.
- [`hip-0024`](hip-0024.md) — HIP-24: Hanzo Sovereign L1 Chain Architecture. Status Final. Hanzo architectural specification.
- [`hip-0025`](hip-0025.md) — HIP-25: Bot Agent Wallet & RPC Billing Protocol. Status Final. Hanzo architectural specification.
- [`hip-0039`](hip-0039.md) — HIP-39: Zen Model Architecture. Status Final. Hanzo architectural specification.
- [`hip-0043`](hip-0043.md) — HIP-43: Hanzo Engine — LLM Inference Engine Standard. Status Final. Hanzo architectural specification.
- [`hip-0095`](hip-0095.md) — HIP-95: QoS Challenge System. Status Draft. Hanzo architectural specification.
- [`hip-0096`](hip-0096.md) — HIP-96: AI Compute Contribution Rewards. Status Final. Hanzo architectural specification.
- [`hip-0097`](hip-0097.md) — HIP-97: Node Identity and the did:hanzo: DID Method. Status Final. Hanzo architectural specification.
- [`hip-0113`](hip-0113.md) — HIP-113: Cognitive Sidecar & Hanzo Engine Provider Runtime for Thinking Chains. Status Final. Hanzo architectural specification.
- [`hip-0114`](hip-0114.md) — HIP-114: ZAP — Cognitive Transport for Thinking Chains. Status Draft. Hanzo architectural specification.
- [`hip-0120`](hip-0120.md) — HIP-120: ZAP Native Transport and gRPC Elimination. Status Final. Hanzo architectural specification.
- [`hip-0121`](hip-0121.md) — HIP-121: BYO Compute Fleet & Metered Billing. Status Final. Hanzo architectural specification.
- [`hip-0127`](hip-0127.md) — HIP-127: V8 · Open Edition — Architecture, Distribution & the Language Seam. Status Final. Hanzo architectural specification.
- [`hip-0128`](hip-0128.md) — HIP-128: Resource Surface Standard — Generated REST over ZAP. Status Final. Hanzo architectural specification.
- [`hip-0130`](hip-0130.md) — HIP-130: Open Core Split — Tenancy Line and Composition Root. Status Final. Hanzo architectural specification.
- [`hip-0133`](hip-0133.md) — HIP-133: Entity Groups — Placement, Durability, Splitting and Promotion. Status Final. Hanzo architectural specification.
- [`hip-0134`](hip-0134.md) — HIP-134: One Process, One Socket, One Identity. Status Final. Hanzo architectural specification.
- [`hip-0135`](hip-0135.md) — HIP-135: What Is Public. Status Living. Hanzo architectural specification.
- [`hip-0137`](hip-0137.md) — HIP-137: One License. Status Living. Hanzo architectural specification.
- [`hip-0140`](hip-0140.md) — HIP-140: Proposing a Capability. Status Living. Hanzo architectural specification.
- [`hip-0141`](hip-0141.md) — HIP-141: Substitution. Status Final. Hanzo architectural specification.
- [`hip-0301`](hip-0301.md) — HIP-301: Agent Runtime Protocols and Cross Platform Parity. Status Final. Hanzo architectural specification.
- [`hip-0303`](hip-0303.md) — HIP-303: Hanzo adopts LP-0010: Brand Sovereignty and Federation Discovery. Status Living. Hanzo architectural specification.
- [`hip-0510`](hip-0510.md) — HIP-510: Enso — Learned Model Routing and Flywheel. Status Final. Hanzo architectural specification.
- [`hip-0516`](hip-0516.md) — HIP-516: Translate — One Endpoint, Two Tiers, Permissive Weights. Status Final. Hanzo architectural specification.
- [`hip-0517`](hip-0517.md) — HIP-517: Branch Naming — main is the Trunk, Everywhere. Status Living. Hanzo architectural specification.
- [`hip-0522`](hip-0522.md) — HIP-522: The Context Graph — Edges, Decisions and Derivation as One Noun. Status Final. Hanzo architectural specification.
- [`hip-0523`](hip-0523.md) — HIP-523: Rooms — One Store, Two Views, Many Bridges. Status Final. Hanzo architectural specification.
- [`hip-0524`](hip-0524.md) — HIP-524: The Personal Agent — What It May Send Without You. Status Final. Hanzo architectural specification.
- [`hip-0525`](hip-0525.md) — HIP-525: Rendezvous — When Every Participant Is a Machine. Status Draft. Hanzo architectural specification.
- [`hip-0526`](hip-0526.md) — HIP-526: Semantic Memory — The Substrate Under the Graph. Status Draft. Hanzo architectural specification.
- [`hip-0901`](hip-0901.md) — HIP-901: Proof of AI — Native Execution Proofs and Governance. Status Draft. Hanzo architectural specification.
- [`hip-0902`](hip-0902.md) — HIP-902: Proof of Code — Consensus over Git Refs. Status Final. Hanzo architectural specification.
- [`hip-0903`](hip-0903.md) — HIP-903: The Agentic Company — Autonomous Firms on Hanzo. Status Final. Hanzo architectural specification.
- [`hip-1126`](hip-1126.md) — HIP-1126: Framework — The DocType Engine. Status Final. Hanzo architectural specification.
- [`hip-1161`](hip-1161.md) — HIP-1161: Wallet — Custody of Key Material. Status Final. Hanzo architectural specification.
- [`hip-1163`](hip-1163.md) — HIP-1163: x402 — Pay Per Request. Status Final. Hanzo architectural specification.
- [`hip-1164`](hip-1164.md) — HIP-1164: Provisioning — Stores on Demand. Status Final. Hanzo architectural specification.
- [`hip-1165`](hip-1165.md) — HIP-1165: S3 — Buckets and Objects. Status Final. Hanzo architectural specification.
- [`hip-1167`](hip-1167.md) — HIP-1167: Dataset — An Immutable Snapshot. Status Final. Hanzo architectural specification.
- [`hip-1180`](hip-1180.md) — HIP-1180: Link — A Record of Provider Accounts. Status Final. Hanzo architectural specification.
- [`hip-1181`](hip-1181.md) — HIP-1181: Plan — The Tier Catalog. Status Final. Hanzo architectural specification.
- [`hip-1189`](hip-1189.md) — HIP-1189: Web3 — Chain Access. Status Final. Hanzo architectural specification.
- [`hip-1190`](hip-1190.md) — HIP-1190: Event — The Product Analytics Plane. Status Final. Hanzo architectural specification.
- [`hip-1195`](hip-1195.md) — HIP-1195: Market — Rate Oracles and Exchange Pools. Status Final. Hanzo architectural specification.
- [`hip-1327`](hip-1327.md) — HIP-1327: Public Channels — Finding a Room in Another Org. Status Draft. Hanzo architectural specification.

### Standards: Cloud & Native Infrastructure
- [`hip-0014`](hip-0014.md) — HIP-14: Application Deployment Standard. Status Final. Hanzo architectural specification.
- [`hip-0033`](hip-0033.md) — HIP-33: Container Registry Standard. Status Final. Hanzo architectural specification.
- [`hip-0036`](hip-0036.md) — HIP-36: CI/CD Build System Standard. Status Final. Hanzo architectural specification.
- [`hip-0049`](hip-0049.md) — HIP-49: DNS Service Standard. Status Final. Hanzo architectural specification.
- [`hip-0050`](hip-0050.md) — HIP-50: Hanzo Edge — Edge AI Runtime Standard. Status Final. Hanzo architectural specification.
- [`hip-0053`](hip-0053.md) — HIP-53: Visor Monitoring & Supervision Standard. Status Draft. Hanzo architectural specification.
- [`hip-0060`](hip-0060.md) — HIP-60: Serverless Functions (FaaS) Standard. Status Final. Hanzo architectural specification.
- [`hip-0065`](hip-0065.md) — HIP-65: Backup & Disaster Recovery Standard. Status Final. Hanzo architectural specification.
- [`hip-0068`](hip-0068.md) — HIP-68: Ingress Standard. Status Final. Hanzo architectural specification.
- [`hip-0069`](hip-0069.md) — HIP-69: Service Discovery and Automated Bridge. Status Final. Hanzo architectural specification.
- [`hip-0074`](hip-0074.md) — HIP-74: Software Bill of Materials & Git Stamp Standard. Status Final. Hanzo architectural specification.
- [`hip-0077`](hip-0077.md) — HIP-77: Mesh Identity, Gossip & Payments (PQ). Status Final. Hanzo architectural specification.
- [`hip-0078`](hip-0078.md) — HIP-78: Z-Chain — Post Quantum Identity and Attestation Rollup. Status Draft. Hanzo architectural specification.
- [`hip-0079`](hip-0079.md) — HIP-79: Q-Chain — Quasar Finality Block Standard. Status Final. Hanzo architectural specification.
- [`hip-0101`](hip-0101.md) — HIP-101: Hanzo Lux Bridge Protocol Integration. Status Final. Hanzo architectural specification.
- [`hip-0102`](hip-0102.md) — HIP-102: Omnichain Bridge Integration. Status Draft. Hanzo architectural specification.
- [`hip-0103`](hip-0103.md) — HIP-103: Bridge Post Quantum Profile. Status Final. Hanzo architectural specification.
- [`hip-0104`](hip-0104.md) — HIP-104: Contract Auth via Z-Chain Proof. Status Final. Hanzo architectural specification.
- [`hip-0105`](hip-0105.md) — HIP-105: In Process Extension Runtime Standard. Status Final. Hanzo architectural specification.
- [`hip-0106`](hip-0106.md) — HIP-106: The Hanzo Plugin Contract. Status Final. Hanzo architectural specification.
- [`hip-0107`](hip-0107.md) — HIP-107: Streaming Replication over VFS. Status Final. Hanzo architectural specification.
- [`hip-0108`](hip-0108.md) — HIP-108: On Demand Subsystem Supervisor and Warm Pool. Status Final. Hanzo architectural specification.
- [`hip-0109`](hip-0109.md) — HIP-109: Hanzo ML Cloud Toolkit. Status Final. Hanzo architectural specification.
- [`hip-0115`](hip-0115.md) — HIP-115: Hanzo Frontend Delivery. Status Draft. Hanzo architectural specification.
- [`hip-0117`](hip-0117.md) — HIP-117: Cloud in a Box — One Binary Three Modes. Status Final. Hanzo architectural specification.
- [`hip-0119`](hip-0119.md) — HIP-119: Hanzo Service Conventions. Status Final. Hanzo architectural specification.
- [`hip-0122`](hip-0122.md) — HIP-122: zip — The ZAP Native Application Server Core. Status Final. Hanzo architectural specification.
- [`hip-0123`](hip-0123.md) — HIP-123: Visor — Fleet & Fabric Autoscaling Across Any Provider. Status Final. Hanzo architectural specification.
- [`hip-0125`](hip-0125.md) — HIP-125: Consensus Plugin Placement Platform. Status Final. Hanzo architectural specification.
- [`hip-0129`](hip-0129.md) — HIP-129: Eval — A Score Over Model Output. Status Final. Hanzo architectural specification.
- [`hip-0132`](hip-0132.md) — HIP-132: One Telemetry Plane — One Endpoint, One Schema, Many Lenses. Status Final. Hanzo architectural specification.
- [`hip-0136`](hip-0136.md) — HIP-136: One Secret, One Path. Status Final. Hanzo architectural specification.
- [`hip-0139`](hip-0139.md) — HIP-139: Capability. Status Final. Hanzo architectural specification.
- [`hip-0142`](hip-0142.md) — HIP-142: One Manifest, Five Kinds. Status Draft. Hanzo architectural specification.
- [`hip-0144`](hip-0144.md) — HIP-144: Where State Lives. Status Final. Hanzo architectural specification.
- [`hip-0200`](hip-0200.md) — HIP-200: Responsible AI Principles and Commitments. Status Living. Hanzo architectural specification.
- [`hip-0201`](hip-0201.md) — HIP-201: Model Risk Management. Status Living. Hanzo architectural specification.
- [`hip-0210`](hip-0210.md) — HIP-210: Safety Evaluation Framework. Status Living. Hanzo architectural specification.
- [`hip-0220`](hip-0220.md) — HIP-220: Bias Detection & Mitigation. Status Living. Hanzo architectural specification.
- [`hip-0230`](hip-0230.md) — HIP-230: AI Transparency & Explainability. Status Living. Hanzo architectural specification.
- [`hip-0240`](hip-0240.md) — HIP-240: AI Incident Response. Status Living. Hanzo architectural specification.
- [`hip-0250`](hip-0250.md) — HIP-250: Sustainability Standards Alignment Matrix. Status Living. Hanzo architectural specification.
- [`hip-0251`](hip-0251.md) — HIP-251: AI Compute Carbon Footprint. Status Living. Hanzo architectural specification.
- [`hip-0260`](hip-0260.md) — HIP-260: Efficient Model Practices. Status Living. Hanzo architectural specification.
- [`hip-0270`](hip-0270.md) — HIP-270: AI Supply Chain Responsibility. Status Living. Hanzo architectural specification.
- [`hip-0302`](hip-0302.md) — HIP-302: Hanzo Replicate: Encrypted SQLite Durability for Base Services. Status Final. Hanzo architectural specification.
- [`hip-0305`](hip-0305.md) — HIP-305: esign — Shared Database Tenancy. Status Final. Hanzo architectural specification.
- [`hip-0306`](hip-0306.md) — HIP-306: AML Transaction Monitoring, Screening and Case Management. Status Final. Hanzo architectural specification.
- [`hip-0512`](hip-0512.md) — HIP-512: Experiment — The Evidence Plane. Status Final. Hanzo architectural specification.
- [`hip-0519`](hip-0519.md) — HIP-519: One Identity Boundary. Status Final. Hanzo architectural specification.
- [`hip-0520`](hip-0520.md) — HIP-520: Serving Topology — Three Tiers, Horizontally Scalable, Pinned Per Entity. Status Final. Hanzo architectural specification.
- [`hip-1049`](hip-1049.md) — HIP-1049: Validator — A Token Redeemed for a Node. Status Final. Hanzo architectural specification.
- [`hip-1060`](hip-1060.md) — HIP-1060: Pubsub — The Tenant Endpoint on the Bus. Status Final. Hanzo architectural specification.
- [`hip-1061`](hip-1061.md) — HIP-1061: MQ — Queues and Streams. Status Final. Hanzo architectural specification.
- [`hip-1062`](hip-1062.md) — HIP-1062: Tasks — The Durable Run. Status Final. Hanzo architectural specification.
- [`hip-1100`](hip-1100.md) — HIP-1100: Ad — Paid Placement Under a Campaign. Status Final. Hanzo architectural specification.
- [`hip-1101`](hip-1101.md) — HIP-1101: Allowance — The Free Lane's Ceiling. Status Final. Hanzo architectural specification.
- [`hip-1103`](hip-1103.md) — HIP-1103: Audit — Tamper Evident Trail. Status Final. Hanzo architectural specification.
- [`hip-1104`](hip-1104.md) — HIP-1104: Base — The Hosted Backend. Status Final. Hanzo architectural specification.
- [`hip-1105`](hip-1105.md) — HIP-1105: Benchmark — The Measurement Arena. Status Final. Hanzo architectural specification.
- [`hip-1107`](hip-1107.md) — HIP-1107: Bot — A Run on a Surface. Status Final. Hanzo architectural specification.
- [`hip-1113`](hip-1113.md) — HIP-1113: Cloudflare — Edge Network Integration. Status Final. Hanzo architectural specification.
- [`hip-1114`](hip-1114.md) — HIP-1114: Code — Search and Symbols. Status Final. Hanzo architectural specification.
- [`hip-1117`](hip-1117.md) — HIP-1117: Crawl — A Page as Markdown. Status Final. Hanzo architectural specification.
- [`hip-1122`](hip-1122.md) — HIP-1122: Deploy — The GitOps Plane. Status Final. Hanzo architectural specification.
- [`hip-1123`](hip-1123.md) — HIP-1123: Domain — Name Registration. Status Final. Hanzo architectural specification.
- [`hip-1124`](hip-1124.md) — HIP-1124: Engine — The Serving Runtime Lens. Status Final. Hanzo architectural specification.
- [`hip-1127`](hip-1127.md) — HIP-1127: Gateway — Live Edge Policy. Status Final. Hanzo architectural specification.
- [`hip-1132`](hip-1132.md) — HIP-1132: Index — Full Text Search. Status Final. Hanzo architectural specification.
- [`hip-1133`](hip-1133.md) — HIP-1133: Ingress — The Embedded Edge. Status Final. Hanzo architectural specification.
- [`hip-1134`](hip-1134.md) — HIP-1134: KMS — Secret Custody. Status Final. Hanzo architectural specification.
- [`hip-1140`](hip-1140.md) — HIP-1140: ML — Model Serving. Status Final. Hanzo architectural specification.
- [`hip-1144`](hip-1144.md) — HIP-1144: Registry — The Artifact Control Plane. Status Final. Hanzo architectural specification.
- [`hip-1145`](hip-1145.md) — HIP-1145: Research — The Experiment Record. Status Final. Hanzo architectural specification.
- [`hip-1146`](hip-1146.md) — HIP-1146: Sandbox — A Lease on Isolated Compute. Status Final. Hanzo architectural specification.
- [`hip-1151`](hip-1151.md) — HIP-1151: Settings — Product Configuration. Status Final. Hanzo architectural specification.
- [`hip-1152`](hip-1152.md) — HIP-1152: Share — A Public URL for a Local Service. Status Final. Hanzo architectural specification.
- [`hip-1154`](hip-1154.md) — HIP-1154: Sync — Two Endpoints Kept in Step. Status Final. Hanzo architectural specification.
- [`hip-1172`](hip-1172.md) — HIP-1172: Compute — Rented Virtual Hardware. Status Final. Hanzo architectural specification.
- [`hip-1173`](hip-1173.md) — HIP-1173: Network — Zero Trust Network Overlay. Status Final. Hanzo architectural specification.
- [`hip-1191`](hip-1191.md) — HIP-1191: Billing — Invoicing and Metered Usage. Status Final. Hanzo architectural specification.
- [`hip-1192`](hip-1192.md) — HIP-1192: Space — Drives Folders and File Storage. Status Final. Hanzo architectural specification.
- [`hip-1198`](hip-1198.md) — HIP-1198: Graph — The Assertion Plane. Status Final. Hanzo architectural specification.
- [`hip-1211`](hip-1211.md) — HIP-1211: AI — The Model API. Status Final. Hanzo architectural specification.
- [`hip-1212`](hip-1212.md) — HIP-1212: Exec — The Code Interpreter. Status Final. Hanzo architectural specification.
- [`hip-1213`](hip-1213.md) — HIP-1213: Tool — Tool Execution Plane. Status Final. Hanzo architectural specification.
- [`hip-1230`](hip-1230.md) — HIP-1230: Platform — The Container Plane. Status Final. Hanzo architectural specification.
- [`hip-1232`](hip-1232.md) — HIP-1232: Git — Repository Hosting. Status Final. Hanzo architectural specification.
- [`hip-1240`](hip-1240.md) — HIP-1240: O11y — The Observability Plane. Status Final. Hanzo architectural specification.
- [`hip-1241`](hip-1241.md) — HIP-1241: Metrics — One Native Store, Three Signals. Status Final. Hanzo architectural specification.
- [`hip-1250`](hip-1250.md) — HIP-1250: Integration — Connection Registry. Status Final. Hanzo architectural specification.
- [`hip-1251`](hip-1251.md) — HIP-1251: Websearch — The Live Web. Status Final. Hanzo architectural specification.
- [`hip-1252`](hip-1252.md) — HIP-1252: Meet — The Join Decision. Status Final. Hanzo architectural specification.
- [`hip-1253`](hip-1253.md) — HIP-1253: Explorer — Chain Data. Status Final. Hanzo architectural specification.
- [`hip-1310`](hip-1310.md) — HIP-1310: Webhook — Delivery to an Endpoint You Own. Status Final. Hanzo architectural specification.
- [`hip-1313`](hip-1313.md) — HIP-1313: Usage — The Metered Record. Status Final. Hanzo architectural specification.
- [`hip-1320`](hip-1320.md) — HIP-1320: Admin — The Operator Console. Status Final. Hanzo architectural specification.
- [`hip-1321`](hip-1321.md) — HIP-1321: DNS — Zones and Records. Status Final. Hanzo architectural specification.
- [`hip-1323`](hip-1323.md) — HIP-1323: Kafka — A Wire Onto the One Bus. Status Final. Hanzo architectural specification.
- [`hip-1324`](hip-1324.md) — HIP-1324: KV — Buckets of Versioned Values. Status Final. Hanzo architectural specification.
- [`hip-1325`](hip-1325.md) — HIP-1325: Node — A Machine an Org Owns. Status Final. Hanzo architectural specification.
- [`hip-1326`](hip-1326.md) — HIP-1326: AMQP — Exchanges, Queues and Bindings Onto the One Bus. Status Final. Hanzo architectural specification.
- [`hip-1328`](hip-1328.md) — HIP-1328: Forge — The Meta Code Host. Status Draft. Hanzo architectural specification.
- [`hip-1329`](hip-1329.md) — HIP-1329: Plugins — Dynamic Process and Host Control. Status Final. Hanzo architectural specification.

### Standards: Interfaces & Realtime Protocols
- [`hip-0004`](hip-0004.md) — HIP-4: LLM Gateway - Unified AI Provider Interface. Status Final. Hanzo architectural specification.
- [`hip-0010`](hip-0010.md) — HIP-10: Model Context Protocol (MCP) Integration Standards. Status Final. Hanzo architectural specification.
- [`hip-0015`](hip-0015.md) — HIP-15: Computer Control Standard. Status Final. Hanzo architectural specification.
- [`hip-0018`](hip-0018.md) — HIP-18: Payment Processing Standard. Status Draft. Hanzo architectural specification.
- [`hip-0038`](hip-0038.md) — HIP-38: Admin Console Standard. Status Draft. Hanzo architectural specification.
- [`hip-0040`](hip-0040.md) — HIP-40: Multilanguage SDK Standard. Status Final. Hanzo architectural specification.
- [`hip-0041`](hip-0041.md) — HIP-41: The Hanzo CLI — a Projection of the Served API. Status Final. Hanzo architectural specification.
- [`hip-0044`](hip-0044.md) — HIP-44: Answer — Budget, Policy, Audit, Search, KB and Graph in Three SDKs. Status Draft. Hanzo architectural specification.
- [`hip-0045`](hip-0045.md) — HIP-45: Documentation Framework Standard. Status Final. Hanzo architectural specification.
- [`hip-0061`](hip-0061.md) — HIP-61: Notification & Messaging Service Standard. Status Draft. Hanzo architectural specification.
- [`hip-0063`](hip-0063.md) — HIP-63: Feature Flags Standard. Status Final. Hanzo architectural specification.
- [`hip-0126`](hip-0126.md) — HIP-126: Integrations, Connectors & the Extension Runtime — One Registry, One Way. Status Final. Hanzo architectural specification.
- [`hip-0300`](hip-0300.md) — HIP-300: Unified MCP — one endpoint, and local servers that forward to it. Status Final. Hanzo architectural specification.
- [`hip-0504`](hip-0504.md) — HIP-504: Unified Cross Platform Design System. Status Final. Hanzo architectural specification.
- [`hip-0518`](hip-0518.md) — HIP-518: AML — The Obligation Plane. Status Final. Hanzo architectural specification.
- [`hip-1020`](hip-1020.md) — HIP-1020: Chain Registry. Status Final. Hanzo architectural specification.
- [`hip-1021`](hip-1021.md) — HIP-1021: Chain RPC Endpoint. Status Final. Hanzo architectural specification.
- [`hip-1022`](hip-1022.md) — HIP-1022: Native Balance Reads. Status Final. Hanzo architectural specification.
- [`hip-1030`](hip-1030.md) — HIP-1030: OpenAPI — The Served Contract. Status Final. Hanzo architectural specification.
- [`hip-1032`](hip-1032.md) — HIP-1032: Errors — The Fault Lens. Status Final. Hanzo architectural specification.
- [`hip-1040`](hip-1040.md) — HIP-1040: Appearance. Status Final. Hanzo architectural specification.
- [`hip-1042`](hip-1042.md) — HIP-1042: Avatar. Status Final. Hanzo architectural specification.
- [`hip-1063`](hip-1063.md) — HIP-1063: Auto — Flows That Run Themselves. Status Final. Hanzo architectural specification.
- [`hip-1064`](hip-1064.md) — HIP-1064: Flow — The Canvas Plane. Status Final. Hanzo architectural specification.
- [`hip-1066`](hip-1066.md) — HIP-1066: Channel — Unified Inbox. Status Final. Hanzo architectural specification.
- [`hip-1067`](hip-1067.md) — HIP-1067: Destination — An Event Translated to a Sink. Status Final. Hanzo architectural specification.
- [`hip-1069`](hip-1069.md) — HIP-1069: Tel — Numbers, Calls and Messages. Status Final. Hanzo architectural specification.
- [`hip-1102`](hip-1102.md) — HIP-1102: Ask — The Grounded Answer. Status Final. Hanzo architectural specification.
- [`hip-1106`](hip-1106.md) — HIP-1106: Blueprint — The Priced Stack. Status Final. Hanzo architectural specification.
- [`hip-1110`](hip-1110.md) — HIP-1110: Campaign — One Push Across Channels. Status Final. Hanzo architectural specification.
- [`hip-1111`](hip-1111.md) — HIP-1111: Captable — Who Owns What. Status Final. Hanzo architectural specification.
- [`hip-1112`](hip-1112.md) — HIP-1112: Catalog — Service Discovery. Status Final. Hanzo architectural specification.
- [`hip-1115`](hip-1115.md) — HIP-1115: Compliance — Verification of Record. Status Final. Hanzo architectural specification.
- [`hip-1116`](hip-1116.md) — HIP-1116: Content — The Marketing Loop. Status Final. Hanzo architectural specification.
- [`hip-1120`](hip-1120.md) — HIP-1120: CRM — The Sales Pipeline. Status Final. Hanzo architectural specification.
- [`hip-1121`](hip-1121.md) — HIP-1121: Dataroom — Documents Shared by Link. Status Final. Hanzo architectural specification.
- [`hip-1125`](hip-1125.md) — HIP-1125: Esign — Documents Out for Signature. Status Final. Hanzo architectural specification.
- [`hip-1130`](hip-1130.md) — HIP-1130: Guide — The Launch Journey. Status Final. Hanzo architectural specification.
- [`hip-1131`](hip-1131.md) — HIP-1131: Help — The Support Desk. Status Final. Hanzo architectural specification.
- [`hip-1135`](hip-1135.md) — HIP-1135: Legal — Documents Drafted, Signed and Filed. Status Final. Hanzo architectural specification.
- [`hip-1136`](hip-1136.md) — HIP-1136: Marketing — Lifecycle Email. Status Final. Hanzo architectural specification.
- [`hip-1137`](hip-1137.md) — HIP-1137: Marketplace — Listings and Installs. Status Final. Hanzo architectural specification.
- [`hip-1141`](hip-1141.md) — HIP-1141: Pref — One Document Per Person. Status Final. Hanzo architectural specification.
- [`hip-1142`](hip-1142.md) — HIP-1142: Prompt — A Named, Versioned Value. Status Final. Hanzo architectural specification.
- [`hip-1143`](hip-1143.md) — HIP-1143: Referral — An Edge From Referrer to Referee. Status Final. Hanzo architectural specification.
- [`hip-1147`](hip-1147.md) — HIP-1147: Search — Hybrid Retrieval. Status Final. Hanzo architectural specification.
- [`hip-1150`](hip-1150.md) — HIP-1150: SEO — Search Visibility as Data. Status Final. Hanzo architectural specification.
- [`hip-1153`](hip-1153.md) — HIP-1153: Social — Publishing to Connected Channels. Status Final. Hanzo architectural specification.
- [`hip-1155`](hip-1155.md) — HIP-1155: Taxonomy — The Catalogue's Shape. Status Final. Hanzo architectural specification.
- [`hip-1156`](hip-1156.md) — HIP-1156: Template — A Deploy Cut From a Catalog. Status Final. Hanzo architectural specification.
- [`hip-1311`](hip-1311.md) — HIP-1311: Experiment — Arms, Assignment, a Verdict. Status Final. Hanzo architectural specification.
- [`hip-1322`](hip-1322.md) — HIP-1322: Skills — The Discovery Catalogue. Status Final. Hanzo architectural specification.

### Standards: Security, IAM & Cryptography
- [`hip-0005`](hip-0005.md) — HIP-5: Post Quantum Security for AI Infrastructure. Status Final. Hanzo architectural specification.
- [`hip-0026`](hip-0026.md) — HIP-26: Identity & Access Management Standard. Status Draft. Hanzo architectural specification.
- [`hip-0027`](hip-0027.md) — HIP-27: Secrets Management Standard. Status Draft. Hanzo architectural specification.
- [`hip-0054`](hip-0054.md) — HIP-54: Zero Trust Architecture Standard. Status Draft. Hanzo architectural specification.
- [`hip-0084`](hip-0084.md) — HIP-84: Pulsar-M — Threshold ML-DSA DKG & Signing. Status Final. Hanzo architectural specification.
- [`hip-0085`](hip-0085.md) — HIP-85: Wallet PQ Account Type (ML-DSA-65 native, 48-byte AccountID). Status Final. Hanzo architectural specification.
- [`hip-0086`](hip-0086.md) — HIP-86: TxAuthEnvelope (typed PQ transaction signing). Status Final. Hanzo architectural specification.
- [`hip-0087`](hip-0087.md) — HIP-87: PQ Permit (replaces EIP-2612). Status Final. Hanzo architectural specification.
- [`hip-0088`](hip-0088.md) — HIP-88: Session KEM (ML-KEM-768/1024 for P2P). Status Final. Hanzo architectural specification.
- [`hip-0089`](hip-0089.md) — HIP-89: DRBG / Randomness Beacon (SP 800-90A/B). Status Final. Hanzo architectural specification.
- [`hip-0098`](hip-0098.md) — HIP-98: Governance / Upgrade Keys (ML-DSA-87 / SLH-DSA cold roots). Status Final. Hanzo architectural specification.
- [`hip-0111`](hip-0111.md) — HIP-111: Hanzo IAM Authentication Standard. Status Final. Hanzo architectural specification.
- [`hip-0118`](hip-0118.md) — HIP-118: SuperAdmin & Tenant Isolation Model. Status Final. Hanzo architectural specification.
- [`hip-0143`](hip-0143.md) — HIP-143: Egress — The Outbound Trust Boundary. Status Final. Hanzo architectural specification.
- [`hip-0521`](hip-0521.md) — HIP-521: Org Hierarchy. Status Final. Hanzo architectural specification.
- [`hip-1004`](hip-1004.md) — HIP-1004: Licensing — Signed Tokens for Paid Binaries. Status Final. Hanzo architectural specification.
- [`hip-1041`](hip-1041.md) — HIP-1041: Authz. Status Final. Hanzo architectural specification.
- [`hip-1043`](hip-1043.md) — HIP-1043: CSRF. Status Final. Hanzo architectural specification.
- [`hip-1044`](hip-1044.md) — HIP-1044: Org Settings. Status Final. Hanzo architectural specification.
- [`hip-1045`](hip-1045.md) — HIP-1045: Orgs. Status Final. Hanzo architectural specification.
- [`hip-1047`](hip-1047.md) — HIP-1047: Security Scanning. Status Final. Hanzo architectural specification.
- [`hip-1065`](hip-1065.md) — HIP-1065: Connectors — A User's Own Credentials. Status Final. Hanzo architectural specification.
- [`hip-1194`](hip-1194.md) — HIP-1194: Trust — Trust Center and Evidence Controls. Status Final. Hanzo architectural specification.

### Standards: Platform & Applications
- [`hip-0124`](hip-0124.md) — HIP-124: Bring Your Own Provider and AI — Unified Usage. Status Final. Hanzo architectural specification.
- [`hip-0400`](hip-0400.md) — HIP-400: Service CRD. Status Final. Hanzo architectural specification.
- [`hip-0401`](hip-0401.md) — HIP-401: Datastore CRD. Status Final. Hanzo architectural specification.
- [`hip-0402`](hip-0402.md) — HIP-402: SQL CRD. Status Final. Hanzo architectural specification.
- [`hip-0403`](hip-0403.md) — HIP-403: KV CRD. Status Final. Hanzo architectural specification.
- [`hip-0404`](hip-0404.md) — HIP-404: DocDB CRD. Status Final. Hanzo architectural specification.
- [`hip-0405`](hip-0405.md) — HIP-405: S3 CRD. Status Final. Hanzo architectural specification.
- [`hip-0406`](hip-0406.md) — HIP-406: DNS CRD. Status Final. Hanzo architectural specification.
- [`hip-0407`](hip-0407.md) — HIP-407: Base CRD. Status Final. Hanzo architectural specification.
- [`hip-0408`](hip-0408.md) — HIP-408: IAM CRD. Status Final. Hanzo architectural specification.
- [`hip-0409`](hip-0409.md) — HIP-409: KMS CRD. Status Final. Hanzo architectural specification.
- [`hip-0410`](hip-0410.md) — HIP-410: LLM CRD. Status Final. Hanzo architectural specification.
- [`hip-0411`](hip-0411.md) — HIP-411: Ingress CRD. Status Final. Hanzo architectural specification.
- [`hip-0412`](hip-0412.md) — HIP-412: Gateway CRD. Status Final. Hanzo architectural specification.
- [`hip-0413`](hip-0413.md) — HIP-413: MPC CRD. Status Final. Hanzo architectural specification.
- [`hip-0414`](hip-0414.md) — HIP-414: Network CRD. Status Final. Hanzo architectural specification.
- [`hip-0418`](hip-0418.md) — HIP-418: Indexer CRD. Status Final. Hanzo architectural specification.
- [`hip-0419`](hip-0419.md) — HIP-419: Explorer CRD. Status Final. Hanzo architectural specification.
- [`hip-0449`](hip-0449.md) — HIP-449: Hanzo Login. Status Final. Hanzo architectural specification.
- [`hip-0506`](hip-0506.md) — HIP-506: Hanzo Studio — Agentic Multimodal Creative Runtime. Status Final. Hanzo architectural specification.
- [`hip-1000`](hip-1000.md) — HIP-1000: Author — A Royalty Bound to a Repository. Status Final. Hanzo architectural specification.
- [`hip-1001`](hip-1001.md) — HIP-1001: Books — Double Entry Ledger. Status Final. Hanzo architectural specification.
- [`hip-1046`](hip-1046.md) — HIP-1046: Risk. Status Final. Hanzo architectural specification.
- [`hip-1048`](hip-1048.md) — HIP-1048: Team. Status Final. Hanzo architectural specification.
- [`hip-1160`](hip-1160.md) — HIP-1160: Todo — The Work Item Board. Status Final. Hanzo architectural specification.
- [`hip-1162`](hip-1162.md) — HIP-1162: World — The News Feed. Status Final. Hanzo architectural specification.
- [`hip-1193`](hip-1193.md) — HIP-1193: Standing — Entity Upkeep and Annual Filings. Status Final. Hanzo architectural specification.
- [`hip-1200`](hip-1200.md) — HIP-1200: Account — The Caller's Own Surface. Status Final. Hanzo architectural specification.
- [`hip-1201`](hip-1201.md) — HIP-1201: Admission — Launch Control. Status Final. Hanzo architectural specification.
- [`hip-1202`](hip-1202.md) — HIP-1202: Entitlement — What an Org May Run. Status Final. Hanzo architectural specification.
- [`hip-1203`](hip-1203.md) — HIP-1203: Affiliate — Commission on Referred Spend. Status Final. Hanzo architectural specification.
- [`hip-1210`](hip-1210.md) — HIP-1210: Agent — Define and Run Agents. Status Final. Hanzo architectural specification.
- [`hip-1214`](hip-1214.md) — HIP-1214: LSP — Live Code Intelligence. Status Final. Hanzo architectural specification.
- [`hip-1220`](hip-1220.md) — HIP-1220: Commerce — The Merchant Half. Status Final. Hanzo architectural specification.
- [`hip-1221`](hip-1221.md) — HIP-1221: Treasury — The Reserve Fund. Status Final. Hanzo architectural specification.
- [`hip-1222`](hip-1222.md) — HIP-1222: Pricing — The Price List and Who May See It. Status Final. Hanzo architectural specification.
- [`hip-1231`](hip-1231.md) — HIP-1231: Project — Static Site Store. Status Final. Hanzo architectural specification.
- [`hip-1242`](hip-1242.md) — HIP-1242: Leaderboard — Who Uses AI Most. Status Final. Hanzo architectural specification.
- [`hip-1260`](hip-1260.md) — HIP-1260: Knowledge — Wiki and Agent Memory. Status Final. Hanzo architectural specification.
- [`hip-1261`](hip-1261.md) — HIP-1261: Label — Ground Truth. Status Final. Hanzo architectural specification.
- [`hip-1262`](hip-1262.md) — HIP-1262: Reference — Lookup Sets. Status Final. Hanzo architectural specification.
- [`hip-1312`](hip-1312.md) — HIP-1312: Company — The Formation Machine. Status Final. Hanzo architectural specification.
