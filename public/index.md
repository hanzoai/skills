# Hanzo Skills

Everything an agent needs to build on Hanzo, as plain markdown.

    curl hanzoskills.com/llms.txt          # the index, for an agent
    curl hanzoskills.com/<skill>.md        # one skill
    curl hanzoskills.com/all.md            # the whole corpus, one file

Every page is `text/markdown`. Paths work with or without the `.md`.

## Start here

    curl hanzoskills.com/hanzo-cloud-architecture.md

That is the shape of the platform: one binary, per-app plugins, ZAP transport,
and a release train that refuses to ship what it cannot verify.

## Everything

### hanzo-standards
- [`hip-0000`](hip-0000.md) — HIP-0: Hanzo AI Architecture & Framework. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0001`](hip-0001.md) — HIP-1: AI Token - Hanzo's Native Currency. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0002`](hip-0002.md) — HIP-2: Hamiltonian Large Language Models (HLLMs) Specification. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0003`](hip-0003.md) — HIP-3: Jin Multimodal AI Architecture. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0004`](hip-0004.md) — HIP-4: LLM Gateway - Unified AI Provider Interface. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0005`](hip-0005.md) — HIP-5: Post-Quantum Security for AI Infrastructure. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0006`](hip-0006.md) — HIP-6: Per-User Fine-Tuning Architecture for Personalized AI. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0007`](hip-0007.md) — HIP-7: Active Inference Integration for Hamiltonian LLMs. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0008`](hip-0008.md) — HIP-8: HMM (Hanzo Market Maker) - Native DEX for AI Compute Resources. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0009`](hip-0009.md) — HIP-9: Agent SDK - Multi-Agent Orchestration Framework. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0010`](hip-0010.md) — HIP-10: Model Context Protocol (MCP) Integration Standards. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0011`](hip-0011.md) — HIP-11: Chat Interface Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0012`](hip-0012.md) — HIP-12: Search Interface Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0013`](hip-0013.md) — HIP-13: Workflow Execution Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0014`](hip-0014.md) — HIP-14: Application Deployment Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0015`](hip-0015.md) — HIP-15: Computer Control Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0017`](hip-0017.md) — HIP-17: Analytics Event Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0018`](hip-0018.md) — HIP-18: Payment Processing Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0019`](hip-0019.md) — HIP-19: Tensor Operations Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0020`](hip-0020.md) — HIP-20: Blockchain Node Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0024`](hip-0024.md) — HIP-24: Hanzo Sovereign L1 Chain Architecture. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0025`](hip-0025.md) — HIP-25: Bot Agent Wallet & RPC Billing Protocol. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0026`](hip-0026.md) — HIP-26: Identity & Access Management Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0027`](hip-0027.md) — HIP-27: Secrets Management Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0028`](hip-0028.md) — HIP-28: Key-Value Store Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0029`](hip-0029.md) — HIP-29: Relational Database Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0030`](hip-0030.md) — HIP-30: Event Streaming Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0033`](hip-0033.md) — HIP-33: Container Registry Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0035`](hip-0035.md) — HIP-35: Image & Video Generation Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0036`](hip-0036.md) — HIP-36: CI/CD Build System Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0039`](hip-0039.md) — HIP-39: Zen Model Architecture. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0040`](hip-0040.md) — HIP-40: Multi-Language SDK Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0041`](hip-0041.md) — HIP-41: The Hanzo CLI — a Projection of the Served API. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0042`](hip-0042.md) — HIP-42: Vector Search Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0043`](hip-0043.md) — HIP-43: Hanzo Engine — LLM Inference Engine Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0045`](hip-0045.md) — HIP-45: Documentation Framework Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0046`](hip-0046.md) — HIP-46: Embeddings Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0047`](hip-0047.md) — HIP-47: Analytics Datastore Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0049`](hip-0049.md) — HIP-49: DNS Service Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0050`](hip-0050.md) — HIP-50: Hanzo Edge — Edge AI Runtime Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0060`](hip-0060.md) — HIP-60: Serverless Functions (FaaS) Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0061`](hip-0061.md) — HIP-61: Notification & Messaging Service Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0063`](hip-0063.md) — HIP-63: Feature Flags & Experimentation Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0064`](hip-0064.md) — HIP-64: Log Aggregation & Search Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0065`](hip-0065.md) — HIP-65: Backup & Disaster Recovery Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0068`](hip-0068.md) — HIP-68: Ingress Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0069`](hip-0069.md) — HIP-69: Service Discovery & Auto-Bridge. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0074`](hip-0074.md) — HIP-74: Software Bill of Materials & Git Stamp Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0077`](hip-0077.md) — HIP-77: Mesh Identity, Gossip & Payments (PQ). Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0078`](hip-0078.md) — HIP-78: Z-Chain — Post-Quantum Identity & Attestation Rollup. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0079`](hip-0079.md) — HIP-79: Q-Chain — Quasar Finality Block Standard. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0084`](hip-0084.md) — HIP-84: Pulsar-M — Threshold ML-DSA DKG & Signing. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0085`](hip-0085.md) — HIP-85: Wallet PQ Account Type (ML-DSA-65 native, 48-byte AccountID). Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0086`](hip-0086.md) — HIP-86: TxAuthEnvelope (typed PQ transaction signing). Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0087`](hip-0087.md) — HIP-87: PQ Permit (replaces EIP-2612). Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0088`](hip-0088.md) — HIP-88: Session KEM (ML-KEM-768/1024 for P2P). Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0089`](hip-0089.md) — HIP-89: DRBG / Randomness Beacon (SP 800-90A/B). Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0095`](hip-0095.md) — HIP-95: QoS Challenge System. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0096`](hip-0096.md) — HIP-96: AI Compute Contribution Rewards. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0097`](hip-0097.md) — HIP-97: Node Identity and the did:hanzo: DID Method. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0098`](hip-0098.md) — HIP-98: Governance / Upgrade Keys (ML-DSA-87 / SLH-DSA cold roots). Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0101`](hip-0101.md) — HIP-101: Hanzo-Lux Bridge Protocol Integration. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0102`](hip-0102.md) — HIP-102: Omnichain Bridge Integration. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0103`](hip-0103.md) — HIP-103: Bridge PQ-Only Profile. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0104`](hip-0104.md) — HIP-104: Contract Auth via Z-Chain Proof. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0105`](hip-0105.md) — HIP-105: In-Process Extension Runtime Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0106`](hip-0106.md) — HIP-106: The Hanzo Plugin Contract. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0107`](hip-0107.md) — HIP-107: Streaming Replication over VFS. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0108`](hip-0108.md) — HIP-108: On-Demand Subsystem Supervisor + Warm Pool. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0109`](hip-0109.md) — HIP-109: Hanzo ML Cloud Toolkit. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0111`](hip-0111.md) — HIP-111: Hanzo IAM Authentication Standard. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0113`](hip-0113.md) — HIP-113: Cognitive Sidecar & Hanzo Engine Provider Runtime for Thinking Chains. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0114`](hip-0114.md) — HIP-114: ZAP — Inter-VM Cognitive Transport for Thinking Chains. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0115`](hip-0115.md) — HIP-115: Hanzo Frontend Delivery. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0116`](hip-0116.md) — HIP-116: Hanzo Plugin & VM Model. Status Superseded. Hanzo's own standard — read this before implementing against it.
- [`hip-0117`](hip-0117.md) — HIP-117: Cloud-in-a-Box — One Binary, Three Modes. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0118`](hip-0118.md) — HIP-118: SuperAdmin & Tenant Isolation Model. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0119`](hip-0119.md) — HIP-119: Hanzo Service Conventions. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0120`](hip-0120.md) — HIP-120: ZAP-Native Transport & gRPC Elimination. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0121`](hip-0121.md) — HIP-121: BYO Compute Fleet & Metered Billing. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0122`](hip-0122.md) — HIP-122: zip — The ZAP-Native Application Server Core. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0123`](hip-0123.md) — HIP-123: Visor — Fleet & Fabric Autoscaling Across Any Provider. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0124`](hip-0124.md) — HIP-124: Bring-Your-Own Provider & AI — Unified Dashboard and Usage. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0125`](hip-0125.md) — HIP-125: Consensus-Backed Plugin-Placement Platform. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0126`](hip-0126.md) — HIP-126: Integrations, Connectors & the Extension Runtime — One Registry, One Way. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0127`](hip-0127.md) — HIP-127: V8 · Open Edition — Architecture, Distribution & the Language Seam. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0128`](hip-0128.md) — HIP-128: Resource Surface Standard — Generated REST over ZAP. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0129`](hip-0129.md) — HIP-129: Eval — The Judgment Plane. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0130`](hip-0130.md) — HIP-130: Open-Core Split — the Tenancy Line, the Composition Root, and Lazy Subsystems. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0132`](hip-0132.md) — HIP-132: One Telemetry Plane — One Door, One Schema, Many Lenses. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0133`](hip-0133.md) — HIP-133: Entity Groups — Placement, Durability, Splitting and Promotion. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0134`](hip-0134.md) — HIP-134: One Process, One Socket, One Identity. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0135`](hip-0135.md) — HIP-135: What Is Public. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0136`](hip-0136.md) — HIP-136: One Secret, One Path. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0137`](hip-0137.md) — HIP-137: One License. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0138`](hip-0138.md) — HIP-138: Cloud — Unified Hanzo Binary. Status Final. Hanzo's own standard — read this before implementing against it.
- [`hip-0200`](hip-0200.md) — HIP-200: Responsible AI Principles and Commitments. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0201`](hip-0201.md) — HIP-201: Model Risk Management. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0210`](hip-0210.md) — HIP-210: Safety Evaluation Framework. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0220`](hip-0220.md) — HIP-220: Bias Detection & Mitigation. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0230`](hip-0230.md) — HIP-230: AI Transparency & Explainability. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0240`](hip-0240.md) — HIP-240: AI Incident Response. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0250`](hip-0250.md) — HIP-250: Sustainability Standards Alignment Matrix. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0251`](hip-0251.md) — HIP-251: AI Compute Carbon Footprint. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0260`](hip-0260.md) — HIP-260: Efficient Model Practices. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0270`](hip-0270.md) — HIP-270: AI Supply Chain Responsibility. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0300`](hip-0300.md) — HIP-300: Unified MCP — one door, and local servers that forward to it. Status Review. Hanzo's own standard — read this before implementing against it.
- [`hip-0301`](hip-0301.md) — HIP-301: Agent Runtime Protocols & Cross-Platform Parity. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0302`](hip-0302.md) — HIP-302: Hanzo Replicate: Encrypted SQLite Durability for Base Services. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0303`](hip-0303.md) — HIP-303: Hanzo adopts LP-0010: Brand Sovereignty and Federation Discovery. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0305`](hip-0305.md) — HIP-305: esign: shared-DB tenancy via team-where, not file-per-tenant. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0306`](hip-0306.md) — HIP-306: AML Transaction Monitoring, Screening and Case Management. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0400`](hip-0400.md) — HIP-400: Service CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0401`](hip-0401.md) — HIP-401: Datastore CRD. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0402`](hip-0402.md) — HIP-402: SQL CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0403`](hip-0403.md) — HIP-403: KV CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0404`](hip-0404.md) — HIP-404: DocDB CRD. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0405`](hip-0405.md) — HIP-405: S3 CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0406`](hip-0406.md) — HIP-406: DNS CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0407`](hip-0407.md) — HIP-407: Base CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0408`](hip-0408.md) — HIP-408: IAM CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0409`](hip-0409.md) — HIP-409: KMS CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0410`](hip-0410.md) — HIP-410: LLM CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0411`](hip-0411.md) — HIP-411: Ingress CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0412`](hip-0412.md) — HIP-412: Gateway CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0413`](hip-0413.md) — HIP-413: MPC CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0414`](hip-0414.md) — HIP-414: Network CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0418`](hip-0418.md) — HIP-418: Indexer CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0419`](hip-0419.md) — HIP-419: Explorer CRD. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0504`](hip-0504.md) — HIP-504: Unified Cross-Platform Design System. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0506`](hip-0506.md) — HIP-506: Hanzo Studio — Agentic Multi-Modal Creative Runtime. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0512`](hip-0512.md) — HIP-512: Experiment — The Evidence Plane. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0516`](hip-0516.md) — HIP-516: Translate — One Endpoint, Two Tiers, Permissive Weights. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0517`](hip-0517.md) — HIP-517: Branch Naming — main is the Trunk, Everywhere. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0518`](hip-0518.md) — HIP-518: AML — The Obligation Plane. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0519`](hip-0519.md) — HIP-519: One Identity Boundary. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0520`](hip-0520.md) — HIP-520: Serving Topology — Three Tiers, Horizontally Scalable, Pinned Per Entity. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0521`](hip-0521.md) — HIP-521: Org Hierarchy. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0522`](hip-0522.md) — HIP-522: The Context Graph — Edges, Decisions and Derivation as One Noun. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0901`](hip-0901.md) — HIP-901: Proof of AI (PoAI) — Native Execution Proofs, Canonical Contract & Operator-LLM Governance. Status Active. Hanzo's own standard — read this before implementing against it.
- [`hip-0902`](hip-0902.md) — HIP-902: Proof of Code — Consensus over Git Refs. Status Draft. Hanzo's own standard — read this before implementing against it.
- [`hip-0903`](hip-0903.md) — HIP-903: The Agentic Company — Autonomous Firms on Hanzo. Status Draft. Hanzo's own standard — read this before implementing against it.

### core
- [`anti-slop`](anti-slop.md) — Comprehensive toolkit for detecting and eliminating "AI slop" - generic, low-quality AI-generated patterns in natural language, code, and design. Use when reviewing or improving content quality, preventing generic AI patterns, cleaning up existing content, or enforcing quality standards in writing, code, or design work.
- [`cloud-kubernetes-deployment`](cloud-kubernetes-deployment.md) — Deploy and manage applications on Kubernetes with production best practices
- [`discover-api`](discover-api.md) — Automatically discover API design skills when working with REST APIs, GraphQL schemas, API authentication, OAuth, JWT, rate limiting, API versioning, error handling, or endpoint design. Activates for backend API development tasks.
- [`discover-build-systems`](discover-build-systems.md) — Automatically discover build system skills when working with build systems. Activates for build-systems development tasks.
- [`discover-caching`](discover-caching.md) — Automatically discover caching and performance skills when working with caching. Activates for caching development tasks.
- [`discover-cicd`](discover-cicd.md) — Automatically discover CI/CD and automation skills when working with CI/CD. Activates for cicd development tasks.
- [`discover-cloud`](discover-cloud.md) — Automatically discover cloud computing and serverless skills when working with cloud. Activates for cloud development tasks.
- [`discover-collaboration`](discover-collaboration.md) — Automatically discover collaboration and teamwork skills when working with collaboration. Activates for collaboration development tasks.
- [`discover-containers`](discover-containers.md) — Automatically discover container and Docker skills when working with Docker. Activates for containers development tasks.
- [`discover-cryptography`](discover-cryptography.md) — Automatically discover cryptography skills when working with encryption, TLS, certificates, PKI, and security
- [`discover-data`](discover-data.md) — Automatically discover data pipeline and ETL skills when working with ETL. Activates for data development tasks.
- [`discover-database`](discover-database.md) — Automatically discover database skills when working with SQL, PostgreSQL, MongoDB, Redis, database schema design, query optimization, migrations, connection pooling, ORMs, or database selection. Activates for database design, optimization, and implementation tasks.
- [`discover-debugging`](discover-debugging.md) — Automatically discover debugging and profiling skills when working with debugging. Activates for debugging development tasks.
- [`discover-deployment`](discover-deployment.md) — Automatically discover deployment and release skills when working with deployment. Activates for deployment development tasks.
- [`discover-diagrams`](discover-diagrams.md) — Automatically discover diagram and visualization skills when working with flowcharts, sequence diagrams, architecture diagrams, ER diagrams, Gantt charts, mindmaps, or any visual documentation using Mermaid.js. Activates for documentation and visualization tasks.
- [`discover-distributed-systems`](discover-distributed-systems.md) — Automatically discover distributed systems skills when working with consensus, CRDTs, replication, partitioning, and distributed algorithms
- [`discover-ebpf`](discover-ebpf.md) — Automatically discover eBPF and kernel skills when working with eBPF. Activates for ebpf development tasks.
- [`discover-engineering`](discover-engineering.md) — Automatically discover software engineering practice skills when working with engineering practices. Activates for engineering development tasks.
- [`discover-formal`](discover-formal.md) — Automatically discover formal methods and verification skills when working with formal methods. Activates for formal development tasks.
- [`discover-frontend`](discover-frontend.md) — Automatically discover frontend development skills when working with React, Next.js, UI components, state management, data fetching, forms, accessibility, performance optimization, or SEO. Activates for frontend web development tasks.
- [`discover-hanzo`](discover-hanzo.md) — Gateway to the Hanzo ecosystem — AI infrastructure, local AI, MCP, privacy-preserving inference and agentic workflows. Use when a task mentions Hanzo or needs orientation before reaching for a more specific Hanzo skill.
- [`discover-hanzo-cloud-architecture`](discover-hanzo-cloud-architecture.md) — Automatically discover the Hanzo Cloud one-binary ZAP-native architecture skill when working on hanzoai services, transports, or deployment topology.
- [`discover-infrastructure`](discover-infrastructure.md) — Automatically discover infrastructure and DevOps skills when working with infrastructure. Activates for infrastructure development tasks.
- [`discover-ir`](discover-ir.md) — Automatically discover intermediate representation and compiler skills when working with intermediate representation. Activates for ir development tasks.
- [`discover-math`](discover-math.md) — Automatically discover mathematics and algorithm skills when working with mathematics. Activates for math development tasks.
- [`discover-ml`](discover-ml.md) — Automatically discover machine learning and AI skills when working with machine learning. Activates for ml development tasks.
- [`discover-mobile`](discover-mobile.md) — Automatically discover mobile development skills when working with iOS. Activates for mobile development tasks.
- [`discover-modal`](discover-modal.md) — Automatically discover Modal platform skills when working with Modal. Activates for modal development tasks.
- [`discover-networking`](discover-networking.md) — Automatically discover networking and connectivity skills when working with networking. Activates for networking development tasks.
- [`discover-observability`](discover-observability.md) — Automatically discover observability and monitoring skills when working with monitoring. Activates for observability development tasks.
- [`discover-plt`](discover-plt.md) — Automatically discover programming language theory skills when working with compilers. Activates for plt development tasks.
- [`discover-product`](discover-product.md) — Automatically discover product management skills when working with product management. Activates for product development tasks.
- [`discover-protocols`](discover-protocols.md) — Automatically discover protocol skills when working with HTTP, TCP, UDP, QUIC, and network protocols
- [`discover-proxies`](discover-proxies.md) — Automatically discover proxy skills when working with forward/reverse proxies, nginx, traefik, envoy, caching, or NATS messaging
- [`discover-realtime`](discover-realtime.md) — Automatically discover realtime communication skills when working with realtime. Activates for realtime development tasks.
- [`discover-research`](discover-research.md) — Gateway skill for research methodology - automatically loads relevant research skills based on task context
- [`discover-security`](discover-security.md) — Automatically discover security skills when working with authentication, authorization, input validation, security headers, vulnerability assessment, or secrets management. Activates for application security, OWASP, and security hardening tasks.
- [`discover-testing`](discover-testing.md) — Automatically discover testing skills when working with testing. Activates for testing development tasks.
- [`discover-tui`](discover-tui.md) — Automatically discover terminal UI skills when working with terminal user interfaces. Activates for tui development tasks.
- [`discover-wasm`](discover-wasm.md) — Automatically discover WebAssembly skills when working with WebAssembly. Activates for wasm development tasks.
- [`discover-workflow`](discover-workflow.md) — Automatically discover workflow and task management skills when working with Beads. Activates for workflow development tasks.
- [`discover-zig`](discover-zig.md) — Automatically discover Zig programming skills when working with Zig. Activates for zig development tasks.
- [`elegant-design`](elegant-design.md) — Create world-class, accessible, responsive interfaces with sophisticated interactive elements including chat, terminals, code display, and streaming content. Use when building user interfaces that need professional polish and developer-focused features.
- [`hanzo-cloud-architecture`](hanzo-cloud-architecture.md) — The canonical Hanzo Cloud architecture — one repo, host + per-app plugin binaries on zip, ZAP-only transport, router-projected OpenAPI, the receipt release train. Read this before touching any hanzoai service.
- [`skill-creation`](skill-creation.md) — Creating new atomic skills for the skills system
- [`skill-prompt-discovery`](skill-prompt-discovery.md) — User makes a request involving specific technologies or frameworks
- [`skill-prompt-planning`](skill-prompt-planning.md) — This skill enables analyzing user prompts, requests, and conversation patterns to identify gaps in the skills catalog and plan creation of new skills that would be genuinely useful.
- [`skill-repo-discovery`](skill-repo-discovery.md) — Starting work on an unfamiliar repository or codebase
- [`skill-repo-planning`](skill-repo-planning.md) — This skill enables systematic analysis of codebases to identify skill gaps—domains, technologies, patterns, and workflows present in repositories that aren't covered by existing skills.
- [`typed-holes-refactor`](typed-holes-refactor.md) — Refactor codebases using Design by Typed Holes methodology - iterative, test-driven refactoring with formal hole resolution, constraint propagation, and continuous validation. Use when refactoring existing code, optimizing architecture, or consolidating technical debt through systematic hole-driven development.

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

### engineering
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

### cryptography
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

### protocols
- [`protocols-amqp-rabbitmq`](protocols-amqp-rabbitmq.md) — RabbitMQ and AMQP 0-9-1 message broker implementation
- [`protocols-grpc-implementation`](protocols-grpc-implementation.md) — Implementing gRPC APIs with Protocol Buffers
- [`protocols-http2-multiplexing`](protocols-http2-multiplexing.md) — HTTP/2 protocol with multiplexing, server push, header compression, and stream prioritization
- [`protocols-kafka-streams`](protocols-kafka-streams.md) — Apache Kafka stream processing and event streaming platform
- [`protocols-mqtt-messaging`](protocols-mqtt-messaging.md) — MQTT pub-sub messaging for IoT and real-time applications
- [`protocols-protobuf-schemas`](protocols-protobuf-schemas.md) — Protocol Buffers schema design, evolution, and code generation
- [`protocols-websocket-protocols`](protocols-websocket-protocols.md) — WebSocket protocol implementation, scaling, and production deployment

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

### collaboration
- [`codetour-guided-walkthroughs`](codetour-guided-walkthroughs.md) — Creating and following CodeTour walkthroughs for codebase understanding
- [`collaboration-github-actions-workflows`](collaboration-github-actions-workflows.md) — Setting up CI/CD pipelines with GitHub Actions
- [`collaboration-github-issues-projects`](collaboration-github-issues-projects.md) — Issue management, templates, labels, milestones, GitHub Projects, and issue linking workflows
- [`collaboration-github-pull-requests`](collaboration-github-pull-requests.md) — Pull request workflow, code review, merge strategies, PR checks, and collaboration best practices
- [`collaboration-github-repository-management`](collaboration-github-repository-management.md) — Repository creation, configuration, branch protection, tags, releases, and GitHub CLI operations
- [`collaboration-github-security-features`](collaboration-github-security-features.md) — Dependabot, code scanning, secret scanning, SBOM generation, security advisories, and security best practices

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

### containers
- [`containers-container-networking`](containers-container-networking.md) — Configuring container network connectivity
- [`containers-container-registry-management`](containers-container-registry-management.md) — Choosing a container registry (ECR/GCR/Harbor/Docker Hub)
- [`containers-container-security`](containers-container-security.md) — Hardening container images for production
- [`containers-docker-compose-development`](containers-docker-compose-development.md) — Setting up local development environments
- [`containers-dockerfile-optimization`](containers-dockerfile-optimization.md) — Writing Dockerfiles for production applications

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

### anti-slop
- [`code-patterns`](code-patterns.md) — Reference guide for detecting AI slop patterns in code including generic variable names, obvious comments, and unnecessary abstractions. Use as reference when reviewing code quality.
- [`design-patterns`](design-patterns.md) — Reference guide for detecting AI slop patterns in visual and UX design including generic gradients, cookie-cutter layouts, and buzzword-heavy copy. Use as reference when reviewing design quality.
- [`text-patterns`](text-patterns.md) — Reference guide for detecting AI slop patterns in natural language text including overused phrases, buzzwords, and meta-commentary. Use as reference when reviewing text content for quality.

### hanzo
- [`skill`](skill.md) — Onboarding manifest for an AI agent operating Hanzo — install, authenticate with a human operator, and work the cloud safely. Read this before running anything against api.hanzo.ai.
