# Hanzo Improvement Proposals (HIPs)

Hanzo Improvement Proposals (HIPs) are the primary mechanism for proposing new features, gathering community input, and documenting design decisions for the Hanzo AI ecosystem. HIPs serve as formal design documents that provide information to the community about proposed changes to the platform.

Repository: github.com/hanzoai/hips
Website: hips.hanzo.ai
License: CC0 1.0 Universal Public Domain Dedication

## What HIPs Cover

- AI architectures, models, and infrastructure
- Standards for agent frameworks, protocols, and interfaces
- Blockchain and consensus mechanisms
- Infrastructure services (databases, security, networking)
- Governance and responsible AI practices
- Cross-chain integrations with Lux and Zoo ecosystems

## HIP Types

- **Standards Track**: Technical changes affecting Hanzo AI (subtypes: Core, Interface, Infrastructure, Bridge, Security)
- **Meta**: Process proposals, governance, responsible AI
- **Informational**: Guidelines and best practices

## HIP Lifecycle

Draft -> Review -> Last Call (14-day final review) -> Final (or Withdrawn/Superseded)

## Core Protocol HIPs (Root Level)

### HIP-001: Post-Quantum Cryptography Standard
- **Status**: Implemented
- **Category**: Security
- Establishes post-quantum cryptography using ML-DSA-87 (FIPS 204) for digital signatures and ML-KEM-768 (FIPS 203) for key encapsulation
- All implementations use the `hanzo-crypto` library (pure Rust, constant-time, automatic secret zeroization)
- Hybrid signatures (classical + PQ) supported during migration from Ed25519/X25519

### HIP-002: AI/Blockchain Convergence Architecture
- **Status**: In Development
- **Category**: Core
- Defines on-chain model registry, decentralized multi-node inference with BFT consensus, zero-knowledge ML proofs (zkSNARKs for small models, optimistic rollups for large), and an inference marketplace with reputation-based node selection and slashing

### HIP-002-ASO: Active Semantic Optimization Protocol
- **Status**: Active
- **Category**: Core
- Training-free adaptation framework for agentic code generation using TF-GRPO (Training-Free Group-Relative Policy Optimization) and decode-time Product-of-Experts (PoE) ensemble
- 1-bit compression via BitDelta (29.5x compression, 1 bit per element + 1 scalar per matrix)
- SWE-bench Verified: 18.2% resolved rate (vs 12.5% Claude 3.5 Sonnet agentic, 8.3% GPT-4 zero-shot)
- CLI: `hanzo dev solve <issue_file> --repo <path> --group-size 4 --test-cmd "pytest"`
- Extended by Zoo's ZIP-001 (Decentralized Semantic Optimization)

### HIP-003: Model Context Protocol (MCP) Integration
- **Status**: Active
- **Category**: Interface
- Standardizes MCP across all Hanzo services with stdio, HTTP, and WebSocket transports
- Tool categories: Blockchain (query/submit/verify/subscribe), AI (list/deploy/inference), Cryptography (ML-DSA sign, ML-KEM encrypt, BLAKE3 hash), Storage (IPFS/Arweave)
- All tool calls signed with ML-DSA, parameters encrypted with ML-KEM

### HIP-004: Hamiltonian Market Maker (HMM)
- **Status**: Active
- **Category**: Core
- Automated market maker for pricing heterogeneous AI compute resources via conserved Hamiltonian invariants
- Single-asset invariant: H(Psi, Theta) = Psi * Theta = kappa (constant-product)
- Multi-asset: H(Psi, Theta) = sum(wi * Psi_i * Theta_i) + lambda * sum((Psi_i^2 + Theta_i^2)/2)
- Oracle-free pricing, risk-adjusted fees, PoAI-integrated settlement
- Job lifecycle: Escrow -> Allocation -> Execution -> Attestation (PoAI) -> Verification -> Settlement
- Testnet: 182ms quote latency, 98.7% price stability, +15.3% capital efficiency vs oracle-based

### HIP-005: KMS Hardware Security Module Integration
- **Status**: Active
- **Category**: Infrastructure
- Extends Lux KMS (LP-325) with AI-specific features: model weight encryption (AES-256-GCM), PoAI attestation signing, HMM settlement key management
- HSM providers: Google Cloud KMS, AWS CloudHSM, YubiHSM 2 FIPS, Zymbit HSM6, SoftHSM2 (dev)
- Multi-tenant key isolation with per-model/per-customer namespaces
- Go API: `github.com/hanzoai/kms`

### HIP-006: AI Mining Protocol
- **Status**: Draft
- **Category**: Core
- Native AI compute mining on Hanzo L1 with quantum-safe ML-DSA wallets
- NVTrust chain-binding prevents double-spend: each unit of AI work bound to a specific chain BEFORE compute runs
- GPU TEE receipts signed by NVIDIA hardware attestation (H100, H200, B100, B200, GB200 supported)
- Rewards teleportable to Hanzo EVM (36963), Zoo EVM (200200), or Lux C-Chain (96369)
- EVM precompile at 0x0300 for on-chain mining balance and ML-DSA verification
- Reference implementation: `github.com/hanzoai/node` (hanzo-mining crate)

### HIP-007: ZAP (Zero-copy Agent Protocol)
- **Status**: Draft
- **Category**: Interface
- High-performance binary RPC protocol using Cap'n Proto serialization, replacing JSON-RPC for AI agent communication
- 10-100x performance over MCP JSON-RPC: 0.2us parse latency vs 45us (1KB), 0 memory allocations per message
- MCP-compatible semantics (tools, resources, prompts) with gateway bridging to existing MCP servers
- Transports: TCP (zap://), TLS (zaps://), Unix socket, WebSocket, HTTP/2, stdio
- Promise pipelining: chain RPC calls without round-trip latency
- Also adopted by Lux Network (LP-120) for VM-Node communication, consensus voting
- Packages: `hanzo-zap` (Rust), `@hanzo/zap` (TypeScript), `hanzo-zap` (Python)

### HIP-008: Unified Payment Platform
- **Status**: Draft
- **Category**: Infrastructure
- PCI DSS v4.0 compliant self-hosted payment infrastructure with three-zone isolation
- Zone 3 (Vault CDE): card tokenization with AES-256-GCM envelope encryption, network tokenization (Visa VTS, Mastercard MDES)
- Zone 2 (Commerce): payment intent orchestration, multi-processor routing (Stripe, Adyen, Square, PayPal, Braintree), double-entry ledger, subscription engine (Temporal workflows)
- Crypto payments: BTC (on-chain + Lightning), ETH, ERC-20 (custodial and non-custodial)
- mTLS with SPIFFE/SPIRE identity attestation between zones
- Repositories: `hanzoai/vault`, `hanzoai/commerce`

### HIP-009: Unified Agent Skills Architecture
- **Status**: Draft
- **Category**: Interface
- `~/.hanzo/skills/` as canonical source of truth for all AI agent skills
- Automatic symlink distribution to Claude Code, Cursor, Codex, Openclaw, Hanzo Bot directories
- Follows open Agent Skills specification (agentskills.io)
- Install: `npx @hanzo/bot skills add org/repo`
- Multi-brand support: Bootnode, Hanzo Web3, Lux Cloud, Zoo Labs, Pars Cloud

## Expanded HIPs (HIPs/ Subdirectory)

### Foundation and Models (HIP-0000 to HIP-0010)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0000 | Hanzo AI Architecture and Framework | Final |
| HIP-0001 | AI Token (Hanzo Native Currency) | Draft |
| HIP-0002 | Hamiltonian Large Language Models (HLLMs) | Draft |
| HIP-0003 | Jin Multimodal AI Architecture | Draft |
| HIP-0004 | LLM Gateway (Unified AI Provider Interface) | Draft |
| HIP-0005 | Post-Quantum Security for AI Infrastructure | Final |
| HIP-0006 | Per-User Fine-Tuning Architecture | Draft |
| HIP-0007 | Active Inference Integration for HLLMs | Draft |
| HIP-0008 | HMM Native DEX for AI Compute Resources | Draft |
| HIP-0009 | Agent SDK (Multi-Agent Orchestration) | Draft |
| HIP-0010 | MCP Integration Standards | Draft |

### Application Interfaces (HIP-0011 to HIP-0025)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0011 | Chat Interface Standard | Draft |
| HIP-0012 | Search Interface Standard | Draft |
| HIP-0013 | Workflow Execution Standard | Draft |
| HIP-0014 | Application Deployment Standard | Draft |
| HIP-0015 | Computer Control Standard | Draft |
| HIP-0016 | Document Processing Standard | Draft |
| HIP-0017 | Analytics Event Standard | Draft |
| HIP-0018 | Payment Processing Standard | Draft |
| HIP-0019 | Tensor Operations Standard | Draft |
| HIP-0020 | Blockchain Node Standard | Draft |
| HIP-0021 | Hanzo IDE | Draft |
| HIP-0022 | Personalized AI (Own Your AI) | Draft |
| HIP-0023 | Decentralized AI Compute Swarm Protocol | Draft |
| HIP-0024 | Hanzo Sovereign L1 Chain Architecture | Final |
| HIP-0025 | Bot Agent Wallet and RPC Billing Protocol | Draft |

### Infrastructure Services (HIP-0026 to HIP-0068)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0026 | Identity and Access Management Standard | Draft |
| HIP-0027 | Secrets Management Standard | Draft |
| HIP-0028 | Key-Value Store Standard | Draft |
| HIP-0029 | Relational Database Standard | Draft |
| HIP-0030 | Event Streaming Standard | Draft |
| HIP-0031 | Observability and Metrics Standard | Draft |
| HIP-0032 | Object Storage Standard | Draft |
| HIP-0033 | Container Registry Standard | Draft |
| HIP-0034 | Automation Platform Standard | Draft |
| HIP-0035 | Image and Video Generation Standard | Draft |
| HIP-0036 | CI/CD Build System Standard | Draft |
| HIP-0037 | AI Cloud Platform Standard | Draft |
| HIP-0038 | Admin Console Standard | Draft |
| HIP-0039 | Zen Model Architecture | Draft |
| HIP-0040 | Multi-Language SDK Standard | Draft |
| HIP-0041 | CLI Standard | Draft |
| HIP-0042 | Vector Search Standard | Draft |
| HIP-0043 | LLM Inference Engine Standard | Active |
| HIP-0044 | API Gateway Standard | Active |
| HIP-0045 | Documentation Framework Standard | Draft |
| HIP-0046 | Embeddings Standard | Draft |
| HIP-0047 | Analytics Datastore Standard | Draft |
| HIP-0048 | Decentralized Identity (DID) Standard | Draft |
| HIP-0049 | DNS Service Standard | Draft |
| HIP-0050 | Edge AI Runtime Standard | Active |
| HIP-0051 | Guard Security Standard | Draft |
| HIP-0052 | Nexus Integration Hub Standard | Draft |
| HIP-0053 | Visor Monitoring Standard | Draft |
| HIP-0054 | Zero Trust Architecture Standard | Draft |
| HIP-0055 | Message Queue Standard | Draft |
| HIP-0056 | PubSub Real-Time Messaging Standard | Draft |
| HIP-0057 | ML Pipeline and Training Standard | Draft |
| HIP-0058 | Unified Database Abstraction Standard | Draft |
| HIP-0059 | Timeseries Database Standard | Draft |
| HIP-0060 | Serverless Functions (FaaS) Standard | Draft |
| HIP-0061 | Notification Service Standard | Draft |
| HIP-0062 | Cron and Job Scheduler Standard | Draft |
| HIP-0063 | Feature Flags Standard | Draft |
| HIP-0064 | Log Aggregation Standard | Draft |
| HIP-0065 | Backup and Disaster Recovery Standard | Draft |
| HIP-0066 | Data Governance Standard | Draft |
| HIP-0067 | Federated Learning Standard | Draft |
| HIP-0068 | Ingress Standard | Active |

### Quantum Computing (HIP-0070 to HIP-0073)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0070 | Quantum Computing Integration | Draft |
| HIP-0071 | Quantum Key Distribution | Draft |
| HIP-0072 | Quantum Machine Learning | Draft |
| HIP-0073 | Quantum Random Number Generation | Draft |

### Governance and Supply Chain (HIP-0074 to HIP-0076)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0074 | Software Bill of Materials Standard | Draft |
| HIP-0075 | OSS Contributor Payout Standard | Draft |
| HIP-0076 | Open AI Protocol Standard | Draft |

### Robotics and Physical AI (HIP-0080 to HIP-0083)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0080 | Robotics and Embodied AI Integration | Draft |
| HIP-0081 | Computer Vision Pipeline Standard | Draft |
| HIP-0082 | Digital Twin and Simulation Standard | Draft |
| HIP-0083 | Sensor Fusion and SLAM Standard | Draft |

### Biotech and Life Sciences (HIP-0090 to HIP-0096)

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0090 | Brain-Computer Interface (BCI) Standard | Draft |
| HIP-0091 | Genomics Pipeline Standard | Draft |
| HIP-0092 | Drug Discovery AI Pipeline Standard | Draft |
| HIP-0093 | Synthetic Biology and DNA Data Storage | Draft |
| HIP-0094 | Medical AI and Clinical Decision Support | Draft |
| HIP-0095 | QoS Challenge System | Draft |
| HIP-0096 | AI Compute Contribution Rewards | Draft |

### Cross-Chain, Responsible AI, and Architecture

| HIP | Title | Status |
|-----|-------|--------|
| HIP-0101 | Hanzo-Lux Bridge Protocol Integration | Draft |
| HIP-0200 | Responsible AI Principles | Draft |
| HIP-0201 | Model Risk Management | Draft |
| HIP-0210 | Safety Evaluation Framework | Draft |
| HIP-0220 | Bias Detection and Mitigation | Draft |
| HIP-0230 | AI Transparency and Explainability | Draft |
| HIP-0240 | AI Incident Response | Draft |
| HIP-0250 | Sustainability Standards Alignment | Draft |
| HIP-0251 | AI Compute Carbon Footprint | Draft |
| HIP-0260 | Efficient Model Practices | Draft |
| HIP-0270 | AI Supply Chain Responsibility | Draft |
| HIP-0280 | AI for Sustainability | Draft |
| HIP-0290 | Evidence Locker Index | Draft |
| HIP-0295 | Hanzo AI Impact Thesis | Draft |
| HIP-0300 | Unified MCP Tools Architecture | Draft |

## Key Protocols

### ASO (Active Semantic Optimization) - HIP-002-ASO
Training-free adaptation for code generation. Uses grouped rollouts to extract semantic advantages, compresses them into token-level expert factors via BitDelta (1-bit quantization), and applies them at decode time via Product-of-Experts. No weight updates needed. Extended by Zoo's DSO (ZIP-001) for decentralized aggregation.

### HMM (Hamiltonian Market Maker) - HIP-004
Prices heterogeneous AI compute (GPU, VRAM, CPU, network, disk) using physics-inspired Hamiltonian invariants. Oracle-free: price emerges from the constant H = kappa constraint. Integrated with Zoo's PoAI (ZIP-002) for quality verification. Supports multi-asset routing with SLA constraints solved as a convex program.

### ZAP (Zero-copy Agent Protocol) - HIP-007
Binary RPC protocol built on Cap'n Proto. Zero memory allocations per message parse. Drop-in replacement for MCP JSON-RPC with full semantic compatibility and gateway bridging. Used for agent-to-agent communication, HMM settlement, and Lux infrastructure.

### AI Mining Protocol - HIP-006
GPU-based AI compute mining with NVTrust hardware attestation. Each job pre-commits to a target chain (Hanzo/Zoo/Lux) before compute runs, preventing double-spend across chains. Rewards are native L1 tokens, teleportable to any EVM chain via Lux Teleport bridge.

## Relationship to Lux Proposals (LPs) and Zoo Proposals (ZIPs)

### Hanzo <-> Lux
- Hanzo operates as a sovereign L1 on Lux Network (HIP-0024)
- Hanzo KMS (HIP-005) extends Lux KMS (LP-325)
- ZAP (HIP-007) adopted by Lux as LP-120
- Hanzo-Lux Bridge Protocol (HIP-0101)
- Shared post-quantum cryptography standards
- AI Mining rewards teleportable to Lux C-Chain (96369)

### Hanzo <-> Zoo
- ASO (HIP-002) extended by Zoo's DSO (ZIP-001) for decentralized prior aggregation
- HMM (HIP-004) uses Zoo's PoAI (ZIP-002) for quality verification and attestation
- Zoo AI Mining Integration (ZIP-005)
- Shared HLLM architecture and model portability
- Zoo EVM (200200) is a mining reward destination

### Key Differences
- **HIPs**: Hanzo-specific standards for AI infrastructure, services, and L1 chain
- **LPs**: Lux blockchain standards for consensus, networking, cross-chain protocols
- **ZIPs**: Zoo Foundation standards for decentralized AI research, training, and governance

## HANZO Tokenomics (1B total supply)

| Allocation | Percentage |
|-----------|-----------|
| Training Rewards | 30% |
| Compute Providers | 20% |
| Model Developers | 15% |
| Community Treasury | 15% |
| Team (4yr vest) | 10% |
| Public Sale | 5% |
| Liquidity | 5% |

Token utility: 0.01 HANZO per interaction, 0.001-0.1 per 1K tokens, 10+ to mint Model NFT, veHANZO governance.

## Creating a New HIP

1. Copy `docs/templates/hip-template.md` to `HIPs/hip-X.md`
2. Submit PR to github.com/hanzoai/hips
3. HIP editors review for completeness
4. Community discussion and feedback
5. Last Call (14-day review period)
6. Final status if accepted

## Validation

```bash
make validate-hip HIP=hip-X   # Format check
make check-links               # Reference check
act -j validate                # Local CI
```

---

Last Updated: 2026-03-13
