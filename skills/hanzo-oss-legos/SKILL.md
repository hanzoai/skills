---
name: hanzo-oss-legos
description: Choose and assemble modular Hanzo open-source repositories like Legos; map capabilities, interface and ownership boundaries, repository evidence, licences and compatibility before proposing a customer stack.
---

# Hanzo OSS Legos — Composable Component Map

Use for choosing Hanzo software building blocks in a customer architecture. The canonical existing index is https://hanzoskills.com/oss/index.json; the companion architecture manifest is hanzo-oss. The map below is a selection guide, not a blanket assertion of licence, release readiness or compatibility.

## Lego selection matrix

| Capability | Candidate Hanzo repositories | Interface / purpose | Use only if |
|---|---|---|---|
| Cloud control plane | hanzoai/cloud, zap-proto/zip, zap-proto/zap | /v1 API and typed service composition | Single-binary/plugin topology fits deployment |
| Identity / access | hanzoai/iam | OIDC/OAuth2, organisations, roles | Customer identity, SSO and permissions agreed |
| Secrets | hanzoai/kms | Tenant secret isolation and rotation | No plaintext secret leakage |
| Data and events | hanzoai/base | Application records, sync and SSE | Fit to relational/event and isolation needs |
| Object storage | hanzoai/s3 | S3-compatible assets, backup | Residency and lifecycle/backup documented |
| Workspace disk | hanzoai/vfs | VFS disk on S3 | VFS compatibility and durability tested |
| Platform orchestration | hanzoai/universe, hanzoai/visor | GitOps topology and workload placement | Maturity, policy and target cluster verified |
| ML training/research | hanzoai/ml, hanzoai/kai | Native model/training/decision tools | Model licence and actual runtime verified |
| AI inference | hanzoai/engine | Model serving | Hardware model support and latency measured |
| API/model routing | hanzoai/llm, hanzoai/ai | Gateway/control plane | Required providers, privacy and price verified |
| Generative models | hanzoai/zen | Optional text generation / reasoning | Only if task benefits; check weights separately |
| CRM shell | hanzoai/console | Audience and commercial workflow UI | Feature availability and fork licence checked |
| Analytics | hanzoai/insights | Events, funnels, experiments | Tenant-level data consent respected |
| Ads | hanzoai/ads | Marketing activation | Ad account rights and consent verified |
| Social | hanzoai/social, hanzoai/post | Scheduling, publication, connectors | Network auth and approval flows verified |
| Content | hanzoai/cms | Content pipeline | Publishing authority exists |
| Commerce | hanzoai/commerce | Billing, purchase and ledger integration | Finance system of record and reconciliation defined |
| Search / vectors | hanzoai/vector | Retrieval / embeddings | Required at all; not for basic structured decisions |
| Telemetry | hanzoai/o11y, hanzoai/telemetry | Traces, audit and metrics | Sensitive-data redaction and retention configured |
| Event workflows | hanzoai/flow | Workflow coordination | Event-driven lifecycle with durable state |
| Agent execution | hanzoai/bot, hanzoai/mcp | Agent tools / sandbox access | No autonomous outreach without approvals |
| CLI and engineering | hanzoai/cli, hanzoai/dev, hanzoai/skills | Build, deploy and agent skills | Operator consent and credential boundary known |
| UX/design | hanzoai/ui, hanzoai/team | Composable workspace front end | Tenant branding and accessibility confirmed |

All repo names here are candidate source references, not assertions of shipped integrations. Hanzo's private repositories and proprietary model weights are not automatically OSS. Public repository visibility does not establish an Apache-2.0/MIT licence; read LICENSE, NOTICE, dependencies and model cards at the pinned version.

## Lego contract (complete for every selected component)

- Repo URL, exact SHA/tag, maintainers, deployed version and build provenance
- Source licence, vendored dependencies, model/weight licence, modifications and obligations
- Exposed interface and typed schema, auth scopes, secrets path, tenancy, logging
- Input and output data classifications, storage region and retention
- Resource requirements, pricing/compute ceiling and measured performance
- Failure behavior, restartability, idempotency, observability, backups and rollback
- Evidence: actually exercised integration test vs an architectural assumption
- Replaceability: data export, migration, interface contract and support owner

## Composition rules

- Never deploy every Lego just because it exists. Start with data + identity + one decision path + observation + a real outcome.
- Tenant/customer isolation is a hard boundary. Cross-tenant aggregation, training and reuse require distinct agreements and privacy engineering.
- Kai returns typed decisions; it is not a replacement for a full text generator. Rules and tree-based baselines may be sufficient or superior for some targets.
- Enso/orchestration and Zen/text generation are optional. Avoid self-hosting expensive generative models for a task that needs only structured output.
- Build customer-specific adapters against typed interfaces so components can be swapped without replacing all business logic.
- Prefer one storage architecture and one source of truth; avoid introducing duplicate databases, gateways, IAM providers or independent disk designs.
- Never choose a vendor model by brand name alone or represent an unsized hardware estimate as a fixed requirement.

## Progressive related skills

See hanzo-solution-composer, hanzo-decision-evaluation, hanzo-eu-deployment, hanzo-enterprise-blueprint, hanzo-cloud-architecture, hanzo-ml-research, hanzo-iam, hanzo-kms, hanzo-s3, hanzo-insights, hanzo-post.
