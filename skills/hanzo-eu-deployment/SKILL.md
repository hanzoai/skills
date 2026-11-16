---
name: hanzo-eu-deployment
description: Plan and verify customer-owned sovereign Hanzo deployments in EU cloud or private infrastructure; consent, data residency, IAM, KMS, GCP budgets, testing, rollback and handoff.
---

# Hanzo EU Private Deployment — Customer-Owned Cloud and Club Nodes

Use for a private single-tenant Hanzo solution, especially EU football clubs, GDPR workloads, and customer-owned Google Cloud projects. This is an architecture and readiness playbook, not evidence that a specific deployment is certified or legally compliant.

## Ownership and topology

- Customer names the Google Cloud project, legal owner, billing owner, administrators and budget owner. Hanzo is granted the minimum approved scoped access.
- Region: choose a specific EU region for data, backups, logs, inference and other storage; inspect actual service residency guarantees instead of relying on a project's region label.
- Isolation: one tenant/project boundary per club unless a documented shared-service design passes security review; no implicit club-to-club raw data sharing.
- Prefer private connectivity, workload identity, managed secret controls, encryption at rest/in transit, immutable audit exports and explicit egress policy.
- For G0 use synthetic fan records and sandbox/mock connectors, isolated from production.
- Hanzo IAM, Hanzo KMS, Base/S3, VFS, inference, Insights and workflows are options, not a fixed bundle. Select only needed components using hanzo-oss-legos.

## Discovery checklist

| Gate | Evidence required |
|---|---|
| Contract | Named customer entity, paid SOW, data rights, approved owners |
| Access | Project ID, approver, IAM scope, expiry and access log |
| Budget | Written ceiling, alerts, automatic stop/escalation and payer |
| Data | Data map and classified sources, consent and retention |
| Legal | Controller/processor allocation, Art. 28 DPA before personal data processing, transfer review |
| Network | VPC, DNS, routing, firewall and permitted egress |
| Storage | Backups, restoration, deletion, encryption and portability |
| Workload | Runtime deployment shape, CPU/GPU sizing tests, concurrency and idle cost |
| Security | SSO/MFA, least privilege, secrets, patching, monitoring, incident plan |
| Exit | Source and model custody boundaries, export, rollback, removal of access |

A customer-owned project and an EU region do not themselves satisfy GDPR. Assess remote access, subprocessors, monitoring/log exports, data subjects' rights, legal basis and incident obligations. Pseudonymised data may still be personal data.

## G0: 30-day safe implementation

1. Week 1 — Confirm owners, budget, acceptance tests and entirely synthetic dataset; inspect source-system inventory without live credentials.
2. Week 2 — Assess Netherlands/EU architecture and hardware; design isolated project/tenant, IAM and data model; test minimal deployment route.
3. Week 3 — Exercise three end-to-end synthetic workflows and capture evidence. Human approval gates all external actions and price changes.
4. Week 4 — Demonstrate repeatable configuration, failure handling, audit trace, budget usage, exit/export; deliver production plan and risk register.

Use the customer billing account and keep within the agreed cap; any higher spend needs written approval. G0 excludes live fan integrations, production SLA, blanket escrow, bespoke trained weights and perpetual production rights unless explicitly contracted.

## G1 / G2 acceptance gates

- G1 private staging: DPA and subprocessor mapping if personal data, signed system access agreements, connector test fixtures, least privilege, disaster recovery and shadow evaluations.
- G2 controlled production: recorded human authorisations, encrypted backups with restore proof, workload performance, audit completeness, consent enforcement, rollback, monitoring and operational owner.
- Never infer available GPUs, service quotas, model runtime or on-prem compatibility without executing hardware/infrastructure benchmarks.

## Minimal deployment runbook

Publish the environment manifest and interface/BOM from hanzo-solution-composer. Record exact git revisions, image digests, region, environment variables by key name (not secret value), cost controls, test reports and runbooks. Deployment commands must be selected from the actual checked-out repository and pinned version, then reviewed before execution. Never paste live secrets or invent unverified CLI commands.

## Related skills

hanzo-solution-composer, hanzo-oss-legos, hanzo-cloud-architecture, hanzo-iam, hanzo-kms, hanzo-s3, hanzo-ml-research, hanzo-sports-growthos.
