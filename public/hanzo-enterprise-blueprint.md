---
name: hanzo-enterprise-blueprint
description: Reusable, industry-neutral Hanzo enterprise reference architecture for decision-driven workflows, isolated data, evaluation, human-approved actions and measurable outcomes; compose minimal OSS building blocks.
---

# Hanzo Enterprise Blueprint — Decision-Driven Workflows

Use this reference for any business process where fragmented inputs must become structured decisions, permitted actions and auditable outcomes. Examples may include operations, risk, customer service, procurement, finance, commerce, logistics and internal workflow automation. It is deliberately not a customer-specific solution.

## Start from a measurable decision

Before selecting software, identify the actor, trigger, inputs, allowed actions, failure consequences, measurement horizon, acceptance tests and data rights. Write explicit hypotheses and define baseline performance. A stated target is not an observed result.

The purpose is to build only what the problem requires; no complete Hanzo suite, generative AI model or distributed training system is mandatory.

## Reference architecture: five adaptable capabilities

| Capability | Minimum useful implementation | Optional / gated extension |
|---|---|---|
| 1. Isolated data | Authoritative source data, typed schemas, provenance, permissions, retention and tenant isolation | Live connectors when access is approved |
| 2. Structured decisions | Deterministic rules and tested baselines, with Kai as candidate for typed recommendations | Task-specific training/adaptation after benchmark and rights review |
| 3. User/agent interface | Human review of proposed actions, audit capture and workflow UI | Enso tool/model orchestration, Zen generation, external execution |
| 4. Cross-tenant learning | Off by default; no shared raw data or weights | Explicitly authorised, privacy-tested federation/aggregation |
| 5. Outcome measurement | Baselines, outcome ledger, experiment/holdout and business KPI evidence | Adaptive allocation after adequate statistical evidence |

All five are *capabilities*, not five services to install. A simple implementation may use a single binary and no model at all. Kafka, Kubernetes, vector search, a language model and federated learning are not prerequisites.

## Example: generic human-approved decision workflow

1. A synthetic event enters the workflow with a tenant ID, provenance, permissions and typed attributes.
2. An eligibility policy decides whether any action is permitted.
3. A baseline or Kai evaluates the case and returns a structured result; missing/low-confidence cases require human review.
4. An operator reviews, approves or rejects the proposed action.
5. A sandbox adapter records simulated execution; real external side effects are disabled in discovery.
6. An outcome ledger connects the decision to an observed result, including cost, latency and error handling.
7. A replay and negative test demonstrate audit, safety and rollback behavior.

This sequence works for approval routing, anomaly triage, invoice exceptions, retention decisions, supply planning or other structured processes. Adapt metric and action type to the actual customer rather than shipping a canned vertical workflow.

## Select from Hanzo OSS Legos

Begin with the minimum set needed for the problem:
- Identity/security: hanzoai/iam, hanzoai/kms
- Application/data: hanzoai/base, hanzoai/console or hanzoai/ui; optional hanzoai/s3 for object storage
- Decisioning/research: Kai, hanzoai/ml and hanzoai/engine as appropriate; compare with deterministic and standard ML baselines
- Orchestration/agents: optional Enso/gateway components, hanzoai/flow, hanzoai/mcp, event-driven hanzoai/bot
- Business connectors: optional Insights, Commerce, CMS, Post, Ads according to user-approved task
- Deployment/observability: Hanzo cloud, universe, visor, o11y/telemetry if the topology needs them

Read hanzo-oss-legos for the full source catalogue. Candidate repositories must be checked for exact version, licence, operational readiness and data residency. Public visibility of source does not establish rights to model weights or customer data.

## Delivery phases: scope to the contract

| Phase | Evidence | Exit condition |
|---|---|---|
| Discovery | Problem, source inventories, business KPI, data boundary, cost limit, component selection | Written owners and agreed acceptance criteria |
| Proof of concept | Synthetic data, sandbox decisions, negative tests, operator approval and outcome ledger | Verified demo and evidence pack |
| Staging | Contracted connectors, security review, DPAs where necessary, repeatable deployment, shadow test | Tested rollback, access and budget controls |
| Production | Explicit authorisation, monitoring, backups/restore, SLO, cost and quality gates | Owned on-call and release/rollback plan |
| Optimisation | Drift detection, controlled experiments, cost/latency and commercial impact | Measurable improvement over baseline |

Do not convert a proof of concept into a perpetual licence, source escrow, exclusive deployment right or production SLA without an explicit agreement. Ownership and support obligations come from signed terms.

## Required outputs

- Problem and value hypothesis, evidence/assumptions register
- Architecture and data-flow diagram
- Source and component BOM, interface schema, licence matrix and cost estimates
- Evaluation matrix: policies, conventional baselines, Kai and optional generative alternatives
- Sandbox demo and test report including failure modes
- Security/privacy, residency, operations, rollback and portability
- Next-phase plan with owners, dependencies, acceptance criteria and budget

## Related skills

hanzo-solution-composer; hanzo-oss-legos; hanzo-decision-evaluation; hanzo-eu-deployment; hanzo-ml-research; hanzo-cloud-architecture.
