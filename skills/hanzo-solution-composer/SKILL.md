---
name: hanzo-solution-composer
description: Design a real customer solution from Hanzo OSS building blocks; gather requirements, select minimal components, verify compatibility and licence, benchmark AI, then deliver a deployment blueprint and evidence.
---

# Hanzo Solution Composer

Use this skill when a customer asks Hanzo to design an AI business solution, sovereign deployment, vertical operating system, private AI stack, or multi-component implementation.

## Principle

Treat Hanzo components as replaceable, versioned building blocks, not as a mandatory suite. Compose the smallest architecture that solves a measured business problem. Never promise that a repo being public means its code, model weights, licence, deployment or integration is approved for customer use.

Read these progressive skills as needed:
- hanzo-oss-legos: the component selection matrix and upstream OSS inventory
- hanzo-decision-evaluation: the decision-model evaluation protocol
- hanzo-eu-deployment: customer-owned EU tenant, privacy and deployment gates
- hanzo-sports-growthos: worked example for an AI-native sports commercial operating system
- hanzo-cloud-architecture and hanzo-ml-research: implementation and native research constraints

## Intake: write the evidence register first

Require a named customer decision-maker, workload owner, and technical owner. Create a structured intake with:

| Area | Questions | Evidence |
|---|---|---|
| Business outcome | Which revenue, cost, risk or time metric changes? | Baseline, numerator/denominator, target, owner |
| Actors | Who approves decisions? What actions are automated? | Roles, approval matrix, workflow |
| Data | Sources, contracts, residency, identifiers, consent, retention | Supplier inventory, controller/processor map |
| Predictions | Churn, propensity, risk, collections, pricing: what is the label? | Label definition, coverage, leakage risks |
| Environment | Customer vs Hanzo-owned project; EU/US; cloud/on-prem; recovery | Architecture constraints and billing owner |
| Interfaces | Existing APIs, exports, auth, events, required latency | Interface contract and integration confidence |
| Economics | Compute, development, licence, operations, margin ceiling | Cost envelope and assumptions |
| Exit | Portability, IP, escrow, backups, migration, termination | Explicit contract and OSS licence matrix |

Classify facts as verified, customer-stated, assumed, blocked or to-be-tested. Never silently convert a statement like “club connected” into verified credentials or data rights.

## Composition workflow

1. Identify the minimum happy-path workflow and one negative / exception path.
2. Define data boundaries, tenant boundaries and system-of-record ownership before selecting models.
3. Select the fewest Lego components needed; use hanzo-oss-legos and the live /oss/index.json catalog.
4. Separate deterministic rules, statistical baselines, Kai typed decisions, and optional generative text tasks.
5. Record interface contracts: caller, endpoint/protocol, schema, auth, state owner, data classification, retry and rollback.
6. Tag every component as production-verified, integrated but unverified, candidate, custom build or future research. Verify claims against a current repository revision, runtime test and licence.
7. Estimate total lifecycle cost including hardware amortisation, egress, storage, model retraining, support and human review. No vendor-specific model or hardware promises without measurement.
8. Define reproducible acceptance tests and a holdout/control design before work begins.
9. Create G0 demo, G1 staging and G2 production gates with explicit exclusions; do not let a proof-of-concept imply a production SLA.
10. Obtain written approval before making cloud changes, incurring spend, loading personal data, changing prices or contacting customers.

## Required output pack

Produce these artifacts, in this order:

1. one-page problem statement with business metric and non-goals;
2. evidence register and data-access/consent inventory;
3. capability map linking outcomes to selected repos and interfaces;
4. diagram of trust boundaries, data flows, deployment and model/agent paths;
5. model choice and evaluation matrix: rules vs classical ML vs Kai vs optional generative model;
6. bill of materials with source links, exact versions, licence and compatibility checks, operational owners and cost;
7. security/privacy controls and data processing readiness;
8. week-by-week implementation plan with tests, dependencies, budget and rollback;
9. customer demo script with synthetic data and measured acceptance criteria;
10. open issues, owners, assumptions and signatures for the next phase.

## Hard stops

- Synthetic data by default in discovery. No personal data without verified authority, role mapping, access controls and signed agreements where required.
- Never mix customers' raw data or weights without explicit authorisation and legal/technical validation.
- Do not present federated updates as anonymous merely because raw records are absent; test reconstruction/inference risk.
- Do not embed always-on autonomous bot services by default. Favor event-driven, short-lived executions with durable state.
- Do not prescribe an LLM for structured decisioning or a single model for all targets without a fair benchmark.
- Do not claim a bespoke trained model or perpetual deployment rights from a 30-day proof-of-concept unless independently scoped and contracted.

## Handoff question

Can an engineer who never attended the sales meeting reconstruct the environment, costs, evaluation, security obligations, acceptance tests and component interfaces from this pack? If not, the design is incomplete.
