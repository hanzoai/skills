---
name: hanzo-sports-growthos
description: Worked Hanzo solution for FanGrowth-style sports revenue operations: isolated club data, Kai-first decisioning, renewal/family/membership journeys, experiments and finance reconciliation.
---

# Hanzo Sports GrowthOS — FanGrowth Reference Pattern

A worked solution for a sports-club revenue operating system. This is an illustrative architecture, not evidence of contracted Lecce rights, live FanGrowth connectors, measured uplifts, or a production build.

## Customer problem

One club has ticketing, memberships, merchandise, app engagement, marketing consent and financial records in separate systems. The project aims to identify legitimate commercial opportunities and measure actual incremental net revenue, while the club retains control of personal data and all fan-facing decisions.

Initial journeys:

1. Renewal Radar — identify eligible members at risk of non-renewal, offer human-reviewed engagement.
2. First-Match Family — help first-time family attendees consider another match or membership where consent permits.
3. Membership Upsell — propose suitable upgrades with eligibility, inventory and commercial limits.

Never promise uplift or assume a 400k fan dataset exists. A stated 7–14% growth target is a testable hypothesis, not a validated outcome.

## Five-layer design

| Layer | Minimum implementation | Optional / future |
|---|---|---|
| 1. Club data vault | Club-owned isolated records, fan identity, provenance and consent; typed adapter specs | Live source connectors after approval |
| 2. Club decisions | Rules, baseline models, and Kai typed decisions; define outcome labels | Club-specific model adaptation after evaluation and rights |
| 3. Operator interface | CRM screens, approved human tasks and output ledger | Zen language/creative, Enso orchestration, social/ads if needed |
| 4. Federation/network | None required for G0; keep cross-club boundary closed | Secure, evaluated opt-in model updates and aggregated playbooks |
| 5. Experiments + finance | Randomised holdouts, purchase/renewal outcomes and auditable reconciliation | Bandit allocation only when traffic and safeguards justify it |

The language layer is optional. Never hardcode Mistral or another generative vendor. When text is required, choose and benchmark an appropriately licensed model against cost, privacy and user need. Kai is not generative; do not use it to draft copy.

## Reusable Hanzo Lego selection

- Core proof: hanzoai/console for CRM/operator shell; hanzoai/insights for measurement; hanzoai/kai or actual deployed Kai service for typed decisions; hanzoai/iam for access; hanzoai/kms for secrets.
- Optional if justified: hanzoai/post and hanzoai/ads for activation; hanzoai/ai or hanzoai/llm for routing; hanzoai/zen for text; hanzoai/commerce for orders and finance; hanzoai/s3 and hanzoai/vfs for storage; hanzoai/o11y for telemetry.
- Do not assume these packages work together without current interface tests, exact revisions and licence checks. See hanzo-oss-legos.
- Agents are event-driven and human-approved, not always-on processes. State and decision history persist between invocations.

## End-to-end demonstration contract

For each journey provide:
- synthetic input profiles with consent, identity and provenance;
- deterministic eligibility and policy constraints;
- typed prediction/recommendation and rationale code;
- human review/approve/reject step;
- simulated activation (never contact a real fan in G0);
- outcome and revenue ledger event;
- trace, rollback/undo, negative test and audit evidence.

Success is objective: all three journeys execute in the agreed EU test project, all agreed positive/negative tests pass, a completion pack is delivered, and a documented Meridian/club stakeholder demo is performed. Stakeholder approval, investment or a follow-on sale are not software acceptance criteria. Customer-provided access and scheduling dependencies must be recorded.

## Four-week G0 plan

- Days 1–5: confirm commercial scope, owners, acceptance matrix, synthetic data and source inventory.
- Days 6–10: private EU architecture, hardware assessment, identity/consent schema, connector specifications.
- Days 11–20: build the three Kai-first synthetic journeys using only needed UI and activation modules.
- Days 21–25: test correctness, controls, audit, baseline metrics, cloud spend and negative paths.
- Days 26–30: demo, deliver architecture and release manifest, estimate cost, propose G1 integration/staging plan and G2 controlled rollout.

Define build cost, model custody, perpetual licence, source escrow and exclusivity separately in later contracts; a G0 demo is not a transfer of Hanzo platform IP. Customer-specific code and deliverable ownership follows the signed SOW, not an architecture sketch.

## Research gates for the next phase

Before deciding a single adaptable model can handle churn, propensity, pricing and collections, compare Kai, deterministic rules, logistic regression and tree-based baselines on leakage-controlled, time-split data. Distinguish ranking performance from actual uplift; for pricing use a lawful and safely constrained design. Federated learning is optional future research requiring secure aggregation, differential privacy and privacy-attack evaluation, not an assumed network effect.

## Follow-on skills

hanzo-solution-composer, hanzo-decision-evaluation, hanzo-eu-deployment, hanzo-oss-legos, hanzo-ml-research, hanzo-insights.
