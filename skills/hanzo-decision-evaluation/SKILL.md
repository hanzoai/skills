---
name: hanzo-decision-evaluation
description: Research and benchmark revenue, churn, pricing, propensity and collections decisions using rules, classical ML and Hanzo Kai; separate predictive quality from causal uplift and cost before deployment.
---

# Hanzo Decision Evaluation — Evidence Before Architecture

Use when choosing a decision model for structured commercial, risk, collection or operational outcomes.

## Define the decision precisely

Create a decision card: action to choose, who approves it, eligible population, label and measurement horizon, cost of mistakes, available features, latency limit, privacy constraints and audit fields.

For club use cases:
- renewal/churn: definition of a renewal, timing, leakage from future activity;
- propensity: specific next purchase, period, channel and inventory eligibility;
- pricing: demand elasticity and policy constraints; prediction alone does not establish safe or optimal price changes;
- collections: invoice-to-payment forecast, ageing, partial payments, exception classification and accounting reconciliation.

## Benchmark ladder

1. Trivial baseline and deterministic policies.
2. Logistic regression and calibrated tree ensemble (XGBoost/LightGBM as appropriate).
3. Kai typed decision program using the same information available at inference time.
4. Optional text generator only for natural-language drafting or explanation; not required to score customers.

A single adaptable Kai model serving many programs is a hypothesis to test, not a requirement or established result. Evaluate per-program quality and cross-program drift; adapt/fine-tune on club data only after securing rights, enough representative examples, model provenance and a reproducible holdout.

## Evaluation protocol

| Property | Minimum check |
|---|---|
| Data validity | Consent, schema completeness, source attribution, leakage |
| Generalisation | Time-split and club-held-out tests where feasible |
| Quality | AUC/PR-AUC, recall, calibration, error cost as applicable |
| Pricing | Off-policy and constrained evaluation; do not infer uplift from model scores alone |
| Causal impact | Randomised holdout, conversion and incremental net contribution |
| Reliability | Drift, missing fields, reversals, feature movement, reproducible results |
| Runtime | p50/p95 latency, tokens/compute, CPU/GPU use, sustained concurrency |
| Governance | Version, reason/decision code, human approval, appeal and override |

Do not claim a 7–14% uplift target as measured performance. Distinguish model prediction metrics from commercial uplift. Register hypotheses and falsifications using hanzo-ml-research and the /v1/research evidence plane.

## Typed decision contract

Document input schema, output schema, permitted next actions and the fail-closed path. Suggested conceptual shape:

~~~json
{
  "program": "renewal_radar",
  "evidence": [{"source": "synthetic_crm", "type": "renewal_history"}],
  "result": {"decision": "ask", "score": 0.0, "action_code": "human_review"},
  "audit": {"model_revision": "pinned", "policy_revision": "pinned"}
}
~~~

This is an illustrative schema, not a claim of a production API response. Validate against the actual Kai decision API and pinned revision before coding.

## Release gate

Publish a model card, evaluation report, data rights, licence/custody decision, cost model, failure/rollback plan and the owner who may approve production. Never commit to weights, source escrow, production guarantees or fixed accuracy without a separate scoped agreement.
