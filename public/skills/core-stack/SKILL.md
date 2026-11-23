# Hanzo ML Research: Core Stack Reference

This reference accompanies the `hanzo-ml-research` skill.

## Canonical repositories

### hanzoai/ml

Rust-native ML compute stack.

Important crates include:

- `hanzo-ml`
- `hanzo-nn`
- `hanzo-train`
- `hanzo-transformers`
- `hanzo-kernel`
- `hanzo-research`

The `hanzo-research` crate is the canonical Rust client for the unified cloud research plane and automatically records source/runtime provenance.

### hanzoai/engine

Native Rust inference/serving engine.

Use it when research touches:

- serving
- batching
- quantized inference
- model loading
- attention kernels
- KV cache
- speculative decoding
- backend-specific runtime behavior
- end-to-end latency

### hanzoai/decision

Kai typed-decision research and production stack.

Use it when research touches:

- typed decisions
- decision programs
- calibration
- safety heads
- option retrieval/reranking
- joint decoding
- benchmarking
- training objectives
- StateHandle / cached decision execution

### Hanzo DSL / kernel stack

Use the local Hanzo DSL/kernel implementation exposed by the active workspace. Prefer extending existing Rust DSL/kernel primitives to introducing a second implementation path.

## Research API

The native Rust SDK in `hanzo-research` uses:

- `POST /v1/research/experiments`
- `GET /v1/research/experiments`
- `GET /v1/research/totals`
- `POST /v1/research/artifacts`

Configuration:

- `HANZO_API_KEY`
- `RESEARCH_BASE`
- `RESEARCH_PROJECT`

The server keys canonical experiment identity by the research record's project/stable experiment identity, while provenance captures git/repository/runtime details.

## Typical research object

```text
hypothesis
prediction
control
treatment
primary metric
non-inferiority metrics
train/dev/final lineage
artifact hashes
raw observations
verdict
because
```

Keep the experiment falsifiable and durable.

## Rust-first rule

Python/notebooks are acceptable for disposable visualization or inspection.

They are not canonical for:

- final model training
- benchmark generation
- feature extraction
- serving latency
- kernel timing
- final threshold selection
- production model behavior

unless the experiment explicitly studies a Python/PyTorch implementation.

Canonical results should be reproduced in Hanzo's Rust stack.
