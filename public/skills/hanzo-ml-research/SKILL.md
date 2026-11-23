---
name: hanzo-ml-research
description: Native Hanzo AI/ML research and systems-design skill for LLMs, encoders, decision models, training, inference, kernels, quantization, retrieval/reranking, safety, evaluation, and scientific benchmarking. Use Hanzo's Rust stack (hanzo-ml, hanzo-nn, hanzo-train, hanzo-transformers, hanzo-kernel/DSL, hanzo-research, engine, and decision/Kai) by default; search current literature when needed; record falsifiable experiments and artifacts through the unified /v1/research plane.
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["cargo", "git"] }
      }
  }
---

# Hanzo ML Research

Use this skill for machine learning, LLM, encoder, decision-model, retrieval, training, inference, kernel, quantization, safety, evaluation, or AI architecture work in the Hanzo stack.

The default operating principle is:

> Research in the same stack that ships.

Exploratory scripts may be used for disposable analysis, but canonical results, model code, training, evaluation, latency measurements, and production paths should be implemented and reproduced in native Rust on Hanzo's ML/runtime stack unless the task explicitly requires another environment.

## Core stack

Prefer the repository/workspace already associated with the task. The important Hanzo components are:

- `hanzoai/ml`
  - `hanzo-ml`: tensor/autograd/device layer
  - `hanzo-nn`: neural-network modules
  - `hanzo-train`: optimizers and training utilities
  - `hanzo-transformers`: transformer/model building blocks
  - `hanzo-kernel` / Hanzo kernel DSL: generated and hand-optimized kernels
  - `hanzo-research`: native research/evidence SDK
- `hanzoai/engine`: native Rust model inference, serving, kernels, batching, quantization, and backend integration
- `hanzoai/decision`: Kai typed-decision models, training, evaluation, decision programs, safety heads, retrieval/reranking, and benchmark harness
- `hanzoai/python-sdk/pkg/hanzo-research`: parity client for the research plane; not the canonical implementation for Rust-native experiments

When a local workspace exposes additional Hanzo DSL, CUDA, ROCm, Metal, Vulkan, distributed-training, or model crates, inspect and reuse them rather than creating parallel implementations.

## Research plane: /v1/research

Canonical R&D evidence should be recorded through `hanzo-research`, which writes to the unified research plane under `/v1/research`.

Rust:

```rust
use hanzo_research::{Research, Verdict};

let r = Research::from_env();

let mut exp = r
    .experiment("pooling", "j6-logmeanexp", "override-detection")
    .metric("recall_at_99_5_specificity")
    .hypothesis("localized pooling recovers procedural override evidence hidden by mean pooling")
    .predict("LogMeanExp improves held-out override recall without increasing benign FP beyond the preregistered gate");

exp.log("frozen encoder; identical split, seed, optimizer, and token cache across arms");
exp.conclude(
    Verdict::Proven,
    "LogMeanExp cleared the preregistered gate",
    Some(0.87),
)?;
```

Environment:

- `HANZO_API_KEY`
- `RESEARCH_BASE` (defaults to `https://api.hanzo.ai`)
- `RESEARCH_PROJECT`

Important native endpoints include:

- `POST /v1/research/experiments`
- `GET /v1/research/experiments`
- `GET /v1/research/totals`
- `POST /v1/research/artifacts`

Use `hanzo-research` APIs rather than hand-rolling HTTP when working in Rust. The SDK automatically captures repository, git, host, and resolved Hanzo crate provenance.

A refutation is a valid research result. Do not rewrite failed hypotheses into successes after the run.

## Research workflow

For substantial research work, follow this order.

### 1. Establish the question

Write a falsifiable hypothesis before looking at the decisive result.

Bad:

> Try some pooling methods and see what works.

Good:

> Frozen P0 token states already encode procedural-override evidence, but global mean pooling dilutes short localized cues. A learned linear token scorer with normalized LogSumExp pooling should improve family-held-out override recall at fixed benign specificity.

Specify:

- primary metric
- control
- treatment
- non-inferiority constraints
- stopping point
- untouched final set
- expected mechanism

### 2. Inspect the actual Hanzo implementation

Before proposing a replacement:

- read the relevant Rust crates
- identify the current execution path
- check whether the desired capability already exists
- inspect tensor shapes, masks, cache layout, quantization, and backend dispatch
- identify CPU/GPU synchronization or host round trips
- confirm whether an apparent model limitation is actually training, data, retrieval, calibration, or runtime behavior

Do not recommend rebuilding components that already exist.

### 3. Search current research

For state-of-the-art or rapidly changing ML topics, search current primary sources before making SOTA claims.

Prefer:

1. peer-reviewed paper or official arXiv manuscript
2. authors' official repository
3. model/system technical report
4. official benchmark repository
5. independent reproduction

Use external research to generate hypotheses and baselines. Reimplement important mechanisms in the Hanzo stack before treating them as production candidates.

Good areas to scan when relevant:

- efficient encoders and long-context bidirectional models
- extreme classification and million-label retrieval
- dense retrieval and late interaction
- distillation and calibration
- preference/ranking objectives
- multi-objective gradient methods
- quantization: FP8/NVFP4/INT8/INT4/ternary
- FlashAttention and unpadded attention
- speculative decoding
- KV-cache compression/quantization
- distributed training and optimizer sharding
- MoE routing
- structured prediction and set-equivariant models
- selective prediction / abstention / conformal risk control
- safety classifiers and localized token readouts
- GPU/ROCm/Metal/Vulkan kernel design

### 4. Build the smallest controlled ablation

Change one causal factor at a time.

Examples:

- pooling: mean vs max vs top-k vs LogMeanExp
- targets: hard vs entropy-matched vs relational
- retrieval: semantic vs +memory vs +late-interaction
- optimizer policy: static vs FAMO/CAGrad/D2
- quantization: BF16 reference vs FP8/NVFP4 candidate
- kernel: reference vs generated DSL vs hand kernel

Keep identical where possible:

- initialization
- seed
- data order
- parent checkpoint
- optimizer
- LR schedule
- evaluation engine
- batching
- numerical policy
- cache
- train/dev/final split

If one arm uses extra capacity, report it as a capacity ablation, not a clean mechanism ablation.

### 5. Assert implementation invariants

Before benchmarking, add tests that make silent confounds fail loudly.

Typical invariants:

- expected tensor shapes
- masks exclude padding
- gradients reach intended trainable parameters
- frozen parameters do not change
- same initialization checksum across arms
- same input/cache checksum across arms
- ragged and padded/segmented implementations agree
- no unintended host scalar extraction in hot paths
- feature-off behavior matches the pre-feature path
- non-target task outputs remain identical when a specialized head is enabled
- checkpoint, optimizer, scheduler, RNG, and data cursor restoration is exact when testing continuation

### 6. Measure the right decomposition

Do not collapse a pipeline failure into one metric.

For retrieval/reranking, separate:

- Recall@K: did the correct option survive retrieval?
- conditional decision accuracy: did the scorer select it given survival?
- final end-to-end accuracy
- latency by stage

For classification/safety, separate:

- recall
- specificity
- false positives
- AUPRC
- calibration
- family-held-out performance
- subgroup/tail behavior

For model adaptation, separate:

- train capacity
- validation accuracy
- NLL
- Brier
- ECE
- tail margins
- repair/damage turnover
- gradient geometry

For latency, separate:

- encode time
- decision-only time
- retrieval
- rerank
- verification
- host synchronization
- cold vs warm
- p50/p95/p99

### 7. Use parent-relative controls when preserving competence

When adding a protection or adaptation mechanism, compare against both:

- frozen parent
- matched continuation without the mechanism

This distinguishes retention from causal prevention of degradation.

When continuation matters, prefer full warm resume:

- model weights
- optimizer first moments
- optimizer second moments
- optimizer step
- scheduler
- RNG
- data cursor

### 8. Seal the result

Once the preregistered endpoint is reached:

- write the canonical report
- save checkpoint/artifact hashes
- log the experiment to `/v1/research`
- record verdict: `proven`, `refuted`, or `inconclusive`
- preserve raw predictions when feasible
- do not retune on the untouched final set

## Rust-native implementation rules

### Tensor graph first

Training-time operations return tensors. Do not call `to_scalar`, `to_vec`, or equivalent host extraction before the loss/backward boundary.

Bad:

```rust
let score: f32 = pooled.to_scalar()?;
```

inside a trainable forward path.

Good:

```rust
let pooled: Tensor = pool.forward(&scores, &mask)?;
let loss: Tensor = objective.forward(&pooled, &labels)?;
loss.backward()?;
```

Convert to host scalars only for logging/evaluation.

### GPU hot path

Avoid:

- CPU sorting inside a CUDA inference/training path
- repeated device transfers
- per-token host loops when a segmented/batched reduction is available
- hidden synchronization caused by scalar reads

Reference implementations may be simple/ragged first. Optimize only after correctness is established.

### Numerical policy

Keep numerically sensitive reductions in F32 unless a validated lower-precision path is equivalent.

Examples:

- LogSumExp / LogMeanExp
- probability normalization
- calibration
- gradient dot products/norms
- loss accumulation

Use BF16/FP16/FP8/NVFP4 where supported and validated, but preserve a higher-precision reference.

## Model-design principles

### Prefer modular architecture

Separate:

- encoder
- cached state representation
- option representation
- retrieval
- reranking
- joint decoder
- verifier
- calibrator
- abstain/escalate policy

Do not force one expensive path on every decision.

### Large option sets

For hundreds to millions of options, prefer a cascade:

```text
StateHandle
   ↓
semantic / memory retrieval
   ↓
top-K
   ↓
late interaction
   ↓
smaller K
   ↓
cross-attention / decision scorer
   ↓
joint verifier
```

Measure retrieval recall separately from decision accuracy.

### Dynamic options

Prefer semantic option representations with optional learned/memory residuals so unseen tools/actions can participate without retraining the full model.

### Set semantics

If option order is semantically irrelevant, architecture and tests should enforce permutation invariance/equivariance rather than relying only on shuffled training examples.

### Calibration

For decision systems, accuracy alone is insufficient.

Report:

- NLL
- Brier
- ECE and/or stronger calibration diagnostics
- confidence on errors
- risk/coverage where abstention exists

Avoid quantization or objective changes that improve speed/accuracy while silently destroying probability quality.

### Safety/control-plane heads

A specialized safety head should be isolated from unrelated decision types. Require a feature-off/non-target invariance test.

For token-localized safety evidence, test global and localized pooling under the same scorer capacity. Prefer semantic labels and family-held-out negatives over keyword rules.

## LLM and inference research

When optimizing generative models, inspect:

- prefill vs decode bottlenecks
- KV-cache size and precision
- paged/block allocation
- speculative acceptance rate
- kernel launch count
- quantized matmul path
- attention backend
- tensor parallel communication
- unified-memory behavior
- batch/sequence shape distribution

Benchmark on the actual hardware/backend targeted for deployment.

Do not extrapolate CUDA results to ROCm/Metal/Vulkan without measurement.

## Kernel and DSL research

For a kernel change:

1. preserve a numerically trusted reference
2. state expected bottleneck: memory bandwidth, arithmetic, occupancy, launch overhead, synchronization, etc.
3. test dominant real shapes, not only toy shapes
4. test cold and warm
5. record backend/device/compiler versions
6. compare generated DSL and hand-written implementations where relevant
7. record a refutation when the generated or hand kernel loses

Use `hanzo-research` so kernel evidence accumulates into the same research corpus as model-level experiments.

## Benchmark integrity

Never claim SOTA from mismatched protocols.

For competitor comparisons, match as closely as possible:

- dataset/version
- split
- prompts/options
- few-shot examples
- candidate count
- hardware where latency is compared
- calibration procedure
- retry/error policy

If protocols differ, say so explicitly.

Prefer paired comparisons on the same examples:

- stable correct
- damage
- repair
- persistent wrong
- McNemar or paired bootstrap where appropriate

## Output style for research tasks

When reporting a result, lead with:

1. what changed
2. what the result supports
3. what it does not yet establish
4. next decisive experiment

Use causal language proportional to the design.

Prefer:

> Persistent first-order gradient conflict is strongly disfavored under this matched regime.

over:

> We proved gradients never conflict.

Prefer:

> The result falsifies the architectural capacity-ceiling hypothesis.

over:

> The architecture is optimal.

## Production promotion gates

A research mechanism is not production-ready only because it improves one metric.

Before promotion require, as applicable:

- canonical Rust implementation
- invariant tests
- matched evaluation
- regression suite
- calibration non-inferiority
- tail/subgroup checks
- latency/memory measurement
- non-target invariance
- checkpoint/artifact provenance
- research record sealed in `/v1/research`

## Default decision rule

When choosing between two mechanisms with similar quality, prefer the one that is:

1. simpler
2. easier to falsify
3. cheaper at runtime
4. easier to reproduce
5. more native to the Hanzo stack
6. less coupled to benchmark-specific heuristics

The goal is not merely to win a benchmark. The goal is to produce a reproducible mechanism that becomes a durable primitive in Hanzo ML.
