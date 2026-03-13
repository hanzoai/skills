# Hanzo ANE - Apple Neural Engine Training

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Hanzo ANE enables **training neural networks directly on Apple Neural Engine** across ALL Apple Silicon (M1+). Reverse-engineered private API for direct ANE access — not just inference but actual gradient computation and weight updates on the neural engine.

### Why Hanzo ANE?

- **ALL Apple Silicon**: Works on M1, M2, M3, M4+ (not just M4)
- **Direct ANE access**: Bypass CoreML, use private ANE API
- **Real training**: Gradient computation + weight updates on ANE
- **3 pipelines**: Static conv, Static+ANE extras, Dynamic matmul
- **Universal runtime**: Chip-specific optimizations auto-detected
- **Research tool**: Understand and benchmark ANE capabilities

### Key Discovery

The private API works on **ALL Apple Silicon (M1+)**. Earlier failures were caused by wrong weight blob format (64-byte header vs correct 128-byte header).

### OSS Base

Fork of **maderix/ANE** (8 commits merged). Repo: `hanzoai/ANE`.

## When to use

- Training small models on Apple Silicon ANE
- Benchmarking ANE vs GPU vs CPU performance
- Research into neural engine capabilities and limits
- Optimizing inference for Apple devices
- Edge AI model development on Mac

## Hard requirements

1. **Apple Silicon Mac** (M1, M2, M3, or M4)
2. **macOS 13+**
3. **Xcode Command Line Tools**
4. C/Objective-C compiler (clang)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/ANE` (fork of maderix/ANE) |
| Build | `make all && make test` |
| Data | `bash training/download_data.sh` |
| Language | Objective-C, Python |
| Upstream | maderix/ANE (8 commits merged, issue #3 has benchmarks) |

## Performance Benchmarks

**Model**: Stories110M (109M parameters)

| Chip | Pipeline | Speed | TFLOPS | Notes |
|------|----------|-------|--------|-------|
| M1 Max | Static (conv) | 206ms/step | 0.45 | ANE private API |
| M1 Max | MLX | N/A | 6.72 peak | Best training backend |
| M4 | Static (conv) | 107ms/step | — | Fastest ANE training |
| M4 | Dynamic (matmul) | 111ms/step | — | No recompile needed |

**Comparison**: MLX remains the best overall training backend on Apple Silicon (6.72 TFLOPS peak on M1 Max), but ANE offers unique advantages for deployment-optimized models.

## Three Pipelines

| Pipeline | Description | Ops | Compatibility |
|----------|-------------|-----|---------------|
| **Static (conv)** | Separate weight files, convolution ops | Conv2D | M1 → M4+ (all chips) |
| **Static + ANE extras** | 14% faster with ANE-specific optimizations | Conv2D + ANE | M1 → M4+ (all chips) |
| **Dynamic (matmul)** | Uses matmul + slice_by_size, no recompile | MatMul | **M4+ only** (fails on M1-M3) |

### Pipeline Selection

```
Which chip?
├── M1, M2, M3 → Static (conv) or Static + ANE extras
└── M4+        → Any pipeline (Dynamic recommended)
```

## Weight Blob Format

**CRITICAL**: Using the wrong format causes silent failures on M1-M3.

### Correct Format (128-byte header)
```
Offset  0: [general header, 64 bytes]
Offset 64: Magic 0xEFBEADDE (4 bytes)
Offset 72: Data size (8 bytes)
Offset 80: Data offset = 128 (8 bytes)
Offset 128: [weight data begins]
```

### Wrong Format (64-byte header)
- Missing magic at offset 64
- Data starts at offset 64 instead of 128
- Works accidentally on M4 but fails on M1-M3

## One-file quickstart

```bash
# Clone
git clone https://github.com/hanzoai/ANE.git
cd ANE

# Download training data
bash training/download_data.sh

# Build everything
make all

# Run tests
make test

# Run training (auto-detects chip and selects best pipeline)
make train

# Run specific pipeline (targets in training/Makefile)
cd training
make train              # Standard training
make train_large        # Large model training
make train_large_ane    # Large model + ANE extras

# Benchmark
make bench
```

### Universal Runtime

```c
// ane_universal.h — auto-detects chip capabilities
#include "ane_universal.h"

ane_device_t device = ane_open();

// Check capabilities
bool has_inmem = ane_supports_inmem(device);  // true on all chips now
bool has_dynamic = ane_supports_dynamic(device);  // true on M4+ only

// Select best pipeline
ane_pipeline_t pipeline;
if (has_dynamic) {
    pipeline = ane_create_dynamic_pipeline(device);
} else {
    pipeline = ane_create_static_pipeline(device);
}

// Load weights (128-byte header format)
ane_weights_t weights = ane_load_weights("model.ane");
ane_bind_weights(pipeline, weights);

// Forward pass
ane_tensor_t input = ane_create_tensor(device, shape, data);
ane_tensor_t output = ane_forward(pipeline, input);

// Backward pass (training)
ane_tensor_t grad = ane_backward(pipeline, output, target);
ane_update_weights(pipeline, grad, learning_rate);

ane_close(device);
```

## Project Structure

```
ANE/
├── ane_universal.h        # Universal runtime (chip detection)
├── Makefile               # Top-level build targets
├── training/
│   ├── Makefile           # Training targets (train, train_large, train_large_ane)
│   ├── download_data.sh   # Download Stories110M data
│   ├── train.m            # Training entry point (Objective-C)
│   ├── test_*.m           # Tests (Objective-C)
│   └── configs/           # Training configs
└── README.md
```

## MLX Comparison

For most training workloads, **MLX is recommended** over direct ANE:

| Feature | ANE (Direct) | MLX |
|---------|-------------|-----|
| Peak TFLOPS (M1 Max) | 0.45 | 6.72 |
| Ease of use | Low (C API) | High (Python) |
| Model support | Custom only | HuggingFace compatible |
| Training | Basic | Full (optimizers, schedulers) |
| Best for | Research, deployment-optimized | General training |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Fails on M1/M2/M3 | Wrong weight format | Use 128-byte header, not 64-byte |
| Dynamic pipeline crash | M1-M3 chip | Use static pipeline instead |
| Build error | Missing Xcode tools | `xcode-select --install` |
| Low TFLOPS | Wrong pipeline | Try Static + ANE extras |
| Magic number wrong | Byte order | Use little-endian `0xEFBEADDE` |

## Related Skills

- `hanzo/hanzo-engine.md` - Rust inference engine (can use ANE for serving)
- `hanzo/hanzo-candle.md` - Rust ML framework (Metal backend complement)
- `hanzo/zenlm.md` - Zen model family (training targets)
- `hanzo/hanzo-jin.md` - Multimodal framework

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ane, apple-silicon, neural-engine, training, m1, m4
**Prerequisites**: Apple Silicon Mac, C/Objective-C, macOS 13+
