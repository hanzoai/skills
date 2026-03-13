# Hanzo ANE - Apple Neural Engine Training

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Hanzo ANE enables **training neural networks on Apple Neural Engine** across ALL Apple Silicon (M1+). Reverse-engineered private API for direct ANE access — not just inference but actual gradient computation and weight updates on the neural engine.

### Key Discovery

The private API works on **ALL Apple Silicon (M1+)**, not just M4. Earlier failures were caused by wrong weight blob format.

### OSS Base

Fork of **maderix/ANE**. Repo: `hanzoai/ANE`. Local: ``github.com/hanzoai/ANE``.

## When to use

- Training small models on Apple Silicon
- Benchmarking ANE performance
- Research into neural engine capabilities
- Optimizing inference for Apple devices

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/ANE` (fork of maderix/ANE) |
| Build | `make all && make test` |
| Data | `bash training/download_data.sh` |
| Upstream | maderix/ANE (8 commits merged) |

## Performance

| Chip | Static Training | Notes |
|------|----------------|-------|
| M1 Max | 206ms/step, 0.45 TFLOPS | Stories110M (109M params) |
| M4 | 107ms/step | Static + dynamic both work |
| MLX | 6.72 TFLOPS peak (M1 Max) | Best training backend overall |

## Three Pipelines

| Pipeline | Description | Compatibility |
|----------|-------------|---------------|
| Static (conv) | Separate weight files, conv ops | M1 → M4+ |
| Static + ANE extras | 14% faster with ANE optimizations | M1 → M4+ |
| Dynamic (matmul) | matmul + slice_by_size | M4+ only |

## Weight Blob Format

**Correct format** (128-byte header):
- Magic: `0xEFBEADDE` at offset 64
- Data size at offset 72
- Data offset 128 at offset 80

**Wrong format** (64-byte header) — causes failures on M1-M3.

## One-file quickstart

```bash
cd ANE
bash training/download_data.sh   # Get training data
make all                         # Build everything
make test                        # Run tests

# Universal runtime detects InMem on all chips
# ane_universal.h handles chip detection
```

## Related Skills

- `hanzo/hanzo-engine.md` - Rust inference engine
- `hanzo/hanzo-candle.md` - Rust ML framework
- `hanzo/zenlm.md` - Zen model family

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ane, apple-silicon, neural-engine, training
**Prerequisites**: Apple Silicon Mac, C/Objective-C, Metal
