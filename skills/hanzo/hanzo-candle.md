# Hanzo Candle - Rust ML Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-ane.md`, `hanzo/rust-sdk.md`

## Overview

Hanzo Candle is a **Rust-based machine learning framework** — fork of HuggingFace candle. Tensor operations, neural network layers, and GPU acceleration (CUDA + Metal) for high-performance ML inference and training.

### OSS Base

Fork of **huggingface/candle**. Local: ``github.com/hanzoai/candle``.

## When to use

- High-performance ML inference in Rust
- CUDA/Metal GPU acceleration
- Loading GGUF, safetensors, ONNX models
- Building custom ML pipelines in Rust
- Embedding candle in Hanzo Engine

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/candle` |
| Build | `cargo build --release` |
| Test | `cargo test` |
| Features | `cuda`, `metal`, `mkl` |

## One-file quickstart

```rust
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder};

fn main() -> candle_core::Result<()> {
    let device = Device::cuda_if_available(0)?;

    // Create tensors
    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    // Matrix multiply
    let c = a.matmul(&b)?;
    println!("Shape: {:?}", c.shape()); // [2, 4]

    Ok(())
}
```

### Load GGUF Model

```rust
use candle_core::quantized::gguf_file;

let model = gguf_file::Content::read(&mut file)?;
let weights = model.tensor(&mut file, "model.layers.0.self_attn.q_proj.weight")?;
```

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| GGUF | `.gguf` | Quantized models (llama.cpp) |
| safetensors | `.safetensors` | HuggingFace standard |
| ONNX | `.onnx` | Cross-framework |
| PyTorch | `.bin`, `.pt` | Legacy |

## Related Skills

- `hanzo/hanzo-engine.md` - Uses candle for inference
- `hanzo/hanzo-ane.md` - Apple Neural Engine
- `hanzo/hanzo-jin.md` - Multimodal LLM framework

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ml, rust, tensor, cuda, metal
**Prerequisites**: Rust, ML fundamentals, GPU programming basics
