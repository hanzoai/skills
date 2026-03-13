# Hanzo Candle - Rust ML Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-ane.md`, `hanzo/rust-sdk.md`

## Overview

Hanzo Candle is a **Rust-based machine learning framework** — fork of HuggingFace candle. Tensor operations, neural network layers, and GPU acceleration (CUDA + Metal) for high-performance ML inference and training.

### Why Hanzo Candle?

- **Rust performance**: Zero-cost abstractions, no GC pauses
- **Multi-backend**: CUDA, Metal, MKL, CPU
- **Model formats**: GGUF, safetensors, ONNX, PyTorch
- **Minimal dependencies**: No Python runtime required
- **Embeddable**: Compile into any Rust binary
- **Hanzo extensions**: ANE support, PQC-safe operations

### OSS Base

Fork of **huggingface/candle**. Repo: `hanzoai/candle`.

## When to use

- High-performance ML inference in Rust applications
- CUDA/Metal GPU acceleration for neural networks
- Loading and running GGUF, safetensors, ONNX models
- Building custom ML pipelines without Python
- Embedding ML in Hanzo Engine or Node

## Hard requirements

1. **Rust 1.75+**
2. **CUDA Toolkit 12+** (for CUDA backend) or **macOS 13+** (for Metal)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/candle` |
| Upstream | huggingface/candle |
| Build | `cargo build --release` |
| Test | `cargo test` |
| Features | `cuda`, `metal`, `mkl`, `accelerate` |
| License | MIT OR Apache-2.0 |

## Workspace Crates

### Core

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations, device management, dtypes |
| `candle-nn` | Neural network layers (Linear, Conv, LayerNorm, Attention) |
| `candle-datasets` | Dataset loading (MNIST, CIFAR, etc.) |

### Backends

| Crate | Purpose |
|-------|---------|
| `candle-kernels` | Custom CUDA kernels |
| `candle-metal-kernels` | Custom Metal kernels |
| `candle-flash-attn` | Flash Attention (CUDA) |

### Model Support

| Crate | Purpose |
|-------|---------|
| `candle-transformers` | Transformer model implementations |
| `candle-onnx` | ONNX model loading and execution |
| `candle-pyo3` | Python bindings (PyO3) |

### Utilities

| Crate | Purpose |
|-------|---------|
| `candle-examples` | Example implementations |
| `candle-wasm-examples` | WebAssembly examples |
| `candle-wasm-tests` | WASM test suite |

## One-file quickstart

### Tensor Operations

```rust
use candle_core::{Device, Tensor, DType};

fn main() -> candle_core::Result<()> {
    // Auto-select best device
    let device = Device::cuda_if_available(0)?;

    // Create tensors
    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    // Matrix multiply
    let c = a.matmul(&b)?;
    println!("Shape: {:?}", c.shape()); // [2, 4]

    // Element-wise operations
    let d = (&a + &a)? * 2.0;
    let e = a.relu()?;
    let f = a.softmax(1)?;  // Softmax over dim 1

    // Reshape
    let g = a.reshape((6,))?;
    let h = a.transpose(0, 1)?;

    // Type conversion
    let a_f16 = a.to_dtype(DType::F16)?;
    let a_bf16 = a.to_dtype(DType::BF16)?;

    Ok(())
}
```

### Neural Network

```rust
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap, Linear, linear, Optimizer, AdamW};

fn main() -> candle_core::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Build a simple MLP
    let layer1 = linear(784, 256, vb.pp("layer1"))?;
    let layer2 = linear(256, 10, vb.pp("layer2"))?;

    // Forward pass
    let input = Tensor::randn(0f32, 1., (32, 784), &device)?;
    let h = layer1.forward(&input)?.relu()?;
    let output = layer2.forward(&h)?;

    // Training
    let mut opt = AdamW::new(varmap.all_vars(), Default::default())?;
    let target = Tensor::zeros((32, 10), DType::F32, &device)?;
    let loss = candle_nn::loss::mse(&output, &target)?;
    opt.backward_step(&loss)?;

    println!("Loss: {}", loss.to_scalar::<f32>()?);
    Ok(())
}
```

### Load GGUF Model

```rust
use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let mut file = File::open("model.gguf")?;
    let model = gguf_file::Content::read(&mut file)?;

    // List tensors
    for (name, info) in model.tensor_infos.iter() {
        println!("{}: {:?}", name, info.shape);
    }

    // Load specific tensor
    let weights = model.tensor(&mut file, "model.layers.0.self_attn.q_proj.weight")?;
    println!("Weight shape: {:?}", weights.shape());

    Ok(())
}
```

### Load safetensors Model

```rust
use candle_core::{Device, DType};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    // Load from safetensors
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["model.safetensors"],
            DType::F32,
            &device,
        )?
    };

    // Access weights
    let weight = vb.get((768, 768), "transformer.h.0.attn.c_attn.weight")?;
    println!("Weight shape: {:?}", weight.shape());

    Ok(())
}
```

### Supported Models (via candle-transformers)

| Model | Type | GGUF | safetensors |
|-------|------|------|-------------|
| LLaMA / Zen | Text LLM | Yes | Yes |
| Mistral | Text LLM | Yes | Yes |
| Phi | Text LLM | Yes | Yes |
| BERT | Encoder | No | Yes |
| Whisper | Audio | No | Yes |
| Stable Diffusion | Image | No | Yes |
| CLIP | Vision-Text | No | Yes |

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| GGUF | `.gguf` | Quantized models (llama.cpp compatible) |
| safetensors | `.safetensors` | HuggingFace standard (fast, safe) |
| ONNX | `.onnx` | Cross-framework interop |
| PyTorch | `.bin`, `.pt` | Legacy format |

## Feature Flags

```toml
# Cargo.toml
[dependencies]
candle-core = { version = "0.8", features = ["cuda"] }  # NVIDIA GPU
candle-core = { version = "0.8", features = ["metal"] }  # Apple GPU
candle-core = { version = "0.8", features = ["mkl"] }    # Intel MKL
candle-core = { version = "0.8", features = ["accelerate"] }  # Apple Accelerate
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA not found | Missing CUDA toolkit | Install CUDA 12+ |
| Metal error | Wrong macOS version | Requires macOS 13+ |
| OOM | Model too large | Use quantized GGUF or reduce batch size |
| Slow CPU | No SIMD | Enable `mkl` or `accelerate` feature |

## Related Skills

- `hanzo/hanzo-engine.md` - Uses candle for inference serving
- `hanzo/hanzo-ane.md` - Apple Neural Engine (complementary to Metal)
- `hanzo/hanzo-jin.md` - Multimodal LLM (uses candle core)
- `hanzo/rust-sdk.md` - Hanzo Rust SDK

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ml, rust, tensor, cuda, metal, inference
**Prerequisites**: Rust, ML fundamentals, GPU programming basics
