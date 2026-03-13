# Hanzo Candle - Rust ML Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-ane.md`, `hanzo/rust-sdk.md`

## Overview

Hanzo Candle is a **Rust-based machine learning framework** — intended fork of HuggingFace candle for high-performance ML inference and training with GPU acceleration (CUDA + Metal).

**NOTE**: The `hanzoai/candle` GitHub repo **does not currently exist**. The upstream HuggingFace candle (v0.9.2-alpha.2) is used directly. A Hanzo fork with ANE support and PQC-safe operations is planned but not yet created.

### Upstream: HuggingFace Candle

The upstream `huggingface/candle` provides:
- Tensor operations with zero-cost Rust abstractions
- CUDA, Metal, MKL, and WASM backends
- 90+ model implementations (LLMs, vision, audio, diffusion)
- GGUF, safetensors, ONNX, PyTorch format support
- No Python runtime required

## When to use

- High-performance ML inference in Rust applications
- CUDA/Metal GPU acceleration for neural networks
- Loading and running GGUF, safetensors, ONNX models
- Building custom ML pipelines without Python
- Embedding ML in Hanzo Engine or Hanzo Node

## Hard requirements

1. **Rust 1.75+**
2. **CUDA Toolkit 12+** (for CUDA backend) or **macOS 13+** (for Metal)

## Quick reference

| Item | Value |
|------|-------|
| Upstream | `github.com/huggingface/candle` |
| Version | 0.9.2-alpha.2 |
| Planned fork | `github.com/hanzoai/candle` (not yet created) |
| Build | `cargo build --release` |
| Test | `cargo test` |
| License | MIT OR Apache-2.0 |

## Workspace Crates

### Core (built by default)

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor ops, Device abstraction, DType system |
| `candle-nn` | Neural network layers (Linear, Conv, LayerNorm, Attention) |
| `candle-transformers` | Transformer model implementations (90+) |
| `candle-datasets` | Dataset loading (MNIST, CIFAR, etc.) |
| `candle-pyo3` | Python bindings via PyO3 |
| `candle-ug` | Universal Graph backend |
| `tensor-tools` | CLI tensor manipulation |

### GPU Backends (opt-in, excluded from default build)

| Crate | Purpose |
|-------|---------|
| `candle-kernels` | Custom CUDA kernels |
| `candle-metal-kernels` | Custom Metal kernels (Apple GPU) |
| `candle-flash-attn` | Flash Attention v2 (CUDA) |
| `candle-flash-attn-v3` | Flash Attention v3 (CUDA) |
| `candle-onnx` | ONNX model evaluation |

### Backend Feature Flags

| Backend | Feature Flag | Notes |
|---------|-------------|-------|
| CPU | default | gemm crate for BLAS |
| CPU (Intel) | `mkl` | Intel Math Kernel Library |
| CPU (Apple) | `accelerate` | Apple Accelerate framework |
| CUDA | `cuda` | Via cudarc 0.18.2, cuBLAS, cuRAND |
| cuDNN | `cudnn` | Additional cuDNN kernels |
| NCCL | `nccl` | Multi-GPU distribution |
| Metal | `metal` | Apple GPU via objc2-metal |
| WASM | (target) | WebAssembly with SIMD |

## One-file quickstart

### Tensor Operations

```rust
use candle_core::{Device, Tensor, DType};

fn main() -> candle_core::Result<()> {
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
    let f = a.softmax(1)?;

    // Type conversion
    let a_bf16 = a.to_dtype(DType::BF16)?;

    Ok(())
}
```

### Neural Network

```rust
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap, Linear, linear, AdamW};

fn main() -> candle_core::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

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

    for (name, info) in model.tensor_infos.iter() {
        println!("{}: {:?}", name, info.shape);
    }

    let weights = model.tensor(&mut file, "model.layers.0.self_attn.q_proj.weight")?;
    println!("Weight shape: {:?}", weights.shape());
    Ok(())
}
```

### Load safetensors

```rust
use candle_core::{Device, DType};
use candle_nn::VarBuilder;

let device = Device::cuda_if_available(0)?;
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(
        &["model.safetensors"],
        DType::F32,
        &device,
    )?
};
let weight = vb.get((768, 768), "transformer.h.0.attn.c_attn.weight")?;
```

## Supported Models (90+ via candle-transformers)

| Category | Models |
|----------|--------|
| **LLMs** | LLaMA 1/2/3, Falcon, Gemma 1/2, Phi 1/1.5/2/3, Mistral, Mixtral, Mamba/Mamba2, StarCoder/2, Qwen3 MoE, Yi, GLM4, DeepSeek v2, SmolLM3, Olmo |
| **Vision** | DINOv2, ConvMixer, EfficientNet, ResNet, ViT, VGG, YOLO v3/v8, SAM, SegFormer, MobileNet v4, CLIP, SigLIP |
| **Audio** | Whisper, EnCodec, MetaVoice, Parler-TTS, Mimi, Silero VAD |
| **Diffusion** | Stable Diffusion 1.5/2.1/XL/3, Flux, Z-Image |
| **Multimodal** | BLIP, LLaVA, Moondream, PaddleOCR-VL, Pixtral, PaliGemma |
| **Quantized** | GGUF/GGML format, llama.cpp compatible |

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| GGUF | `.gguf` | Quantized models (llama.cpp compatible) |
| safetensors | `.safetensors` | HuggingFace standard (fast, safe) |
| ONNX | `.onnx` | Cross-framework interop |
| PyTorch | `.bin`, `.pt` | Legacy format |

## Cargo.toml Setup

```toml
[dependencies]
candle-core = { version = "0.9", features = ["cuda"] }  # NVIDIA GPU
# or
candle-core = { version = "0.9", features = ["metal"] }  # Apple GPU
# or
candle-core = { version = "0.9", features = ["mkl"] }    # Intel MKL

candle-nn = "0.9"
candle-transformers = "0.9"
```

## Planned Hanzo Extensions

When the `hanzoai/candle` fork is created:
- ANE (Apple Neural Engine) backend integration
- PQC-safe tensor operations
- Hanzo Engine serving integration
- Optimized Zen model loaders

## Related Skills

- `hanzo/hanzo-engine.md` - Uses candle for inference serving
- `hanzo/hanzo-ane.md` - Apple Neural Engine (complementary to Metal)
- `hanzo/hanzo-jin.md` - Visual JEPA framework
- `hanzo/rust-sdk.md` - Hanzo Rust SDK

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ml, rust, tensor, cuda, metal, inference
**Prerequisites**: Rust, ML fundamentals
