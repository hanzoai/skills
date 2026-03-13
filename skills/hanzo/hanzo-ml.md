# Hanzo ML - Rust ML Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-candle.md`, `hanzo/hanzo-engine.md`, `hanzo/hanzo-kensho.md`

## Overview

Hanzo ML is a **Rust-based minimalist ML framework** forked from HuggingFace Candle, optimized for edge AI, quantization, and multimodal inference. It provides the tensor computation layer for the Hanzo ecosystem, with GPU acceleration via CUDA, Metal (Apple Silicon), and CPU optimizations (MKL, Accelerate).

The repository maintains **dual crate namespaces**: original `candle-*` crates for upstream compatibility plus `hanzo-*` crates as the forward-looking Hanzo-branded API. Both coexist in the workspace and share the same underlying code.

### Key Differentiators from Upstream Candle

- **Hanzo-branded crates**: `hanzo-ml`, `hanzo-nn`, `hanzo-transformers`, `hanzo-datasets` published at v0.9.2-alpha.2
- **HANZO_INTEGRATION.md**: Integration guide for Hanzo Engine (mistral-rs fork)
- **Hanzo-specific kernels**: `hanzo-flash-attn`, `hanzo-metal-kernels`, `hanzo-kernels`
- **Python bindings**: `hanzo-ml-pyo3` for Python interop
- **WASM examples**: `hanzo-ml-wasm-examples` for browser inference

## When to use

- High-performance ML inference in Rust applications
- CUDA/Metal GPU acceleration without Python runtime
- Loading GGUF, safetensors, ONNX models in Rust
- Browser-based ML via WebAssembly
- Integration with Hanzo Engine for model serving
- Building custom inference pipelines with zero-cost abstractions

## Hard requirements

1. **Rust 1.75+**
2. **CUDA Toolkit 12+** (for CUDA backend) or **macOS 13+** (for Metal)
3. For WASM: `wasm-pack` and compatible browser

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/ml` |
| Branch | `main` |
| Language | Rust (primary), C++/CUDA/Metal/Python |
| Version | 0.9.2-alpha.2 (hanzo crates), 0.9.2 (candle crates) |
| Build | `cargo build --workspace` |
| Test | `cargo test --workspace` |
| License | BSD-3-Clause OR Apache-2.0 |

## Workspace Crates

### Hanzo-Branded (v0.9.2-alpha.2)

| Crate | Purpose |
|-------|---------|
| `hanzo-ml` | Core tensor operations, Device, DType, quantization |
| `hanzo-nn` | Neural network layers (Linear, Conv, LayerNorm, Attention) |
| `hanzo-transformers` | Transformer model implementations (90+) |
| `hanzo-datasets` | Dataset loading (MNIST, CIFAR, TinyStories) |
| `hanzo-ml-pyo3` | Python bindings via PyO3 |
| `hanzo-flash-attn` | Flash Attention v2 CUDA kernels |
| `hanzo-metal-kernels` | Custom Metal GPU kernels |
| `hanzo-kernels` | Custom CUDA kernels |
| `hanzo-onnx` | ONNX model evaluation |
| `hanzo-ug` | Universal Graph backend |
| `hanzo-ml-examples` | Example binaries |
| `hanzo-ml-wasm-examples` | Browser WASM examples |
| `hanzo-ml-wasm-tests` | WASM test suite |

### Upstream-Compatible (v0.9.2)

| Crate | Purpose |
|-------|---------|
| `candle-core` | Original tensor core (same code as hanzo-ml) |
| `candle-nn` | Original NN layers |
| `candle-transformers` | Original transformer models |
| `candle-datasets` | Original datasets |
| `candle-examples` | Original examples |
| `candle-pyo3` | Original Python bindings |

### GPU Backend Crates (opt-in)

| Crate | Purpose |
|-------|---------|
| `hanzo-kernels` | Custom CUDA kernels (reduce, cast, affine, etc.) |
| `hanzo-metal-kernels` | Custom Metal kernels (Apple GPU) |
| `hanzo-flash-attn` | Flash Attention v2 (CUDA SM80+) |

## Feature Flags

| Feature | Effect |
|---------|--------|
| `cuda` | NVIDIA GPU via cudarc 0.19.1 |
| `cudnn` | Additional cuDNN kernels |
| `nccl` | Multi-GPU distribution |
| `mkl` | Intel Math Kernel Library |
| `accelerate` | Apple Accelerate framework |
| `metal` | Apple GPU via hanzo-metal-kernels + objc2-metal |
| `ug` | Universal Graph backend |

## One-file quickstart

### Tensor Operations

```rust
use hanzo_ml::{Device, Tensor, DType};

fn main() -> hanzo_ml::Result<()> {
    let device = Device::cuda_if_available(0)?;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
    let c = a.matmul(&b)?;
    println!("Shape: {:?}", c.shape()); // [2, 4]

    let d = a.relu()?;
    let e = a.softmax(1)?;
    let f = a.to_dtype(DType::BF16)?;

    Ok(())
}
```

### Neural Network Training

```rust
use hanzo_ml::{Device, Tensor, DType, Module};
use hanzo_nn::{VarBuilder, VarMap, Linear, linear, AdamW};

fn main() -> hanzo_ml::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let layer1 = linear(784, 256, vb.pp("layer1"))?;
    let layer2 = linear(256, 10, vb.pp("layer2"))?;

    let input = Tensor::randn(0f32, 1., (32, 784), &device)?;
    let h = layer1.forward(&input)?.relu()?;
    let output = layer2.forward(&h)?;

    let mut opt = AdamW::new(varmap.all_vars(), Default::default())?;
    let target = Tensor::zeros((32, 10), DType::F32, &device)?;
    let loss = hanzo_nn::loss::mse(&output, &target)?;
    opt.backward_step(&loss)?;

    println!("Loss: {}", loss.to_scalar::<f32>()?);
    Ok(())
}
```

### Load GGUF Model

```rust
use hanzo_ml::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let mut file = File::open("model.gguf")?;
    let model = gguf_file::Content::read(&mut file)?;

    for (name, info) in model.tensor_infos.iter() {
        println!("{}: {:?}", name, info.shape);
    }
    Ok(())
}
```

### Load safetensors

```rust
use hanzo_ml::{Device, DType};
use hanzo_nn::VarBuilder;

let device = Device::cuda_if_available(0)?;
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(
        &["model.safetensors"], DType::F32, &device,
    )?
};
let weight = vb.get((768, 768), "transformer.h.0.attn.c_attn.weight")?;
```

## Cargo.toml Setup

```toml
[dependencies]
# From crates.io (when published)
hanzo-ml = { version = "0.9.2-alpha.2", features = ["metal"] }
hanzo-nn = "0.9.2-alpha.2"
hanzo-transformers = "0.9.2-alpha.2"

# From git (current)
hanzo-ml = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-nn = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-transformers = { git = "https://github.com/hanzoai/ml", branch = "main" }
```

## Integration with Hanzo Engine

Hanzo Engine (mistral-rs fork) uses Hanzo ML as its tensor backend:

```toml
# In engine Cargo.toml
[dependencies]
hanzo-ml = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-nn = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-transformers = { git = "https://github.com/hanzoai/ml", branch = "main" }

[features]
default = ["metal"]
metal = ["hanzo-ml/metal", "hanzo-nn/metal"]
cuda = ["hanzo-ml/cuda"]
```

### Quantization Support

| Format | Use Case |
|--------|----------|
| GGUF/GGML | Universal, llama.cpp compatible |
| AFQ (Affine) | Optimized for Metal/Apple Silicon |
| GPTQ/AWQ | GPU-optimized quantization |
| ISQ | In-situ runtime quantization |

## Supported Models (90+ via hanzo-transformers)

| Category | Models |
|----------|--------|
| **LLMs** | LLaMA 1/2/3, Falcon, Gemma 1/2, Phi 1-3, Mistral, Mixtral, Mamba/2, StarCoder/2, Qwen3 MoE, Yi, GLM4, DeepSeek v2, SmolLM3, Olmo |
| **Vision** | DINOv2, ConvMixer, EfficientNet, ResNet, ViT, VGG, YOLO v3/v8, SAM, SegFormer, MobileNet v4, CLIP, SigLIP |
| **Audio** | Whisper, EnCodec, MetaVoice, Parler-TTS, Mimi, Silero VAD |
| **Diffusion** | Stable Diffusion 1.5/2.1/XL/3, Flux |
| **Multimodal** | BLIP, LLaVA, Moondream, PaddleOCR-VL, Pixtral, PaliGemma |
| **Quantized** | GGUF/GGML format, llama.cpp compatible |

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| GGUF | `.gguf` | Quantized models (llama.cpp compatible) |
| safetensors | `.safetensors` | HuggingFace standard (fast, safe) |
| ONNX | `.onnx` | Cross-framework interop |
| PyTorch | `.bin`, `.pt` | Legacy format |

## Project Structure

```
ml/
├── hanzo-ml/                 # Core tensor ops (hanzo-branded)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── tensor.rs         # Tensor type
│   │   ├── device.rs         # CPU/CUDA/Metal device
│   │   ├── dtype.rs          # Data types (F16, BF16, F32, etc.)
│   │   ├── backend.rs        # Backend trait
│   │   ├── cuda_backend/     # CUDA implementation
│   │   ├── metal_backend/    # Metal implementation
│   │   ├── cpu_backend/      # CPU implementation
│   │   ├── quantized/        # GGUF/GGML quantization
│   │   └── safetensors.rs    # safetensors loading
│   ├── benches/              # Performance benchmarks
│   └── tests/                # Unit tests
├── hanzo-nn/                 # Neural network layers
├── hanzo-transformers/       # 90+ transformer model implementations
├── hanzo-datasets/           # Dataset loading utilities
├── hanzo-ml-pyo3/            # Python bindings
├── hanzo-flash-attn/         # Flash Attention CUDA kernels
│   └── kernels/              # CUDA kernel source files
├── hanzo-metal-kernels/      # Metal GPU kernels
├── hanzo-kernels/            # Generic CUDA kernels
├── hanzo-onnx/               # ONNX evaluation
├── hanzo-ml-examples/        # Example binaries
├── hanzo-ml-wasm-examples/   # WASM browser examples
├── candle-core/              # Upstream-compatible core (v0.9.2)
├── candle-nn/                # Upstream-compatible NN
├── candle-transformers/      # Upstream-compatible transformers
├── candle-datasets/          # Upstream-compatible datasets
├── candle-examples/          # Upstream-compatible examples
├── candle-book/              # Documentation book
├── tensor-tools/             # CLI tensor manipulation
├── Cargo.toml                # Workspace root
├── Makefile
├── HANZO_INTEGRATION.md      # Engine integration guide
└── LLM.md
```

## Development Workflow

```bash
# Build entire workspace
cargo build --workspace

# Test everything
cargo test --workspace

# Build with Metal (Apple Silicon)
cargo build --workspace --features metal

# Build with CUDA
cargo build --workspace --features cuda

# Run example (LLaMA inference)
cargo run --release --example llama -- --model meta-llama/Llama-3.2-3B-Instruct

# Sync from upstream
git remote add upstream https://github.com/huggingface/candle.git
git fetch upstream
git merge upstream/main
```

## Performance Considerations

### Apple Silicon (Metal)
- Use `AFQ4` quantization for best throughput
- Enable `--features "metal accelerate"` for CPU fallback ops
- Group size 64 balances speed and accuracy

### CUDA
- Use GPTQ or AWQ quantization
- Flash Attention enabled via `hanzo-flash-attn` (SM80+)
- PagedAttention for memory efficiency in long sequences

### CPU
- GGUF models with appropriate quantization level
- `mkl` feature for Intel platforms
- `accelerate` feature for Apple platforms

## Related Skills

- `hanzo/hanzo-candle.md` - Upstream candle documentation (planned fork)
- `hanzo/hanzo-engine.md` - Inference engine using hanzo-ml
- `hanzo/hanzo-ane.md` - Apple Neural Engine (complementary to Metal)
- `hanzo/hanzo-kensho.md` - Image generation model
- `hanzo/hanzo-sho.md` - Text diffusion engine

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: ml, rust, tensor, cuda, metal, inference, quantization, wasm
**Prerequisites**: Rust, ML fundamentals, GPU programming concepts
