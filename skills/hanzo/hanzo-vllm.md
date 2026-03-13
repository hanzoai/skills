# Hanzo vLLM - Rust-Based LLM Inference Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-candle.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-engine.md`

## Overview

Hanzo vLLM is a **Rust LLM inference server** with an OpenAI-compatible API, PagedAttention, continuous batching, and multi-GPU/multi-node support. Fork of `EricLBuehler/candle-vllm`, built on top of `guoqingbao/candle` (a fork of Hugging Face's Candle ML framework). Binary name: `candle-vllm`. Version 0.2.1.

### Why Hanzo vLLM?

- **Pure Rust**: No Python runtime -- single binary, fast cold start, minimal dependencies
- **OpenAI API compatible**: Drop-in replacement at `/v1/chat/completions` and `/v1/models`
- **PagedAttention**: Efficient KV cache management for high throughput
- **Continuous batching**: Batched decoding across concurrent requests
- **Multi-GPU**: Both multi-process (recommended) and multi-threaded modes via NCCL
- **Multi-node**: MPI runner for distributed inference across machines
- **Apple Silicon**: Metal backend for Mac inference (GGUF models)
- **Quantization**: In-situ quantization, GGUF, GPTQ, AWQ, Marlin formats
- **11 model architectures**: LLAMA, Mistral, Phi, Qwen2/3, Yi, StableLM, Gemma-2/3, DeepSeek, QwQ, GLM4

### Tech Stack

- **Language**: Rust (edition 2021)
- **ML Framework**: Candle (guoqingbao/candle fork, v0.8.3)
- **HTTP Server**: Axum 0.7.4 with tower-http CORS
- **Tokenizer**: HuggingFace tokenizers 0.21.2
- **GPU**: CUDA + NCCL (multi-GPU), Metal (Apple Silicon)
- **Quantization**: Custom CUDA kernels (`kernels/`), Metal kernels (`metal-kernels/`)
- **Distributed**: MPI (optional, for multi-node)
- **Template Engine**: MiniJinja (chat templates)

### OSS Base

Repo: `hanzoai/vllm` (90MB). Fork of `EricLBuehler/candle-vllm`. MIT License (Copyright 2023 Eric Buehler). Default branch: `master`.

## When to use

- Serving LLMs locally with an OpenAI-compatible API
- Running inference on Apple Silicon (Metal) without Python
- Multi-GPU inference for large models (DeepSeek-R1 671B)
- Production inference where a single Rust binary is preferred over Python vLLM
- Quantized model serving (GGUF, GPTQ, AWQ, Marlin)

## When NOT to use

- **LLM Gateway routing**: Use `hanzo/hanzo-llm-gateway.md` (proxies to 100+ providers)
- **Python vLLM**: If you need the full Python vLLM ecosystem (LoRA, speculative decoding, vision models)
- **Training**: This is inference-only; use MLX or PyTorch for training

## Hard requirements

1. **Rust 1.83.0+** toolchain
2. **CUDA Toolkit** in PATH for GPU builds (NVIDIA)
3. **Metal** for Apple Silicon builds
4. **NCCL** for multi-GPU
5. **MPI** (libopenmpi-dev) for multi-node

## Quick reference

| Item | Value |
|------|-------|
| Binary | `candle-vllm` |
| Version | `0.2.1` |
| Default Port | `2000` |
| API | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) |
| Repo | `github.com/hanzoai/vllm` |
| Branch | `master` |
| License | MIT |
| Upstream | `EricLBuehler/candle-vllm` |

## Supported Models

| Model | Type | BF16 Speed (A100) | Quantized Speed |
|-------|------|--------------------|-----------------|
| LLAMA | llama/llama3 | 65 tks/s (8B) | 115 tks/s (8B, Marlin) |
| Mistral | mistral | 70 tks/s (7B) | 115 tks/s (7B, Marlin) |
| Phi | phi2/phi3 | 107 tks/s (3.8B) | 135 tks/s (3.8B) |
| Qwen2/Qwen3 | qwen2/qwen3 | 81 tks/s (8B) | - |
| Yi | yi | 75 tks/s (6B) | 105 tks/s (6B) |
| StableLM | stable-lm | 99 tks/s (3B) | - |
| Gemma-2/3 | gemma/gemma3 | 60 tks/s (9B) | 73 tks/s (9B, Marlin) |
| DeepSeek R1 Distill | deep-seek | 48 tks/s (14B) | 62 tks/s (14B) |
| DeepSeek V2/V3/R1 | deep-seek | - | ~20 tks/s (AWQ 671B, tp=8) |
| QwQ-32B | qwen2 | 30 tks/s (32B, tp=2) | 36 tks/s (32B, Q4K) |
| GLM4 | glm4 | 55 tks/s (9B) | 92 tks/s (9B, Q4K) |

## Repository Structure

```
vllm/
  Cargo.toml              # candle-vllm v0.2.1, feature flags
  build.rs
  LICENSE                 # MIT
  README.md
  README-CN.md
  src/
    main.rs               # CLI entry, Axum server setup
    lib.rs                 # Core library exports
    openai/                # OpenAI-compatible API layer
      openai_server.rs     # Axum routes (/v1/chat/completions, /v1/models)
      communicator.rs      # Request/response coordination
      distributed.rs       # Multi-GPU/multi-node communication
      streaming.rs         # SSE streaming responses
      requests.rs          # Request types
      responses.rs         # Response types
      sampling_params.rs   # Temperature, top-k, top-p, penalties
      logits_processor.rs  # Token sampling logic
      utils.rs
      conversation/        # Chat template handling
      models/              # Model implementations (17 files)
        llama.rs, mistral.rs, phi2.rs, phi3.rs
        qwen.rs, yi.rs, stable_lm.rs, gemma.rs, gemma3.rs
        glm4.rs, deepseek.rs
        quantized_llama.rs, quantized_phi3.rs
        quantized_qwen.rs, quantized_glm4.rs
        linear.rs          # Shared linear layer (Marlin/GPTQ)
        mod.rs             # Model registry and loading
      pipelines/           # Inference pipeline
        pipeline.rs        # Model loading, weight management
        llm_engine.rs      # Continuous batching engine
        mod.rs
    backend/               # Low-level inference backend
      paged_attention.rs   # PagedAttention implementation
      cache.rs             # KV cache management
      gguf.rs              # GGUF format loader
      gptq.rs              # GPTQ format loader
      heartbeat.rs         # Health monitoring
      progress.rs          # Loading progress bars
      custom_ops/          # Custom operations
      mod.rs
    paged_attention/       # PagedAttention core algorithms
    scheduler/             # Request scheduling (continuous batching)
  kernels/                 # CUDA kernels
    Cargo.toml
    build.rs               # CUDA kernel compilation
    src/
  metal-kernels/           # Metal kernels (Apple Silicon)
    Cargo.toml
    src/
  examples/
    chat.py                # Python chat client
    benchmark.py           # Batch throughput benchmark
    convert_marlin.py      # GPTQ to Marlin conversion
    convert_awq_marlin.py  # AWQ to Marlin conversion
    llama.py               # Simple LLAMA example
  tests/
```

## Build

```bash
# Clone
git clone git@github.com:hanzoai/vllm.git
cd vllm

# Apple Silicon (Metal)
cargo build --release --features metal

# CUDA single-node (single or multi-GPU)
export PATH=$PATH:/usr/local/cuda/bin/
cargo build --release --features cuda,nccl

# CUDA with flash attention (faster for long context)
cargo build --release --features cuda,nccl,flash-attn

# CUDA multi-node (MPI)
sudo apt install libopenmpi-dev openmpi-bin clang libclang-dev -y
cargo build --release --features cuda,nccl,mpi
```

### Cargo Features

| Feature | What it enables |
|---------|----------------|
| `cuda` | NVIDIA GPU support + CUDA kernels |
| `metal` | Apple Silicon GPU support + Metal kernels |
| `nccl` | Multi-GPU communication (requires cuda) |
| `flash-attn` | Flash Attention (requires cuda, faster long-context) |
| `mpi` | Multi-node distributed inference |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL |
| `cudnn` | cuDNN |

## Running Models

### Uncompressed (BF16/F16)

```bash
# From local path
target/release/candle-vllm --port 2000 \
  --weight-path /home/DeepSeek-R1-Distill-Llama-8B/ llama3 \
  --temperature 0. --penalty 1.0

# From HuggingFace
target/release/candle-vllm \
  --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B qwen3
```

### GGUF Quantized

```bash
# Apple Silicon
cargo run --release --features metal -- --port 2000 --dtype bf16 \
  --weight-file /home/qwq-32b-q4_k_m.gguf qwen2 \
  --quant gguf --temperature 0. --penalty 1.0

# From HuggingFace
target/release/candle-vllm \
  --model-id unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF \
  --weight-file DeepSeek-R1-0528-Qwen3-8B-Q2_K.gguf qwen3 --quant gguf
```

### In-Situ Quantization

```bash
# Load unquantized model as quantized
target/release/candle-vllm --port 2000 \
  --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q4k
```

Quantization options: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`

### GPTQ/Marlin

```bash
target/release/candle-vllm --dtype bf16 --port 2000 \
  --weight-path /home/model-GPTQ-4bit qwen2 \
  --quant gptq --temperature 0. --penalty 1.0
```

### Multi-GPU

```bash
# Multi-process mode (recommended)
cargo run --release --features cuda,nccl -- \
  --multi-process --dtype bf16 --port 2000 \
  --device-ids "0,1" --weight-path /home/QwQ-32B/ qwen2 \
  --penalty 1.0 --temperature 0.

# GPU count must be power of 2 (2, 4, 8)
```

### DeepSeek-R1 671B (CPU offloading)

```bash
# Convert AWQ to Marlin format
python3 examples/convert_awq_marlin.py \
  --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/

# Run on 8x A100 with 120/256 experts offloaded to CPU
cargo run --release --features cuda,nccl -- \
  --log --multi-process --dtype bf16 --port 2000 \
  --device-ids "0,1,2,3,4,5,6,7" \
  --weight-path /data/DeepSeek-R1-AWQ-Marlin/ deep-seek \
  --quant awq --temperature 0. --penalty 1.0 \
  --num-experts-offload-per-rank 15
```

## Sending Requests

### curl

```bash
curl -X POST "http://127.0.0.1:2000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### Python (OpenAI SDK)

```python
import openai
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Explain Rust."}],
    max_tokens=64,
)
print(completion.choices[0].message.content)
```

### Chat Client

```bash
pip install openai rich click
python3 examples/chat.py                           # Plain text
python3 examples/chat.py --thinking True            # Reasoning models
python3 examples/chat.py --live                     # Markdown rendering
```

### Benchmark

```bash
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--port` | Server port | 2000 |
| `--dtype` | Data type (bf16, f16) | Model default |
| `--weight-path` | Local model directory | - |
| `--model-id` | HuggingFace model ID | - |
| `--weight-file` | Specific weight file (GGUF) | - |
| `--quant` | Quantization format | None |
| `--device-ids` | GPU IDs ("0,1,2,3") | "0" |
| `--multi-process` | Multi-process GPU mode | false |
| `--kvcache-mem-gpu` | KV cache GPU memory (MB) | 4096 |
| `--max-gen-tokens` | Max output tokens | 1/5 of max_seq_len |
| `--temperature` | Sampling temperature | 0.7 |
| `--penalty` | Repetition penalty | 1.0 |
| `--top-k` | Top-k sampling | - |
| `--top-p` | Top-p (nucleus) sampling | - |
| `--thinking` | Enable reasoning mode | false |
| `--log` | Enable logging | false |
| `--num-experts-offload-per-rank` | CPU offload (MoE models) | 0 |

## Relationship to Hanzo Candle

Hanzo vLLM depends on `guoqingbao/candle` (a fork of `huggingface/candle`) for tensor operations, neural network layers, and GPU backends. The Hanzo Candle repo (`hanzoai/candle`) is the same Candle framework but maintained under the Hanzo org. They serve different purposes:

- **hanzoai/candle** -- ML framework (tensors, layers, GPU ops)
- **hanzoai/vllm** -- Inference server (API, batching, scheduling, PagedAttention) built on Candle

## Related Skills

- `hanzo/hanzo-candle.md` - Rust ML framework (underlying tensor engine)
- `hanzo/hanzo-cloud.md` - Cloud LLM inference API (uses this as a backend)
- `hanzo/hanzo-llm-gateway.md` - LLM proxy gateway (routes to this and 100+ providers)
- `hanzo/hanzo-engine.md` - GPU scheduling and ML job management

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: vllm, inference, rust, llm, gpu, metal, quantization
**Prerequisites**: Rust 1.83+, CUDA or Metal, model weights
