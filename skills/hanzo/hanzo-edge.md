# Hanzo Edge - On-Device AI Inference Runtime

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Hanzo Edge is an **on-device AI inference runtime** for running Zen models and any GGUF model locally on macOS, Linux, Web (WASM), and embedded devices. Zero cloud dependency, full data privacy, zero network latency, works completely offline. Rust workspace with three crates. Live at `edge.hanzo.ai`.

### Why Hanzo Edge?

- **On-device**: Data never leaves the device, works offline
- **Cross-platform**: macOS (Metal), Linux (CPU/CUDA), Web (WASM), embedded (ARM)
- **OpenAI-compatible**: Local API server as drop-in replacement
- **GGUF native**: First-class quantized model support (Q4_K, Q5_K, Q8_0)
- **Streaming**: Token-by-token via SSE and callbacks
- **HuggingFace Hub**: Automatic model download and caching

### Tech Stack

- **Language**: Rust (edition 2021)
- **ML Backend**: Candle (candle-core, candle-nn, candle-transformers v0.9)
- **HTTP Server**: Axum 0.7
- **CLI**: Clap 4
- **WASM**: wasm-bindgen 0.2, web-sys 0.3
- **Tokenizer**: HuggingFace tokenizers 0.20
- **Async**: Tokio 1

### OSS Base

Repo: `hanzoai/edge`. Built on [Hanzo ML](https://github.com/hanzoai/ml) (Candle fork) for tensor operations.

## When to use

- Running AI models locally with full data privacy
- Mobile, embedded, or offline inference
- In-browser AI via WebAssembly
- Low-latency inference without network round-trips
- Prototyping with local OpenAI-compatible API

## Edge vs Engine

| | **Edge** | **Engine** |
|---|---|---|
| **Where** | On-device (local CPU/GPU) | Cloud GPU clusters |
| **Latency** | Zero network overhead | Network round-trip |
| **Privacy** | Data never leaves device | Data sent to cloud |
| **Models** | Quantized GGUF (Q4/Q5/Q8) | Full-precision (FP16/BF16) |
| **Best for** | Mobile, embedded, offline, privacy | Production serving, large models, scale |

## Hard requirements

1. **Rust toolchain** (stable)
2. **wasm-pack** for WASM builds
3. **CUDA toolkit** for NVIDIA GPU acceleration (optional)
4. **Xcode Command Line Tools** for Metal on macOS (optional)

## Quick reference

| Item | Value |
|------|-------|
| Docs | `https://edge.hanzo.ai` |
| Crates.io | `hanzo-edge`, `hanzo-edge-core`, `hanzo-edge-wasm` |
| Image | `ghcr.io/hanzoai/edge:latest` |
| API Port | 8080 (local serve) |
| License | Apache-2.0 |
| Repo | `github.com/hanzoai/edge` |

## One-file quickstart

### Run a model locally

```bash
# Install
cargo install hanzo-edge

# Run inference (auto-downloads model from HuggingFace)
hanzo-edge run --model zenlm/zen3-nano --prompt "Hello!"

# Start OpenAI-compatible local server
hanzo-edge serve --model zenlm/zen4-mini --port 8080

# Then use from any OpenAI client
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zen4-mini",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    }'
```

### Docker

```bash
# CPU inference
docker run --rm -it ghcr.io/hanzoai/edge:latest \
    run --model zenlm/zen3-nano --prompt "Hello!"

# Serve as API
docker run --rm -p 8080:8080 ghcr.io/hanzoai/edge:latest \
    serve --model zenlm/zen3-nano --port 8080
```

## Core Concepts

### Architecture

```
hanzo-edge (workspace)
+-- edge-core/              # Core inference runtime (library)
|   +-- src/
|       +-- lib.rs           # Public API: Model, InferenceSession, SamplingParams
|       +-- model.rs         # Model trait, GGUF loading, HF Hub download
|       +-- session.rs       # Autoregressive generation + streaming iterator
|       +-- sampling.rs      # Temperature, top-k, top-p, repeat penalty
|       +-- tokenizer.rs     # HF tokenizer wrapper with EOS detection
+-- edge-cli/               # CLI binary
|   +-- src/
|       +-- main.rs          # Clap-based CLI with 4 subcommands
|       +-- loader.rs        # HF Hub download with progress bars
|       +-- cmd/
|           +-- run.rs       # Streaming inference to stdout
|           +-- serve.rs     # OpenAI-compatible HTTP server (Axum)
|           +-- bench.rs     # TTFT, throughput, memory benchmarking
|           +-- info.rs      # Model metadata inspection
+-- edge-wasm/              # WebAssembly module
    +-- src/lib.rs           # WASM bindings: EdgeModel, generate, generate_stream
    +-- index.html           # Demo page
```

### Crates

| Crate | Description | Install |
|-------|-------------|---------|
| `hanzo-edge-core` | Core inference runtime and `Model` trait | `cargo add hanzo-edge-core` |
| `hanzo-edge` | CLI binary with run, serve, bench, info | `cargo install hanzo-edge` |
| `hanzo-edge-wasm` | Browser WASM module with streaming | `wasm-pack build edge-wasm` |

### Feature Flags

| Feature | Description | Build |
|---------|-------------|-------|
| `cpu` | CPU backend (default) | `cargo build --release` |
| `metal` | Metal backend for macOS/iOS | `cargo build --release --features metal` |
| `cuda` | CUDA backend for NVIDIA GPUs | `cargo build --release --features cuda` |

### CLI Subcommands

```bash
# Streaming inference
hanzo-edge run --model zenlm/zen-eco --prompt "Write a haiku" \
    --max-tokens 128 --temperature 0.7 --top-p 0.9

# Model metadata
hanzo-edge info --model zenlm/zen4-mini

# Benchmarking (TTFT, tok/s, memory)
hanzo-edge bench --model zenlm/zen3-nano --prompt "Hello" \
    --max-tokens 128 -n 5

# OpenAI-compatible API server
hanzo-edge serve --model zenlm/zen4-mini --port 8080
```

### API Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Health check |

### Rust SDK Usage

```rust
use hanzo_edge_core::{load_model, InferenceSession, SamplingParams, ModelConfig};

let config = ModelConfig {
    model_id: "zenlm/zen4-mini".to_string(),
    model_file: Some("zen4-mini.Q4_K_M.gguf".to_string()),
    ..Default::default()
};
let (mut model, tokenizer) = load_model(&config)?;

let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    max_tokens: 256,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
};
let mut session = InferenceSession::new(&mut *model, &tokenizer, params);
let output = session.generate("Explain quantum computing")?;
```

### WASM (Browser)

```bash
# Build
wasm-pack build edge-wasm --target web
```

```javascript
import init, { EdgeModel } from './pkg/edge_wasm.js';

await init();
const model = new EdgeModel(modelBytes, tokenizerBytes);
const output = model.generate("Hello!", 256, 0.7);

// Streaming
model.generate_stream("Write a poem", 256, 0.7, (token) => {
    document.getElementById('output').textContent += token;
});
```

### Zen Models for Edge

| Model | Params | Quantized Size | Use Case |
|-------|--------|----------------|----------|
| `zenlm/zen3-nano` | 600M | ~400MB (Q4_K_M) | Ultra-lightweight, embedded, IoT |
| `zenlm/zen-eco` | 4B | ~2.5GB (Q4_K_M) | General purpose, mobile, tablets |
| `zenlm/zen4-mini` | 8B | ~5GB (Q4_K_M) | High quality, desktop and laptop |

## Building from Source

```bash
git clone https://github.com/hanzoai/edge && cd edge

# CPU
cargo build --release -p hanzo-edge

# Metal (Apple Silicon)
cargo build --release -p hanzo-edge --features metal

# CUDA
cargo build --release -p hanzo-edge --features cuda

# WASM
make build-wasm

# Tests
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings
```

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (Apple Silicon) | Metal | Production |
| macOS (Intel) | CPU / Accelerate | Production |
| Linux x86_64 | CPU | Production |
| Linux x86_64 | CUDA | Production |
| Linux ARM64 | CPU | Production |
| Web (WASM) | CPU | Stable |
| iOS | Metal / CoreML | Planned |
| Android | Vulkan / NNAPI | Planned |

## Related Skills

- `hanzo/hanzo-engine.md` - Cloud GPU inference (complementary to Edge)
- `hanzo/hanzo-llm-gateway.md` - Unified LLM proxy for 100+ providers
- `hanzo/hanzo-candle.md` - Rust ML framework (tensor ops, used by Edge)
- `hanzo/zenlm.md` - Zen model family

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: edge, inference, wasm, on-device, gguf, rust
**Prerequisites**: Rust toolchain, GGUF model understanding
