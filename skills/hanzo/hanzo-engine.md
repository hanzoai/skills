# Hanzo Engine - Native Rust Inference & Embedding Engine

**Category**: Hanzo Ecosystem
**Skill Level**: Intermediate to Advanced
**Prerequisites**: Rust knowledge helpful but not required for API usage
**Related Skills**: hanzo-node.md, python-sdk.md, go-sdk.md, zenlm.md

## Overview

**Hanzo Engine** is Hanzo AI's high-performance, Rust-native inference and embedding engine that powers the entire Hanzo ecosystem. Built on mistral.rs with Hanzo-specific optimizations, it provides **blazing-fast local inference** for all ZenLM models and industry-standard LLMs.

**Core Philosophy**: Maximum performance, native Rust implementation, multimodal support, and seamless integration with Hanzo Node and Cloud.

## Key Features

### 🚀 **Blazingly Fast Performance**
- **Native Rust**: Zero-overhead implementation
- **PagedAttention**: Memory-efficient attention mechanism
- **FlashAttention**: 2-8x faster attention computation
- **ISQ (In-Situ Quantization)**: On-the-fly model quantization
- **MLX**: Optimized for Apple Silicon (M1/M2/M3)

### 🔮 **All-in-One Multimodal**
- **Text ↔ Text**: Standard LLM inference
- **Text + Vision ↔ Text**: Vision-language models
- **Text + Vision + Audio ↔ Text**: Multimodal understanding
- **Text → Speech**: Text-to-speech generation
- **Text → Image**: Image generation (coming soon)

### 🎯 **Embeddings First-Class**
- **Optimized for Qwen3-Embedding**: #1 on MTEB multilingual benchmark
- **Multiple dimensions**: 1024, 2048, 4096 dims
- **Reranking support**: Qwen3-Reranker models
- **Production-ready**: High-throughput batch processing

### 🌐 **Multiple APIs**
- **Rust API**: Native high-performance integration
- **Python API**: PyO3 bindings (via mistralrs-pyo3)
- **OpenAI HTTP**: Compatible with OpenAI Chat Completions & Embeddings API
- **MCP Support**: Model Context Protocol for agentic workflows

### 🔗 **Embedded Everywhere**
- **Hanzo Node**: Local inference and mining
- **Cloud Nodes**: Distributed inference clusters
- **Hanzo Desktop**: Native macOS app with mining
- **Python/Go SDKs**: Connect via OpenAI-compatible API

## Architecture

### **Hanzo Engine in the Stack**

```
┌─────────────────────────────────────────────────┐
│           ZenLM Models (zen-nano, zen-eco)      │
│  zen-agent, zen-musician, zen-thinking, etc.    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│         Hanzo Engine (Rust - port 36900)        │
│  ├─ Native inference & embeddings               │
│  ├─ Optimized for Qwen3 models (#1 MTEB)       │
│  ├─ Multimodal: text, vision, audio            │
│  ├─ PagedAttention, FlashAttention, ISQ, MLX   │
│  └─ MCP support built-in                        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│           Embedded in:                          │
│  ├─ Hanzo Node (local inference)                │
│  └─ Cloud Nodes (distributed inference)         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│           SDKs connect to Node/Cloud:           │
│  ├─ Python SDK (hanzoai package)                │
│  ├─ Go SDK (github.com/hanzoai/go-sdk)          │
│  └─ JavaScript/TypeScript SDK (@hanzo/sdk)      │
└─────────────────────────────────────────────────┘
```

### **Based on mistral.rs**

Hanzo Engine extends [mistral.rs](https://github.com/EricLBuehler/mistral.rs) with:
- Custom CLI tool (`hanzo-engine` command)
- Model management (pull, list, cache)
- OpenAI-compatible server (default port: 36900)
- Hanzo-specific optimizations for ZenLM models
- Integration with Hanzo Node and Cloud infrastructure

**Upstream Sync**: Fully synchronized with mistral.rs (commit `530463af1`)

## Installation

### From Source (Recommended)

```bash
# Clone Hanzo Engine
git clone https://github.com/hanzoai/engine.git
cd engine

# Build for macOS (Metal backend)
cargo build --package hanzo-engine --release --no-default-features --features metal

# Build for Linux (CUDA backend)
cargo build --package hanzo-engine --release --features cuda

# Install binary
cargo install --path hanzo-engine --no-default-features --features metal
```

### Via Cargo

```bash
# Install from GitHub
cargo install --git https://github.com/hanzoai/engine hanzo-engine

# With CUDA support (Linux)
cargo install --git https://github.com/hanzoai/engine hanzo-engine --features cuda
```

### Verify Installation

```bash
hanzo-engine --version
hanzo-engine --help
```

## Quick Start

### 1. Pull a Model

```bash
# Pull ZenLM model (zen-eco-4B)
hanzo-engine pull qwen/qwen3-4b

# Pull embedding model (Qwen3-Embedding-8B)
hanzo-engine pull qwen/qwen3-embedding-8b

# Pull from Ollama
hanzo-engine pull ollama://llama3.2:3b

# Pull GGUF model from URL
hanzo-engine pull https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

### 2. Start the Server

```bash
# Start on default port (36900)
hanzo-engine serve

# With specific port
hanzo-engine serve --port 36900

# With specific model
hanzo-engine serve --model qwen3-4b

# With custom model directory
hanzo-engine serve --model-dir ~/.hanzo/models
```

### 3. Test Inference

```bash
# Chat completions (OpenAI-compatible)
curl -X POST http://localhost:36900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Explain Rust ownership"}],
    "temperature": 0.7
  }'

# Embeddings
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-8b",
    "input": "Hello, Hanzo Engine!"
  }'
```

## ZenLM Native Support

All **ZenLM models** are **natively supported** in Hanzo Engine:

### zen-nano (0.6B)
```bash
hanzo-engine pull qwen/qwen3-0.6b
hanzo-engine serve --model qwen3-0.6b
```

### zen-eco (4B) - instruct/thinking/agent variants
```bash
# zen-eco-instruct
hanzo-engine pull qwen/qwen3-4b

# zen-eco-thinking (with chain-of-thought)
hanzo-engine pull qwen/qwen3-4b-cot

# zen-agent (tool calling & function execution)
hanzo-engine pull qwen/qwen3-4b-function-calling
```

### zen-musician (7B) - Music generation with lyrics
```bash
hanzo-engine pull map/yue-s1-7b-anneal-en-cot
```

### zen-vision (Vision-language)
```bash
# Qwen3-VL for vision understanding
hanzo-engine pull qwen/qwen3-vl-8b
```

**Performance**: All ZenLM models run at **50+ tokens/sec** on consumer hardware (M1, RTX 3060).

## Integration with Hanzo Node

Hanzo Engine is the **core inference backend** for Hanzo Node:

### Configuration

```bash
# In Hanzo Node configuration (.env or environment)
export EMBEDDINGS_SERVER_URL="http://localhost:36900"  # Engine port
export EMBEDDING_MODEL_TYPE="qwen3-embedding-8b"
export USE_NATIVE_EMBEDDINGS="true"
export HANZO_ENGINE_ENABLED="true"
```

### Automatic Connection

When you start Hanzo Node, it **automatically connects** to Hanzo Engine:

```bash
# Start Hanzo Engine
hanzo-engine serve --port 36900

# Start Hanzo Node (connects to Engine on port 36900)
hanzo-node start --mine --gpu
```

Hanzo Node will:
- Use Engine for all inference requests
- Generate embeddings via Engine's Qwen3-Embedding models
- Leverage Engine's multimodal capabilities
- Benefit from PagedAttention and FlashAttention optimizations

## Python SDK Integration

```python
from hanzo import Hanzo

# Connect to local Hanzo Engine (via Hanzo Node or directly)
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'  # Hanzo Node (which uses Engine)
)

# Or connect directly to Engine
hanzo_engine = Hanzo(
    base_url='http://localhost:36900',
    api_key='not-needed-for-local'
)

# Chat completion (uses zen-eco via Engine)
response = hanzo_engine.chat.completions.create(
    model='qwen3-4b',
    messages=[{'role': 'user', 'content': 'Explain Rust'}]
)

# Embeddings (uses Qwen3-Embedding via Engine)
embedding = hanzo_engine.embeddings.create(
    model='qwen3-embedding-8b',
    input='Hello, Hanzo Engine!'
)

print(f"Embedding dimensions: {len(embedding.data[0].embedding)}")  # 4096
```

## Go SDK Integration

```go
package main

import (
    "context"
    "fmt"
    "github.com/hanzoai/go-sdk"
    "github.com/hanzoai/go-sdk/option"
)

func main() {
    // Connect to Hanzo Engine via Hanzo Node
    client := hanzoai.NewClient(
        option.WithInferenceMode("local"),
        option.WithNodeURL("http://localhost:8080"),
    )

    // Chat completion
    response, err := client.Chat.Completions.Create(
        context.Background(),
        hanzoai.ChatCompletionCreateParams{
            Model: hanzoai.F("qwen3-4b"),
            Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                hanzoai.UserMessage("Explain Rust ownership"),
            }),
        },
    )

    if err != nil {
        panic(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

## Rust Native API

For maximum performance, use the native Rust API:

```rust
use mistralrs::{
    ChatCompletionRequest, IsqType, Loader, MistralRs, ModelDType,
    NormalLoaderBuilder, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, SchedulerConfig, TokenSource
};
use tokio::sync::mpsc::channel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load ZenLM model (zen-eco-4B)
    let loader = NormalLoaderBuilder::new(
        "qwen/qwen3-4b",
        None,
        None,
        Some(ModelDType::Auto),
    )
    .build();

    let model = loader.load_model().await?;
    let pipeline = model.build_pipeline(SchedulerConfig::default_config())?;

    let mistralrs = MistralRs::new(pipeline)?;

    // Create request
    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::Chat(vec![
            ChatCompletionRequest {
                role: "user".to_string(),
                content: "Explain Rust ownership".to_string(),
            }
        ]),
        sampling_params: SamplingParams::default(),
        response: tx,
        ..Default::default()
    });

    // Send request
    mistralrs.get_sender().send(request).await?;

    // Receive response
    let response = rx.recv().await.unwrap();
    println!("{}", response.choices[0].text);

    Ok(())
}
```

## Model Management

### Pull Models

```bash
# From HuggingFace
hanzo-engine pull qwen/qwen3-4b

# From Ollama
hanzo-engine pull ollama://llama3.2:3b

# From MLX Community
hanzo-engine pull mlx://mlx-community/Llama-3.2-3B-Instruct-4bit

# From direct URL
hanzo-engine pull https://example.com/model.gguf
```

### List Downloaded Models

```bash
hanzo-engine list

# Output:
# Downloaded models:
# - qwen3-4b (4.3 GB) - /Users/z/.hanzo/models/qwen3-4b
# - qwen3-embedding-8b (8.1 GB) - /Users/z/.hanzo/models/qwen3-embedding-8b
# - llama3.2-3b (3.2 GB) - /Users/z/.hanzo/models/llama3.2-3b
```

### Model Storage

Default model directory: `~/.hanzo/models/`

Custom directory:
```bash
export HANZO_MODELS_DIR=/path/to/models
hanzo-engine serve --model-dir /path/to/models
```

## Performance Optimizations

### PagedAttention

Memory-efficient attention with dynamic memory allocation:

```bash
# Enable PagedAttention (default for long contexts)
hanzo-engine serve --paged-attention

# Adjust memory usage
hanzo-engine serve --gpu-memory-fraction 0.9
```

**Benefits**:
- 2-4x higher throughput
- Support for longer sequences
- Efficient KV cache management

### FlashAttention

Ultra-fast attention computation:

```bash
# FlashAttention is automatic with CUDA
cargo build --release --features "cuda flash-attn"

# Verify FlashAttention is active
hanzo-engine serve --log-level debug
# Look for: "Using FlashAttention for faster inference"
```

**Benefits**:
- 2-8x faster attention
- 10-20% less memory usage
- No accuracy degradation

### In-Situ Quantization (ISQ)

On-the-fly model quantization for lower memory:

```bash
# Quantize to 4-bit on load
hanzo-engine serve --isq Q4K

# Quantize to 8-bit
hanzo-engine serve --isq Q8_0

# Available formats: Q4K, Q4_0, Q5K, Q8_0, Q8_1
```

**Benefits**:
- 50-75% memory reduction
- Minimal accuracy loss (<2% on benchmarks)
- Faster loading times

### MLX (Apple Silicon)

Optimized for M1/M2/M3 Macs:

```bash
# Build with Metal backend (macOS)
cargo build --package hanzo-engine --release --no-default-features --features metal

# Use MLX models for maximum performance
hanzo-engine pull mlx://mlx-community/Llama-3.2-3B-Instruct-4bit
hanzo-engine serve --model llama-3.2-3b-instruct-4bit
```

**Benefits on M1 Max**:
- 50+ tokens/sec for 4B models
- 30+ tokens/sec for 7B models
- Unified memory = faster transfers

## MCP Support

Hanzo Engine has **built-in MCP support** for agentic workflows:

```bash
# Start Engine with MCP enabled
hanzo-engine serve --mcp-enabled --mcp-port 3691

# MCP tools automatically exposed:
# - hanzo_infer: Run inference on any model
# - hanzo_embed: Generate embeddings
# - hanzo_list_models: List available models
```

### MCP Integration Example

```typescript
import { MCPClient } from '@hanzo/mcp'

const client = new MCPClient({
  host: 'localhost',
  port: 3691
})

// Use Hanzo Engine tools via MCP
const result = await client.callTool('hanzo_infer', {
  model: 'qwen3-4b',
  prompt: 'Explain MCP',
  temperature: 0.7
})

console.log(result.response)
```

## Supported Models

### Embedding Models (Optimized)
- **Qwen3-Embedding-8B**: 4096 dims, #1 on MTEB multilingual
- **Qwen3-Embedding-4B**: 2048 dims, balanced performance
- **Qwen3-Embedding-0.6B**: 1024 dims, lightweight

### Reranker Models
- **Qwen3-Reranker-4B**: Superior reranking quality
- **Qwen3-Reranker-0.6B**: Lightweight reranking

### LLM Models
All models from mistral.rs:
- **Llama**: 2, 3, 3.1, 3.2, 3.3
- **Mistral**: v0.1, v0.2, v0.3, Nemo, Large
- **Phi**: Phi-2, Phi-3, Phi-3.5
- **Gemma**: Gemma, Gemma 2
- **Qwen**: Qwen2, Qwen2.5, **Qwen3** (ZenLM)
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3
- **Yi**: Yi-1.5, Yi-Coder
- And many more...

### Vision-Language Models
- **Qwen3-VL**: Vision understanding (ZenLM)
- **Llama-3.2-Vision**: Vision + language
- **Pixtral**: Mistral's vision model
- **Idefics2/3**: Open vision-language models

## Hardware Requirements

### Minimum (zen-nano 0.6B)
- **CPU**: Any modern CPU (x86_64, ARM64)
- **RAM**: 2 GB
- **GPU**: Optional (runs on CPU)

### Recommended (zen-eco 4B)
- **CPU**: 4+ cores
- **RAM**: 8 GB
- **GPU**: 8 GB VRAM (RTX 3060, M1 8GB)

### Optimal (zen-musician 7B)
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **GPU**: 16 GB VRAM (RTX 3090, M1 Max 24GB)

### Performance Benchmarks

| Model | Hardware | Tokens/sec | Latency |
|-------|----------|------------|---------|
| zen-nano 0.6B | M1 8GB | 100+ | 10ms |
| zen-eco 4B | M1 16GB | 50+ | 20ms |
| zen-eco 4B | RTX 3060 12GB | 45+ | 22ms |
| zen-musician 7B | M1 Max 32GB | 35+ | 28ms |
| zen-musician 7B | RTX 3090 24GB | 60+ | 16ms |

## Troubleshooting

### Engine won't start

```bash
# Check port availability
lsof -i :36900

# Use different port
hanzo-engine serve --port 37000

# Check logs
hanzo-engine serve --log-level debug
```

### Model not found

```bash
# Verify model is downloaded
hanzo-engine list

# Re-pull model
hanzo-engine pull qwen/qwen3-4b --force

# Check model directory
ls ~/.hanzo/models/
```

### Out of memory

```bash
# Use quantized model
hanzo-engine serve --model qwen3-4b --isq Q4K

# Reduce GPU memory usage
hanzo-engine serve --gpu-memory-fraction 0.8

# Use smaller model
hanzo-engine pull qwen/qwen3-0.6b
```

### Slow inference

```bash
# Enable FlashAttention (CUDA)
cargo build --release --features "cuda flash-attn"

# Enable PagedAttention
hanzo-engine serve --paged-attention

# Use quantized model for faster loading
hanzo-engine pull qwen/qwen3-4b-q4k
```

## Hanzo Engine vs Alternatives

| Feature | Hanzo Engine | llama.cpp | vLLM | Ollama |
|---------|--------------|-----------|------|--------|
| **Language** | Rust | C++ | Python | Go |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Multimodal** | ✅ All | ✅ Limited | ✅ Yes | ✅ Yes |
| **Embeddings** | ✅ Optimized | ❌ No | ✅ Yes | ✅ Yes |
| **MCP Support** | ✅ Native | ❌ No | ❌ No | ❌ No |
| **PagedAttention** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **FlashAttention** | ✅ Yes | ✅ Limited | ✅ Yes | ❌ No |
| **Apple Silicon** | ✅ MLX | ✅ Metal | ❌ No | ✅ Metal |
| **Model Management** | ✅ Built-in | Manual | Manual | ✅ Built-in |
| **OpenAI API** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Hanzo Integration** | ✅ Native | ❌ No | ❌ No | ❌ No |

**Why Hanzo Engine?**
- **Native Hanzo integration**: Seamless with Node, Cloud, SDKs
- **ZenLM optimized**: Best performance for Zoo Labs models
- **Embeddings first-class**: Qwen3-Embedding #1 on MTEB
- **MCP native**: Built-in agentic workflow support
- **Rust performance**: Maximum speed with memory safety
- **All-in-one**: Text, vision, audio, embeddings, reranking

## Related Skills

- **hanzo-node.md**: Hanzo Node embeds Engine for local inference
- **python-sdk.md**: Python SDK connects to Engine via Node
- **go-sdk.md**: Go SDK connects to Engine via Node
- **zenlm.md**: ZenLM models run natively in Engine
- **hanzo-mcp.md**: MCP integration patterns with Engine

## Additional Resources

- **GitHub**: https://github.com/hanzoai/engine
- **Documentation**: https://docs.hanzo.ai/engine
- **Hanzo AI**: https://hanzo.ai
- **mistral.rs**: https://github.com/EricLBuehler/mistral.rs (upstream)

---

**Remember**: Hanzo Engine is the **native Rust inference and embedding engine** powering the entire Hanzo ecosystem. All ZenLM models run natively with optimal performance through Engine's advanced features like PagedAttention, FlashAttention, and ISQ.
