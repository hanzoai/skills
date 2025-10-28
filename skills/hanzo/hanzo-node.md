# Hanzo Node - Distributed AI Mining & Networking

**Category**: Hanzo Ecosystem  
**Related Skills**: `zig/zig-project-setup.md`, `containers/docker-compose-development.md`, `observability/prometheus-monitoring.md`

## Overview

Hanzo Node is the foundational infrastructure piece of the Hanzo ecosystem - a **Rust-based distributed compute node** for privacy-first AI inference, model serving, and network mining. It provides high-performance, safe, and reliable AI infrastructure that runs on your hardware.

### Core Architecture

```
┌─────────────────────────────────────────────────┐
│           Hanzo Node Architecture                │
├─────────────────────────────────────────────────┤
│  API Layer (HTTP/gRPC/MCP)                      │
│  ├─ Inference Endpoints                         │
│  ├─ Model Management                            │
│  └─ Cluster Coordination                        │
├─────────────────────────────────────────────────┤
│  Execution Engine (Rust)                        │
│  ├─ GPU/CPU Scheduler                           │
│  ├─ Model Router                                │
│  ├─ Request Queue                               │
│  └─ Cache Layer                                 │
├─────────────────────────────────────────────────┤
│  P2P Networking                                  │
│  ├─ Peer Discovery                              │
│  ├─ Gossip Protocol                             │
│  ├─ Load Balancing                              │
│  └─ Consensus Engine                            │
├─────────────────────────────────────────────────┤
│  Storage & Models                                │
│  ├─ Model Registry (GGUF)                       │
│  ├─ KV Cache                                    │
│  └─ Metrics Store                               │
└─────────────────────────────────────────────────┘
```

### Why Rust?

- **Performance**: Near-C++ speeds with zero-cost abstractions
- **Safety**: Memory safety without garbage collection
- **Concurrency**: Fearless concurrency with ownership system
- **GPU**: Excellent CUDA/Metal bindings for acceleration
- **Deployment**: Single binary, minimal dependencies

### Key Capabilities

**Local Inference**: Run LLMs locally with GPU acceleration
- Llama 3 (8B, 70B), Mistral (7B, 8x7B), Phi-3, custom models
- GGUF format support for quantized models
- Automatic GPU layer offloading

**Network Mining**: Join distributed AI compute network
- Proof-of-inference consensus
- Reward distribution for compute contributions
- Automatic failover and load balancing

**Model Serving**: Production-ready API endpoints
- OpenAI-compatible API
- gRPC for high performance
- MCP for agentic workflows

**Cluster Coordination**: High availability deployment
- Leader election via Raft consensus
- Automatic peer discovery
- Health checking and recovery

## Quick Start

### Installation

```bash
# From source
git clone https://github.com/hanzoai/node.git
cd node
cargo build --release

# Install binary
cargo install hanzo-node

# Or via Docker
docker pull hanzoai/node:latest
```

### Configuration

```bash
# Initialize node
hanzo-node init --network mainnet

# This creates ~/.hanzo/config.toml:
```

```toml
[node]
network_id = "mainnet"
listen_addr = "0.0.0.0:8080"
p2p_port = 30303

[inference]
gpu_layers = 35          # GPU acceleration
max_batch_size = 32
context_size = 4096
rope_scaling = 1.0

[models]
cache_dir = "~/.hanzo/models"
auto_download = true

[mining]
enabled = true
wallet_address = "0x..."

[api]
cors_origins = ["http://localhost:3000"]
rate_limit = 100         # requests per minute
```

### Running Node

```bash
# Start with GPU mining
hanzo-node start --mine --gpu

# CPU-only mode
hanzo-node start --cpu-only

# Cluster mode (multi-node)
hanzo-node start --cluster --peers "node1:30303,node2:30303"

# Development mode (verbose logging)
RUST_LOG=debug hanzo-node start --dev
```

### Basic Operations

```bash
# Check status
hanzo-node status

# List available models
hanzo-node models list

# Download model
hanzo-node models pull llama-3-8b

# View peers
hanzo-node peers

# Check mining rewards
hanzo-node rewards balance

# Metrics
hanzo-node metrics
```

## Core Concepts

### GPU/CPU Scheduling

Hanzo Node intelligently schedules inference across available hardware:

```rust
// Automatic GPU layer offloading
// Model: Llama 3 70B (35GB)
// GPU VRAM: 24GB (RTX 4090)

// Hanzo Node automatically splits:
// - First 20 layers on GPU (fits in 24GB)
// - Remaining 15 layers on CPU
// - Efficient memory management

// Result: 3-4x faster than CPU-only
```

**Configuration**:
```toml
[inference]
gpu_layers = "auto"      # Automatic calculation
# or
gpu_layers = 20          # Manual override

[hardware]
gpu_memory_fraction = 0.9
cpu_threads = 16
```

### Model Registry

Models are stored in GGUF format with metadata:

```bash
~/.hanzo/models/
├── llama-3-8b-q4_k_m.gguf      # 4.6GB
├── llama-3-70b-q4_k_m.gguf     # 35GB
├── mistral-7b-instruct.gguf     # 4.1GB
├── phi-3-mini.gguf              # 2.3GB
└── custom-model.gguf

# Model metadata
~/.hanzo/models/registry.json
```

**Downloading Models**:
```bash
# From Hugging Face
hanzo-node models pull meta-llama/Llama-3-8B --quantization q4_k_m

# From custom URL
hanzo-node models add my-model https://example.com/model.gguf

# List installed
hanzo-node models list
# llama-3-8b (4.6GB) - Ready
# llama-3-70b (35GB) - Ready
# mistral-7b (4.1GB) - Ready
```

### P2P Networking

Nodes form a peer-to-peer network for:
- **Load balancing**: Distribute inference across cluster
- **Failover**: Automatic rerouting if node fails
- **Discovery**: Find other nodes automatically
- **Consensus**: Agree on network state

```toml
[p2p]
enabled = true
listen_addr = "0.0.0.0:30303"
bootstrap_nodes = [
    "/ip4/node1.hanzo.ai/tcp/30303",
    "/ip4/node2.hanzo.ai/tcp/30303"
]
max_peers = 50
```

**Peer Discovery**:
```bash
# View connected peers
hanzo-node peers
# Peer 1: node-a3f9 (latency: 45ms, models: 3)
# Peer 2: node-7c2b (latency: 120ms, models: 5)
# Peer 3: node-d8e1 (latency: 30ms, models: 2)

# Test peer connectivity
hanzo-node peers ping node-a3f9
```

### Request Router

Intelligent routing based on:
- Model availability across nodes
- Current load and queue depth
- Network latency to peers
- GPU/CPU capacity

```rust
// Automatic routing logic
match request {
    Model::Llama3_8B if local_gpu_available => route_local(),
    Model::Llama3_70B if peer_has_model => route_to_peer(closest_peer),
    _ if queue_full => return_429_rate_limit(),
    _ => queue_request()
}
```

## Hanzo Node API

### OpenAI-Compatible Endpoints

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Explain Rust"}],
    "temperature": 0.7,
    "stream": false
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Write a story"}],
    "stream": true
  }'

# Embeddings
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "input": "Text to embed"
  }'
```

### Native gRPC API

```protobuf
service HanzoNode {
  rpc Infer(InferRequest) returns (InferResponse);
  rpc InferStream(InferRequest) returns (stream InferResponse);
  rpc ListModels(Empty) returns (ModelsResponse);
  rpc NodeStatus(Empty) returns (StatusResponse);
}

message InferRequest {
  string model = 1;
  string prompt = 2;
  float temperature = 3;
  int32 max_tokens = 4;
}
```

**Using gRPC (Rust)**:
```rust
use hanzo_node_client::HanzoNodeClient;

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = HanzoNodeClient::connect("http://localhost:8080").await?;
    
    let request = InferRequest {
        model: "llama-3-8b".to_string(),
        prompt: "Explain ownership in Rust".to_string(),
        temperature: 0.7,
        max_tokens: 500,
    };
    
    let response = client.infer(request).await?;
    println!("Response: {}", response.text);
    
    Ok(())
}
```

### MCP Integration

Expose Hanzo Node as MCP tool for agents:

```typescript
// mcp-hanzo-node.ts
import { MCPServer, Tool } from '@hanzo/mcp'
import { HanzoNodeClient } from '@hanzo/node-client'

const node = new HanzoNodeClient({ url: 'http://localhost:8080' })

const inferTool: Tool = {
  name: 'hanzo_node_infer',
  description: 'Run inference on local Hanzo Node',
  parameters: {
    model: { 
      type: 'string',
      enum: ['llama-3-8b', 'llama-3-70b', 'mistral-7b'],
      required: true
    },
    prompt: { type: 'string', required: true },
    temperature: { type: 'number', default: 0.7 }
  },
  async execute({ model, prompt, temperature }) {
    const response = await node.infer({
      model,
      prompt,
      temperature
    })
    return response.text
  }
}

const server = new MCPServer({
  name: 'hanzo-node',
  tools: [inferTool]
})

server.listen(8081)
```

## Integration with Hanzo Ecosystem

### Hanzo Python SDK

```python
from hanzo import Hanzo

# Connect to local node
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'
)

# Inference
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

# Streaming
for chunk in hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[...],
    stream=True
):
    print(chunk.choices[0].delta.content, end='')
```

### @hanzo/ui Components

```typescript
import { useHanzoNode } from '@hanzo/ui/hooks'
import { AIChat, ModelSelector } from '@hanzo/ui'

export function Dashboard() {
  const node = useHanzoNode({ url: 'http://localhost:8080' })
  
  return (
    <>
      <ModelSelector 
        models={node.models}
        onSelect={node.selectModel}
      />
      
      <AIChat 
        inference={node.infer}
        streaming
      />
    </>
  )
}
```

### Hanzo Dev Agent

```bash
# Configure Hanzo Dev to use local node
hanzo-dev config set inference.provider hanzo-node
hanzo-dev config set inference.url http://localhost:8080

# Execute tasks with local inference
hanzo-dev task "Refactor authentication module"
```

## Production Deployment

### Docker

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN cargo install hanzo-node

COPY config.toml /root/.hanzo/config.toml

EXPOSE 8080 30303

CMD ["hanzo-node", "start", "--gpu"]
```

```bash
# Run container
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -p 30303:30303 \
  -v ~/.hanzo/models:/root/.hanzo/models \
  hanzoai/node:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  hanzo-node:
    image: hanzoai/node:latest
    ports:
      - "8080:8080"
      - "30303:30303"
    volumes:
      - ./models:/root/.hanzo/models
      - ./config.toml:/root/.hanzo/config.toml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      RUST_LOG: info
    command: ["start", "--gpu", "--mine"]
```

### Kubernetes

```yaml
# hanzo-node-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hanzo-node
spec:
  serviceName: hanzo-node
  replicas: 3
  selector:
    matchLabels:
      app: hanzo-node
  template:
    metadata:
      labels:
        app: hanzo-node
    spec:
      containers:
      - name: hanzo-node
        image: hanzoai/node:latest
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 30303
          name: p2p
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: models
          mountPath: /root/.hanzo/models
        - name: config
          mountPath: /root/.hanzo/config.toml
          subPath: config.toml
  volumeClaimTemplates:
  - metadata:
      name: models
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 200Gi
```

### Monitoring with Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hanzo-node'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

```bash
# Enable Prometheus metrics
hanzo-node config set metrics.enabled true
hanzo-node config set metrics.port 9090

# Metrics exposed:
# - hanzo_inference_requests_total
# - hanzo_inference_latency_seconds
# - hanzo_gpu_utilization_percent
# - hanzo_queue_depth
# - hanzo_peers_connected
```

## Advanced Configuration

### GPU Memory Management

```toml
[inference.gpu]
# Automatic (recommended)
layers = "auto"
memory_fraction = 0.9    # Use 90% of VRAM

# Manual control
layers = 20              # Specific layer count
kv_cache_size = "4GB"    # KV cache allocation

[inference.offload]
# Flash attention for memory efficiency
flash_attention = true

# Offload strategy
strategy = "balanced"    # balanced | gpu_first | cpu_first
```

### Request Queueing

```toml
[queue]
max_size = 1000
priority_levels = 3

[queue.timeouts]
enqueue_timeout_ms = 5000
processing_timeout_ms = 30000

[queue.fairness]
max_concurrent_per_client = 5
round_robin = true
```

### Caching Strategies

```toml
[cache.kv]
enabled = true
max_size_gb = 8
eviction = "lru"         # lru | lfu | fifo

[cache.prompt]
enabled = true
similarity_threshold = 0.95
max_entries = 10000
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify Hanzo Node sees GPU
hanzo-node info hardware
# GPU: NVIDIA RTX 4090 (24GB VRAM)
# CUDA: 12.2

# Force GPU usage
CUDA_VISIBLE_DEVICES=0 hanzo-node start --gpu
```

### Out of Memory

```toml
# Reduce GPU layers
[inference]
gpu_layers = 15          # Lower number

# Use smaller models
hanzo-node models pull llama-3-8b  # Instead of 70B

# Enable offloading
[inference.offload]
enabled = true
swap_size_gb = 16
```

### Slow Inference

```bash
# Check model quantization
hanzo-node models info llama-3-8b
# Quantization: Q4_K_M (recommended)

# Enable flash attention
hanzo-node config set inference.gpu.flash_attention true

# Increase batch size
hanzo-node config set inference.max_batch_size 64
```

### Peer Connection Issues

```toml
[p2p]
# Increase peer limits
max_peers = 100

# Add bootstrap nodes
bootstrap_nodes = [
    "/ip4/node1.hanzo.ai/tcp/30303",
    "/ip4/node2.hanzo.ai/tcp/30303"
]

# Enable hole punching for NAT
nat_traversal = true
```

## Performance Tuning

### Llama 3 8B Benchmarks

| Hardware | Config | Tokens/sec | Latency (P95) |
|----------|--------|-----------|---------------|
| RTX 4090 | 35 GPU layers | 120 | 8ms |
| RTX 3090 | 35 GPU layers | 95 | 10ms |
| M2 Max | Metal 30 layers | 85 | 12ms |
| CPU (16c) | No GPU | 25 | 40ms |

### Optimization Checklist

- ✅ Use Q4_K_M quantization (4-bit) for 8B-70B models
- ✅ Enable flash attention (`flash_attention = true`)
- ✅ Set `gpu_layers = "auto"` for optimal split
- ✅ Use batch inference when possible (`max_batch_size = 32`)
- ✅ Enable KV cache (`cache.kv.enabled = true`)
- ✅ Monitor with Prometheus (`metrics.enabled = true`)

## Related Skills

**Prerequisites**:
- `zig/zig-project-setup.md` - Similar systems programming concepts
- `containers/docker-compose-development.md` - Local development setup
- `cloud/kubernetes-deployment.md` - Production cluster deployment

**Integration**:
- `hanzo/hanzo-mcp.md` - Expose via MCP for agents
- `hanzo/python-sdk.md` - Python client integration
- `hanzo/hanzo-ui.md` - UI components for monitoring
- `hanzo/hanzo-dev.md` - Terminal agent integration

**Next Steps**:
- `hanzo/local-ai-architecture.md` - Full architecture patterns
- `hanzo/distributed-ai.md` - Multi-node cluster setup
- `observability/prometheus-monitoring.md` - Production monitoring

---

**Last Updated**: 2025-10-28  
**Category**: Hanzo Ecosystem  
**Related**: infrastructure, ml, observability  
**Prerequisites**: Rust basics, Docker, GPU drivers (NVIDIA/AMD/Metal)
