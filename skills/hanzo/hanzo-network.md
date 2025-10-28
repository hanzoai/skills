# Hanzo Network - Distributed AI Agent Topology

**Category**: Hanzo Ecosystem
**Skill Level**: Intermediate to Advanced
**Prerequisites**: Understanding of distributed systems, P2P networking, gRPC
**Related Skills**: python-sdk.md, go-sdk.md, hanzo-node.md, hanzo-mcp.md

## Overview

Hanzo Network enables **distributed AI agent execution** across heterogeneous devices (M1 MacBooks, NVIDIA GPUs, Linux servers, Raspberry Pis, etc.) with automatic peer discovery, device capability detection, and intelligent load balancing.

**Core Innovation**: Scan QR code (or use config file) → Join network → Share compute → Run AI agents across devices seamlessly.

## Why Hanzo Network?

### The Problem: Siloed AI Compute

- **Single device limited**: Your MacBook can't leverage your desktop's GPU
- **Manual configuration**: Complex networking setup required
- **No discovery**: Can't find other devices on network automatically
- **Wasted compute**: Idle machines not utilized
- **Heterogeneous chaos**: M1, NVIDIA, CPU - all different

### The Hanzo Network Solution

- **Automatic discovery**: UDP broadcast finds peers instantly
- **QR code joining**: Scan code → Join network (no manual config)
- **Device capabilities**: Auto-detect M1, M2, M3, RTX 4090, etc.
- **Intelligent routing**: Route AI workloads to optimal device
- **100% Python/Go parity**: Same features in Python SDK and Go Node

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│         Heterogeneous Device Network                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌────────────┐   ┌────────────┐   ┌────────────┐    │
│   │ MacBook M3 │   │ RTX 4090   │   │ Raspberry  │    │
│   │ 16GB RAM   │   │ 24GB VRAM  │   │ Pi 4 8GB   │    │
│   │ 14.2 TFLOPS│   │ 82.6 TFLOPS│   │ 0.1 TFLOPS │    │
│   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘    │
│         │                 │                 │           │
│         └────────┬────────┴────────┬────────┘           │
│                  │  UDP Discovery  │                    │
│                  │  gRPC Comms     │                    │
│                  │  State Sync     │                    │
│         ┌────────┴────────┬────────┴────────┐           │
│         │                 │                 │           │
│   ┌─────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐    │
│   │ Agent A    │   │ Agent B    │   │ Agent C    │    │
│   │ (Research) │   │ (Code Gen) │   │ (Monitor)  │    │
│   └────────────┘   └────────────┘   └────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘

Routing Logic:
- Heavy LLM inference → RTX 4090 (82.6 TFLOPS)
- Light tasks → M3 MacBook (14.2 TFLOPS)
- Background monitoring → Raspberry Pi (0.1 TFLOPS)
```

## Key Features

### 1. Automatic Peer Discovery

**UDP Broadcast Discovery** (Python & Go):
- Broadcasts node presence on local network
- Discovers other nodes automatically
- Maintains peer health checks
- Supports network interface prioritization

**Manual Discovery** (JSON Config):
- Explicit peer configuration
- Works across VPNs/Tailscale
- No UDP broadcast needed
- QR code scanning (planned)

### 2. Device Capability Detection

**Supported Platforms**:
- **macOS**: M1, M1 Pro, M1 Max, M2, M2 Pro, M2 Max, M3, M3 Pro, M3 Max, M4
- **Linux**: NVIDIA GPUs (RTX 4090, 4080, 4070, 3090, 3080), AMD GPUs
- **Windows**: NVIDIA GPUs, Intel integrated
- **ARM**: Raspberry Pi, Jetson Nano, other ARM devices

**Detected Metrics**:
- **Model**: Device name (MacBook Pro, Linux Box, etc.)
- **Chip**: Processor (Apple M3, NVIDIA RTX 4090, etc.)
- **Memory**: Total RAM in MB
- **FLOPS**: FP32, FP16, INT8 performance in TFLOPS

### 3. Distributed Agent Execution

**Cross-Node State Synchronization**:
- Shared state across all nodes
- Real-time updates via gRPC
- Conflict resolution
- Checkpoint/restore

**Load Balancing**:
- Route to device with optimal capabilities
- Consider memory, TFLOPS, current load
- Fallback if primary device busy
- Parallel execution across multiple devices

### 4. Network Topology Configuration

**QR Code Joining** (Planned):
```python
# Device 1: Generate QR code
network = create_distributed_network(name="my-network")
qr_code = network.generate_join_qr()
print_qr(qr_code)  # Display QR code

# Device 2: Scan and join
network = join_network_from_qr(scan_qr())
# Automatically discovers peers and shares capabilities
```

**JSON Config** (Current):
```json
{
  "peers": {
    "macbook-m3": {
      "address": "192.168.1.10",
      "port": 5681,
      "device_capabilities": {
        "model": "MacBook Pro",
        "chip": "Apple M3 Max",
        "memory": 65536,
        "flops": {
          "fp32": 14.20,
          "fp16": 28.40,
          "int8": 56.80
        }
      }
    },
    "gpu-server": {
      "address": "192.168.1.20",
      "port": 5682,
      "device_capabilities": {
        "model": "GPU Server",
        "chip": "NVIDIA GEFORCE RTX 4090",
        "memory": 131072,
        "flops": {
          "fp32": 82.58,
          "fp16": 165.16,
          "int8": 330.32
        }
      }
    }
  }
}
```

## Installation

### Python SDK

```bash
# Install hanzo-network
pip install hanzo-network

# Or from source
cd ~/work/hanzo/python-sdk/pkg/hanzo-network
pip install -e .
```

### Go Node (Rust/Go Implementation)

```bash
# Clone Lux node (contains Go network implementation)
git clone https://github.com/luxfi/node.git
cd node

# Build node
make build

# Run with distributed network
./build/luxd --network-peer-list="192.168.1.10:5681,192.168.1.20:5682"
```

## Usage Examples

### Python: Automatic Discovery (UDP)

```python
from hanzo_network import create_distributed_network, create_agent

# Create agents
weather_agent = create_agent(
    name="weather",
    description="Get weather information",
    tools=["web_search", "api_call"]
)

math_agent = create_agent(
    name="math",
    description="Solve math problems",
    tools=["calculator", "wolfram_alpha"]
)

# Create distributed network with UDP discovery
network = create_distributed_network(
    agents=[weather_agent, math_agent],
    name="my-network",
    node_id="node-1",
    listen_port=5681,
    broadcast_port=5678  # All nodes use same broadcast port
)

# Start network (discovers peers automatically)
await network.start(wait_for_peers=0)

# Check discovered peers
status = network.get_network_status()
print(f"Discovered {status['peer_count']} peers")

# Execute agent (routes to optimal device automatically)
result = await network.run("What's the weather in San Francisco?", initial_agent=weather_agent)
print(result)
```

### Python: Manual Discovery (JSON Config)

```python
from hanzo_network import create_distributed_network, create_agent

# Create network with manual discovery
network = create_distributed_network(
    agents=[agent1, agent2],
    name="my-network",
    node_id="macbook-m3",  # Must match config file
    discovery_mode="manual",
    config_path="./network_topology.json"
)

# Start network (reads config file)
await network.start()

# Peers are loaded from config file
# Health checks run every 5 seconds
```

### Python: Multi-Node Setup

**Node 1 (MacBook M3)**:
```python
network1 = create_distributed_network(
    agents=[weather_agent],
    node_id="node-1",
    listen_port=5681,
    broadcast_port=5678,
    device_capabilities={
        "model": "MacBook Pro",
        "chip": "Apple M3",
        "memory": 16384,
        "flops": {"fp32": 3.55, "fp16": 7.10, "int8": 14.20}
    }
)
await network1.start()
```

**Node 2 (RTX 4090 Server)**:
```python
network2 = create_distributed_network(
    agents=[llm_agent],
    node_id="node-2",
    listen_port=5682,
    broadcast_port=5678,  # Same broadcast port
    device_capabilities={
        "model": "GPU Server",
        "chip": "NVIDIA GEFORCE RTX 4090",
        "memory": 131072,
        "flops": {"fp32": 82.58, "fp16": 165.16, "int8": 330.32}
    }
)
await network2.start()
```

Nodes automatically discover each other and share agent capabilities.

### Go: Distributed Network

```go
package main

import (
    "context"
    "github.com/luxfi/node/network"
    "github.com/luxfi/node/network/peer"
)

func main() {
    // Create network config
    config := network.Config{
        ListenAddr:    "0.0.0.0:5681",
        BroadcastPort: 5678,
        HealthCheckInterval: time.Second * 5,
    }

    // Create network
    net, err := network.NewNetwork(config)
    if err != nil {
        panic(err)
    }

    // Start peer discovery
    net.Start(context.Background())

    // Get discovered peers
    peers := net.GetPeers()
    for _, peer := range peers {
        fmt.Printf("Peer: %s (%s)\n", peer.ID(), peer.IP())
    }

    // Execute agent on optimal peer
    result, err := net.ExecuteAgent(ctx, "weather", query)
    if err != nil {
        panic(err)
    }

    fmt.Println(result)
}
```

## Device Capability Detection

### Python: Auto-Detection

```python
from hanzo_network.topology import device_capabilities

# Automatically detect current device
caps = device_capabilities()

print(f"Model: {caps.model}")
print(f"Chip: {caps.chip}")
print(f"Memory: {caps.memory}MB")
print(f"FLOPS: {caps.flops}")

# Example output on M3 MacBook:
# Model: MacBook Pro
# Chip: Apple M3
# Memory: 16384MB
# FLOPS: fp32: 3.55 TFLOPS, fp16: 7.10 TFLOPS, int8: 14.20 TFLOPS
```

### Go: Auto-Detection

```go
package main

import (
    "fmt"
    "github.com/luxfi/node/network/peer"
)

func main() {
    // Auto-detect device capabilities
    caps := peer.DetectDeviceCapabilities()

    fmt.Printf("Model: %s\n", caps.Model)
    fmt.Printf("Chip: %s\n", caps.Chip)
    fmt.Printf("Memory: %dMB\n", caps.Memory)
    fmt.Printf("FP32: %.2f TFLOPS\n", caps.Flops.FP32)
}
```

### Supported Devices and FLOPS

| Device | FP32 TFLOPS | FP16 TFLOPS | INT8 TFLOPS | Memory |
|--------|-------------|-------------|-------------|---------|
| **Apple M1** | 2.29 | 4.58 | 9.16 | 8-16GB |
| **Apple M1 Pro** | 5.30 | 10.60 | 21.20 | 16-32GB |
| **Apple M1 Max** | 10.60 | 21.20 | 42.40 | 32-64GB |
| **Apple M2** | 3.55 | 7.10 | 14.20 | 8-24GB |
| **Apple M3** | 3.55 | 7.10 | 14.20 | 8-24GB |
| **Apple M3 Max** | 14.20 | 28.40 | 56.80 | 36-128GB |
| **Apple M4** | 4.26 | 8.52 | 17.04 | 16-32GB |
| **RTX 4090** | 82.58 | 165.16 | 330.32 | 24GB |
| **RTX 4080** | 48.74 | 97.48 | 194.96 | 16GB |
| **RTX 4070** | 29.0 | 58.0 | 116.0 | 12GB |
| **RTX 3090** | 35.6 | 71.2 | 142.4 | 24GB |
| **RTX 3080** | 29.8 | 59.6 | 119.2 | 10GB |

## Intelligent Agent Routing

### Capability-Based Routing

```python
from hanzo_network import create_distributed_network, RoutingStrategy

# Create network with intelligent routing
network = create_distributed_network(
    agents=[light_agent, heavy_agent],
    routing_strategy=RoutingStrategy.CAPABILITY_BASED
)

# Light agent routed to M3 MacBook (3.55 TFLOPS)
result1 = await network.run("Simple math: 2+2", initial_agent=light_agent)

# Heavy agent routed to RTX 4090 (82.58 TFLOPS)
result2 = await network.run("Generate image from prompt", initial_agent=heavy_agent)
```

### Load-Balanced Routing

```python
# Create network with load balancing
network = create_distributed_network(
    agents=[agent1, agent2, agent3],
    routing_strategy=RoutingStrategy.LEAST_LOADED
)

# Requests automatically distributed across devices
# based on current load
for i in range(100):
    result = await network.run(f"Query {i}", initial_agent=agent1)
```

### Custom Routing Logic

```python
def custom_router(agents, state, peers):
    """Route based on device capabilities and task type"""

    # Get task requirements
    requires_gpu = "image" in state.query.lower()
    requires_memory = len(state.context) > 10000

    # Find optimal peer
    if requires_gpu:
        # Route to peer with highest TFLOPS
        best_peer = max(peers, key=lambda p: p.capabilities.flops.fp16)
        return best_peer
    elif requires_memory:
        # Route to peer with most RAM
        best_peer = max(peers, key=lambda p: p.capabilities.memory)
        return best_peer
    else:
        # Use least loaded peer
        best_peer = min(peers, key=lambda p: p.current_load)
        return best_peer

# Use custom router
network = create_distributed_network(
    agents=[agent1, agent2],
    custom_router=custom_router
)
```

## Python/Go Feature Parity

### Feature Comparison

| Feature | Python SDK | Go Node | Status |
|---------|------------|---------|--------|
| **UDP Discovery** | ✅ | ✅ | 100% parity |
| **Manual Discovery** | ✅ | ✅ | 100% parity |
| **Device Capabilities** | ✅ | ✅ | 100% parity |
| **gRPC Communication** | ✅ | ✅ | 100% parity |
| **State Synchronization** | ✅ | ✅ | 100% parity |
| **Load Balancing** | ✅ | ✅ | 100% parity |
| **Health Checks** | ✅ | ✅ | 100% parity |
| **Agent Execution** | ✅ | ✅ | 100% parity |
| **QR Code Joining** | 🔄 Planned | 🔄 Planned | Coming soon |
| **Tailscale VPN** | ✅ | ✅ | 100% parity |

### Interoperability

```python
# Python node can discover and communicate with Go node
# Go node can discover and communicate with Python node
# Both use same gRPC protocol and discovery mechanism

# Example: Python node + Go node
# Python (MacBook):
python_network = create_distributed_network(node_id="python-node")

# Go (GPU Server):
# luxd --network-id=go-node --listen-addr=0.0.0.0:5682

# They discover each other automatically!
# Python can execute agents on Go node and vice versa
```

## QR Code Joining (Planned Feature)

### Vision

Scan QR code to join heterogeneous devices to the same network:

**Device 1 (Host - MacBook)**:
```python
network = create_distributed_network(name="hanzo-home")

# Generate QR code with network info
qr_data = network.generate_join_qr()
# QR contains: network_id, listen_addr, encryption_key, capabilities

print_qr(qr_data)  # Display QR code on screen
```

**Device 2 (Join - Raspberry Pi)**:
```python
# Scan QR code (using camera or manual input)
qr_data = scan_qr()

# Join network automatically
network = join_network_from_qr(qr_data)
# Automatically configures peer discovery, encryption, capabilities

# Now part of the network!
status = network.get_network_status()
print(f"Joined network with {status['peer_count']} peers")
```

**Current Workaround**: Use JSON config files + QR code generator:

```python
import qrcode
import json

# Generate config
config = {
    "peers": {
        "macbook": {
            "address": "192.168.1.10",
            "port": 5681,
            "device_capabilities": {...}
        }
    }
}

# Generate QR code
qr = qrcode.make(json.dumps(config))
qr.save("network_join.png")

# Device 2: Scan QR, save to config.json, join network
```

## Production Deployment

### Multi-Device Setup

**1. Home Office Setup** (3 devices):
```
┌─────────────────────────────────────────────────┐
│         Home Office Network                      │
├─────────────────────────────────────────────────┤
│  MacBook M3 (16GB) → Light tasks + UI           │
│  GPU Server RTX 4090 (24GB) → Heavy LLM         │
│  Raspberry Pi 4 (8GB) → Background monitoring   │
└─────────────────────────────────────────────────┘

# MacBook (coordinator)
network = create_distributed_network(
    node_id="macbook",
    agents=[ui_agent, light_agent],
    listen_port=5681
)

# GPU Server (compute)
network = create_distributed_network(
    node_id="gpu-server",
    agents=[llm_agent, image_agent],
    listen_port=5682
)

# Raspberry Pi (monitor)
network = create_distributed_network(
    node_id="rpi",
    agents=[health_monitor, log_agent],
    listen_port=5683
)
```

**2. Startup Setup** (5+ devices):
```
┌─────────────────────────────────────────────────┐
│      Startup Distributed AI Network             │
├─────────────────────────────────────────────────┤
│  2× MacBook Pro M3 Max → Development            │
│  3× GPU Server RTX 4090 → Production inference  │
│  1× Monitor Server → Metrics + logs             │
└─────────────────────────────────────────────────┘

# Use manual discovery (JSON config) for reliability
# config.json contains all 6 devices
network = create_distributed_network(
    discovery_mode="manual",
    config_path="./production_network.json",
    routing_strategy=RoutingStrategy.LEAST_LOADED
)
```

### Security and VPN

**Tailscale Integration** (Recommended):
```python
# All devices on Tailscale VPN
# Automatic encrypted mesh network
# Access from anywhere securely

network = create_distributed_network(
    node_id="mobile-macbook",
    tailscale=True,  # Use Tailscale IPs
    discovery_mode="manual",
    config_path="./tailscale_network.json"
)

# Peers use Tailscale IPs (100.x.x.x)
# Encrypted by default, zero-config
```

**Manual TLS** (Advanced):
```python
network = create_distributed_network(
    node_id="secure-node",
    tls_cert="/path/to/cert.pem",
    tls_key="/path/to/key.pem",
    tls_ca="/path/to/ca.pem",
    verify_peers=True
)
```

## Monitoring and Debugging

### Network Status

```python
# Get network status
status = network.get_network_status()

print(f"Node ID: {status['node_id']}")
print(f"Peers: {status['peer_count']}")
print(f"Agents: {status['agent_count']}")
print(f"Active tasks: {status['active_tasks']}")

for peer_id, peer in status['peers'].items():
    print(f"\nPeer: {peer_id}")
    print(f"  Address: {peer['address']}")
    print(f"  Healthy: {peer['healthy']}")
    print(f"  Capabilities: {peer['capabilities']}")
    print(f"  Load: {peer['current_load']:.1%}")
```

### Health Checks

```python
# Check peer health
health = await network.check_peer_health("gpu-server")
print(f"Healthy: {health.healthy}")
print(f"Latency: {health.latency_ms}ms")
print(f"Last seen: {health.last_seen}")

# Auto-healing: unhealthy peers removed automatically
# Health checks run every 5 seconds
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Detailed network logs
# - Peer discovery
# - gRPC communication
# - Agent routing
# - State synchronization
```

## Best Practices

### 1. Use UDP Discovery for LAN

```python
# Same local network → UDP discovery (zero config)
network = create_distributed_network(
    broadcast_port=5678,  # All devices use same port
    listen_port=5681      # Unique per device
)
```

### 2. Use Manual Discovery for Production

```python
# Production/VPN → Manual discovery (reliable)
network = create_distributed_network(
    discovery_mode="manual",
    config_path="./production_network.json"
)
```

### 3. Route by Capability

```python
# Let Hanzo Network choose optimal device
# Based on TFLOPS, memory, current load
network = create_distributed_network(
    routing_strategy=RoutingStrategy.CAPABILITY_BASED
)
```

### 4. Health Check Intervals

```python
# Production: longer intervals (less overhead)
network = create_distributed_network(
    health_check_interval=30  # 30 seconds
)

# Development: shorter intervals (faster detection)
network = create_distributed_network(
    health_check_interval=5  # 5 seconds
)
```

## Troubleshooting

### Peers Not Discovering

**Problem**: Devices on same network not finding each other

**Solutions**:
1. Check broadcast port matches: `broadcast_port=5678`
2. Check firewall allows UDP: `sudo ufw allow 5678/udp`
3. Check network interface: `--network-interface=en0`
4. Try manual discovery instead

### High Latency

**Problem**: Slow agent execution across devices

**Solutions**:
1. Check peer health: `network.check_peer_health()`
2. Use local agents for low-latency tasks
3. Enable Tailscale for VPN optimization
4. Reduce health check frequency

### Device Not Detected

**Problem**: Device capabilities showing as "Unknown"

**Solutions**:
1. Update device detection: `pip install -U hanzo-network`
2. Manually specify capabilities: `device_capabilities={...}`
3. Check system profiler: `system_profiler SPHardwareDataType`
4. Submit PR to add your device

## Related Skills

- **python-sdk.md** - Python SDK for Hanzo AI
- **go-sdk.md** - Go SDK for backend services
- **hanzo-node.md** - Local AI inference infrastructure
- **hanzo-mcp.md** - Model Context Protocol integration
- **hanzo-engine.md** - Native Rust inference engine

## Additional Resources

- **GitHub**: https://github.com/hanzoai/network
- **Python Package**: https://pypi.org/project/hanzo-network/
- **Lux Node**: https://github.com/luxfi/node (Go implementation)
- **Hanzo AI**: https://hanzo.ai

---

**Remember**: Hanzo Network enables **distributed AI agent execution** across heterogeneous devices with automatic discovery, device capability detection, and intelligent routing - perfect for leveraging all your compute resources (M1 MacBook + RTX 4090 + Raspberry Pi) seamlessly.
