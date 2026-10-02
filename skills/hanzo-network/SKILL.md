---
name: hanzo-network
description: "Distributed agent network orchestration. Create multi-agent swarms with gRPC, ring topology, local/remote LLM providers, and shard distribution."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-network",
              "label": "Install Hanzo Network (pip)",
            },
          ],
      },
  }
---

# Hanzo Network — Distributed Agent Orchestration

`pip install hanzo-network`

Build distributed multi-agent networks with gRPC, ring topology, local/remote LLM inference, and model shard distribution.

## Quick Start

```python
from hanzo_network import create_agent, create_network, Tool

# Create agents with tools
researcher = create_agent(
    name="Researcher",
    instructions="Research topics thoroughly",
    tools=[search_tool]
)

writer = create_agent(
    name="Writer",
    instructions="Write clear content",
    tools=[write_tool]
)

# Create a network
network = create_network(agents=[researcher, writer])
result = await network.run("Write about quantum computing")
```

## Distributed Networks

```python
from hanzo_network import create_distributed_network

# gRPC-based distributed agent network
network = create_distributed_network(
    agents=[agent1, agent2, agent3],
    topology="ring"  # Ring topology with memory-weighted partitioning
)

# Agents communicate via gRPC + protobuf
# Discovery via UDP/Tailscale
result = await network.run("Complex multi-step task")
```

## Local LLM Providers

```python
from hanzo_network import create_local_agent, check_local_llm_status

# Check available local inference engines
status = check_local_llm_status()

# Create agent with local LLM (MLX, Tinygrad, Ollama)
agent = create_local_agent(
    name="Local Agent",
    provider="ollama",  # or "mlx", "tinygrad"
    model="llama3"
)
```

## LLM Providers

```python
from hanzo_network import HanzoNetProvider, LocalLLMProvider, OllamaProvider, MLXProvider

# Hanzo cloud provider
provider = HanzoNetProvider(api_key="...")

# Local Ollama
provider = OllamaProvider(model="llama3")

# Apple Silicon MLX
provider = MLXProvider(model="mlx-community/Meta-Llama-3-8B")
```

## Agent Routing

```python
from hanzo_network import Router

router = Router(agents=[researcher, writer, coder])

# Semantic routing — automatically selects best agent
result = await router.route("Write a Python script for data analysis")
```

## Shard Distribution

```python
# Distribute model shards across network nodes
# Ring topology with memory-weighted partitioning
network = create_distributed_network(
    agents=[...],
    shard_strategy="memory_weighted"
)
```
