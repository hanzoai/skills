---
name: hanzo-consensus
description: "Metastable consensus protocol for multi-agent agreement. Two-phase finality with k-peer sampling for distributed AI decision making."
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
              "package": "hanzo-consensus",
              "label": "Install Hanzo Consensus (pip)",
            },
          ],
      },
  }
---

# Hanzo Consensus — Multi-Agent Agreement

`pip install hanzo-consensus`

Metastable consensus protocol for distributed multi-agent decision making. Two-phase finality with configurable k-peer sampling.

## Quick Start

```python
from hanzo_consensus import Consensus, run

# Simple consensus among participants
result = run(
    participants=["agent-1", "agent-2", "agent-3"],
    proposal="Should we deploy to production?",
    rounds=10
)
print(result.decision)  # True/False
print(result.confidence)  # 0.0-1.0
```

## Async Usage

```python
from hanzo_consensus import Consensus

consensus = Consensus(
    rounds=10,
    k=3,           # Sample size per round
    alpha=0.8,     # Agreement threshold
    beta_1=15,     # Preference threshold (Phase I)
    beta_2=20      # Decision threshold (Phase II)
)

result = await consensus.run(participants, proposal)
```

## MCP Mesh Consensus

```python
from hanzo_consensus import MCPMesh, MCPAgent, run_mcp_consensus

# Create MCP-based consensus mesh
mesh = MCPMesh(agents=[
    MCPAgent("analyst", mcp_server="..."),
    MCPAgent("reviewer", mcp_server="..."),
    MCPAgent("approver", mcp_server="...")
])

result = await run_mcp_consensus(mesh, "Evaluate code quality of PR #42")
```

## Protocol Parameters

| Parameter | Description                  | Default |
| --------- | ---------------------------- | ------- |
| `rounds`  | Number of sampling rounds    | 10      |
| `k`       | Peers sampled per round      | 3       |
| `alpha`   | Agreement threshold (0-1)    | 0.8     |
| `beta_1`  | Phase I preference threshold | 15      |
| `beta_2`  | Phase II decision threshold  | 20      |

## Protocol Design

Two-phase finality based on [Lux Consensus](https://github.com/luxfi/consensus):

1. **Sampling Phase**: Each round, sample k random peers and query their preference
2. **Finality Phase**: After sufficient consecutive agreements, finalize decision

Pure Python — no external dependencies.
