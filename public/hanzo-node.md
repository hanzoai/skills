---
name: hanzo-node
description: "Install and manage Hanzo AI compute nodes. Cross-platform installer for the Rust-based hanzo-node binary with version management."
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
              "package": "hanzo-node",
              "label": "Install Hanzo Node (pip)",
            },
          ],
      },
  }
---

# Hanzo Node — AI Compute Node

`pip install hanzo-node`

Cross-platform installer and manager for the Hanzo AI compute node (Rust binary). Nodes participate in the distributed compute network for AI inference, task scheduling, and blockchain integration.

## Quick Start

```bash
# Install the node binary
hanzo-node install

# Check status
hanzo-node status

# Run the node
hanzo-node
```

## Installation Management

```bash
# Install latest version
hanzo-node install

# Install specific version
hanzo-node install --version=0.5.0

# Force reinstall
hanzo-node install --force

# Upgrade to latest
hanzo-node upgrade

# Uninstall
hanzo-node uninstall
```

## Python API

```python
from hanzo_node import install, uninstall, is_installed, get_binary_path, get_installed_version

# Check if installed
if not is_installed():
    install()

# Get info
print(get_binary_path())       # ~/.hanzo/bin/hanzo-node
print(get_installed_version()) # 0.5.0
```

## Node Capabilities

- **Task Scheduling**: Define and schedule compute tasks
- **Code Generation**: Automated code generation pipelines
- **Crypto Support**: Wallet management, transaction signing
- **Distributed Compute**: Join the Hanzo compute mesh
- **Auto-detection**: Platform-specific binary (macOS, Linux, Windows)

## Binary Location

```
~/.hanzo/bin/hanzo-node
```
