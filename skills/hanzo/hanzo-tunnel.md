# Hanzo Tunnel - Cloud Tunnel Client for Remote Agent Control

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-bot.md`, `hanzo/hanzo-agent.md`, `hanzo/hanzo-dev.md`

## Overview

Hanzo Tunnel is a **Rust library and Python agent bridge** that connects local Hanzo app instances (dev machines, bots, nodes) to the cloud control plane at `app.hanzo.bot`. It provides WebSocket-based registration, remote command execution, terminal streaming, session checkpoint/restore (via S3), and local service exposure (ngrok-like). The Rust crate (`hanzo-tunnel`) is the protocol library; the Python `agent-bridge.py` is the standalone agent process.

### Why Hanzo Tunnel?

- **Remote control**: Execute commands on local machines from the web UI at app.hanzo.bot
- **Session migration**: Checkpoint session state to S3, restore on another machine or cloud agent
- **Terminal streaming**: Open/close/resize interactive PTY sessions remotely
- **Service exposure**: Expose local HTTP/WebSocket/TCP services through the tunnel (ngrok-like)
- **Bot gateway integration**: Register as a node in the Hanzo bot gateway's NodeRegistry
- **Swarm mode**: Launch N parallel agent instances for testing

### Tech Stack

- **Rust crate**: `hanzo-tunnel` v0.1.1 (library, no binary)
- **Python agent**: `agent-bridge.py` (standalone bridge process)
- **Docker image**: `ghcr.io/hanzoai/cloud-agent:latest` (Python agent in Alpine)
- **Protocol**: JSON frames over WebSocket (register/registered/command/response/event/ping/pong)
- **Dependencies (Rust)**: tokio, tokio-tungstenite, serde, futures, uuid, chrono, tracing, thiserror
- **Dependencies (Python)**: websockets, boto3

### OSS Base

Repo: `hanzoai/tunnel`.

## When to use

- Connecting a local dev machine to the Hanzo cloud control plane
- Running remote commands (chat, exec, terminal) on local machines from a web UI
- Checkpointing and migrating dev sessions between machines
- Exposing local services through a public tunnel URL
- Integrating with the Hanzo bot gateway as a node
- Building custom Hanzo app integrations that need cloud connectivity

## Hard requirements

1. **WebSocket connectivity** to `wss://app.hanzo.bot/v1/tunnel` or `wss://api.hanzo.ai/v1/relay`
2. **Auth token**: Hanzo IAM JWT or API key (for Rust library)
3. **S3 credentials** (optional, for session checkpoint/restore): `S3_ACCESS_KEY`, `S3_SECRET_KEY`
4. **Claude CLI** (optional, for `chat.send` command in Python agent)

## Quick reference

| Item | Value |
|------|-------|
| Rust crate | `hanzo-tunnel` v0.1.1 |
| Relay URL | `wss://api.hanzo.ai/v1/relay` |
| Agent bridge | `agent-bridge.py` |
| Agent image | `ghcr.io/hanzoai/cloud-agent:latest` |
| Bot gateway | `ws://bot-gateway/v1/tunnel` (in-cluster) |
| S3 endpoint | `https://s3.hanzo.ai` |
| S3 bucket | `hanzo-sessions` |
| K8s namespace | `hanzo` |
| Repo | `github.com/hanzoai/tunnel` |

## One-file quickstart

### Rust Library

```rust
use hanzo_tunnel::{connect, TunnelConfig, AppKind};

#[tokio::main]
async fn main() -> Result<(), hanzo_tunnel::TunnelError> {
    let conn = connect(TunnelConfig {
        relay_url: "wss://api.hanzo.ai/v1/relay".into(),
        auth_token: "hk-abc123".into(),
        app_kind: AppKind::Dev,
        display_name: "z-macbook".into(),
        capabilities: vec!["chat".into(), "exec".into()],
        ..Default::default()
    }).await?;

    println!("instance_id = {}", conn.instance_id);

    // Receive commands from cloud
    while let Some(frame) = conn.recv_command().await {
        // Handle commands...
    }
    Ok(())
}
```

### Python Agent Bridge

```bash
# Run locally
TUNNEL_URL=wss://app.hanzo.bot/v1/tunnel python3 agent-bridge.py

# With S3 session support
S3_ACCESS_KEY=xxx S3_SECRET_KEY=yyy python3 agent-bridge.py

# Docker
docker run -d \
  -e TUNNEL_URL=wss://app.hanzo.bot/v1/tunnel \
  -e APP_KIND=cloud \
  ghcr.io/hanzoai/cloud-agent:latest
```

### Swarm Mode

```bash
# Launch 4 parallel agents
./swarm.sh 4 wss://app.hanzo.bot/v1/tunnel
```

## Core Concepts

### Wire Protocol

All communication is via JSON frames over WebSocket:

```
Frame types:
  register      -> Instance to Cloud: register this instance
  registered    <- Cloud to Instance: registration confirmed (includes session_url)
  command       <- Cloud to Instance: execute a command
  response      -> Instance to Cloud: command result
  event         -> Instance to Cloud: streaming output, state changes
  ping/pong     <- Bidirectional heartbeat
```

### App Kinds

```rust
enum AppKind {
    Dev,        // Local dev machine
    Node,       // Server/infrastructure node
    Desktop,    // Desktop app
    Bot,        // Bot instance
    Extension,  // Browser/IDE extension
}
```

### Commands (Python Agent)

| Command | Description |
|---------|-------------|
| `chat.send` | Send message to Claude CLI, return response |
| `exec.run` | Execute shell command (30s timeout) |
| `status.get` | Return instance status, session stats |
| `session.checkpoint` | Save session state + git workspace to S3 |
| `session.restore` | Restore session from S3 checkpoint URL |
| `session.list` | List available checkpoints in S3 |

### Commands (Rust Dispatcher)

| Command | Description |
|---------|-------------|
| `terminal.open` | Spawn interactive PTY session |
| `terminal.input` | Send input to terminal session |
| `terminal.close` | Close terminal session |
| `terminal.resize` | Resize terminal (cols/rows) |
| `terminal.list` | List active terminal sessions |
| `system.info` | Return OS, arch, hostname, shell, user |
| `system.run` | Execute shell command with timeout |
| `dev.launch` | Launch dev/hanzo-dev/codex session |
| `dev.status` | Check running dev processes |

### Session Checkpoint/Restore

Sessions can be migrated between machines:

1. `session.checkpoint` captures conversation history, command log, scrollback, and git workspace state
2. Checkpoint uploaded to S3 (`hanzo-sessions` bucket)
3. `session.restore` downloads checkpoint, restores session state, optionally clones repo + applies diff

### Service Exposure (ngrok-like)

```rust
use hanzo_tunnel::expose::{expose, ExposedService, ExposedProtocol};

expose(&tx, &ExposedService {
    name: "api".into(),
    local_addr: "127.0.0.1:8080".into(),
    protocol: ExposedProtocol::Http,
    subdomain: Some("my-api".into()),
}).await?;
```

### Bot Gateway Integration

The `gateway` module implements the Hanzo bot gateway's native protocol (challenge/connect/hello-ok handshake). This lets tunnel clients register as nodes in the gateway's NodeRegistry:

```rust
use hanzo_tunnel::gateway::{connect_gateway, GatewayConfig};

let conn = connect_gateway(GatewayConfig {
    url: "ws://127.0.0.1:18789".into(),
    auth_token: "device-token".into(),
    client_id: "my-node".into(),
    display_name: "Dev Machine".into(),
    ..Default::default()
}).await?;

// Handle node.invoke requests
while let Some(req) = conn.recv_invoke().await {
    // Process command, send result back
    conn.send_invoke_result(&req.id, true, Some(payload), None).await?;
}
```

### Architecture

```
+-------------------+          +---------------------+
|  Local Machine    |          |  Cloud              |
|                   |          |                     |
|  agent-bridge.py  |--WSS--->|  app.hanzo.bot      |
|  or Rust client   |         |  /v1/tunnel         |
|                   |         |                     |
|  Terminal PTY     |         |  Web UI             |
|  Claude CLI       |         |  (command dispatch) |
|  Local services   |         |                     |
+-------------------+          +---------------------+
                                        |
                               +--------+--------+
                               |  Bot Gateway    |
                               |  NodeRegistry   |
                               +--------+--------+
                                        |
                               +--------+--------+
                               |  Cloud Agents   |
                               |  (K8s pods)     |
                               +-----------------+
```

### Repository Structure

```
src/
  lib.rs          # TunnelConfig, TunnelConnection, connect()
  protocol.rs     # Wire protocol types (Frame, AppKind, payloads)
  transport.rs    # WebSocket transport with reconnection
  gateway.rs      # Bot gateway protocol adapter
  commands.rs     # Node command dispatcher
  terminal.rs     # Interactive PTY session management
  expose.rs       # Local service exposure (ngrok-like)
  registry.rs     # Instance registry types
  discovery.rs    # mDNS discovery (optional feature)
  auth.rs         # Auth token handling
agent-bridge.py   # Python standalone agent bridge
swarm.sh          # Launch N parallel agents
migrate.py        # Session migration utility
test-migration.py # Migration tests
test-swarm.py     # Swarm tests
k8s/
  cloud-agents.yaml   # K8s Deployment (2 replicas, emptyDir workspace)
  s3-credentials.yaml # S3 secret template
Dockerfile        # Python Alpine image for cloud agent
Cargo.toml        # Rust crate definition
```

### K8s Cloud Agents

Cloud agents run as K8s pods that accept migrated sessions:

```yaml
# 2 replicas, ghcr.io/hanzoai/cloud-agent:latest
# Connects to ws://bot-gateway/v1/tunnel
# S3 credentials from hanzo-s3 secret
# ANTHROPIC_API_KEY from bot-secrets
# 250m-1 CPU, 512Mi-2Gi memory, 5Gi workspace
```

### Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `reconnect` | yes | Automatic reconnection with exponential backoff |
| `mdns` | no | mDNS service discovery |
| `tls-rustls` | no | TLS via rustls (alternative to native-tls) |

## Build and Test

```bash
# Build
cargo build

# Run tests
cargo test

# Build Docker image (Python agent)
docker build -t ghcr.io/hanzoai/cloud-agent:latest .
```

## Environment Variables (Python Agent)

```bash
TUNNEL_URL=wss://app.hanzo.bot/v1/tunnel   # Cloud relay endpoint
INSTANCE_ID=hostname-pid                    # Auto-generated if not set
APP_KIND=dev                                # dev, cloud, node, bot
WORKSPACE_DIR=/workspace                    # Working directory
S3_ENDPOINT_URL=https://s3.hanzo.ai         # S3 endpoint
S3_ACCESS_KEY=                              # Required for session commands
S3_SECRET_KEY=                              # Required for session commands
S3_BUCKET=hanzo-sessions                    # Checkpoint bucket
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "claude CLI not found" | Claude Code not installed | `npm i -g @anthropic-ai/claude-code` |
| S3 commands fail | Missing credentials | Set `S3_ACCESS_KEY` and `S3_SECRET_KEY` |
| WebSocket connect timeout | Relay unreachable | Check `TUNNEL_URL`, verify network |
| Gateway auth error | Invalid token | Check `auth_token` or device password |
| Session restore git fail | Repo already exists | Clone skipped if `.git/` exists |

## Related Skills

- `hanzo/hanzo-bot.md` - Bot gateway and NodeRegistry
- `hanzo/hanzo-agent.md` - Multi-agent SDK
- `hanzo/hanzo-dev.md` - Dev CLI tool
- `hanzo/hanzo-operative.md` - Computer use agent

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: tunnel, websocket, remote-control, agent, session-migration, ngrok
**Prerequisites**: Rust/Python, WebSocket basics, S3 (optional)
