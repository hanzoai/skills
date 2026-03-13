# Hanzo Relay - Cloud Tunnel for Hanzo Apps

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-bot.md`, `hanzo/hanzo-code.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Relay (crate name: `hanzo-relay`, internally `hanzo-tunnel`) is a **Rust library** that connects any local Hanzo app (dev, node, desktop, bot, extension) to the cloud relay at `api.hanzo.ai` for remote management and multiplayer. WebSocket-based with automatic reconnection, JSON wire protocol, auth via JWT or API key, and optional mDNS LAN discovery. Think ngrok built into every Hanzo app.

### Why Hanzo Relay?

- **One crate, all apps**: Same tunnel for dev agents, nodes, desktop apps, bots, browser extensions
- **Register + stream**: Apps register with the cloud and stream events (chat deltas, exec output)
- **Remote commands**: Cloud sends commands (chat messages, config changes, approvals) back to instances
- **Service exposure**: Expose local HTTP/WS/TCP ports through the tunnel (like ngrok)
- **Auto-reconnect**: Exponential backoff with configurable limits
- **LAN discovery**: Optional mDNS for local network peer discovery (feature-gated)
- **Auth**: JWT from hanzo.id or API keys (`sk-hanzo-...`)

### Tech Stack

- **Language**: Rust (edition 2021)
- **Crate**: `hanzo-relay` v0.1.0
- **Transport**: tokio-tungstenite (WebSocket with native-tls)
- **Async**: Tokio (rt, macros, sync, time, net)
- **Serialization**: serde + serde_json
- **Error handling**: thiserror
- **Logging**: tracing
- **Optional**: mdns-sd (LAN discovery), rustls (TLS alternative)

Repo: `github.com/hanzoai/relay`

## When to use

- Connecting a Hanzo dev agent to cloud management (app.hanzo.bot)
- Building apps that need remote control from the Hanzo dashboard
- Exposing local development servers through cloud tunnel
- Peer discovery between Hanzo apps on a LAN

## Quick reference

| Item | Value |
|------|-------|
| Crate | `hanzo-relay` |
| Repo | `github.com/hanzoai/relay` |
| Version | 0.1.0 |
| License | MIT |
| Default relay | `wss://api.hanzo.ai/v1/relay` |
| Default features | `reconnect` |

## Usage

### Connect to Cloud

```rust
use hanzo_tunnel::{connect, TunnelConfig, AppKind};

let conn = connect(TunnelConfig {
    relay_url: "wss://api.hanzo.ai/v1/relay".into(),
    auth_token: "sk-hanzo-...".into(),
    app_kind: AppKind::Dev,
    display_name: "z-macbook".into(),
    capabilities: vec!["chat".into(), "exec".into()],
    ..Default::default()
}).await?;

// Send events to cloud
conn.send_event("chat.delta", serde_json::json!({"text": "hello"})).await?;

// Receive commands from cloud
while let Some(cmd) = conn.recv_command().await {
    println!("got: {} {}", cmd.method, cmd.params);
    conn.respond(cmd.id, true, None).await?;
}
```

### Connect and Wait for Registration

```rust
use hanzo_tunnel::connect_and_register;

let conn = connect_and_register(config).await?;
println!("Session URL: {:?}", conn.session_url);
// e.g., https://app.hanzo.bot/i/abc-123
```

### Expose Local Services

```rust
use hanzo_tunnel::expose::{expose, ExposedService, ExposeProtocol};

expose(&conn.event_sender(), ExposedService {
    name: "app-server".into(),
    local_addr: "127.0.0.1:3000".into(),
    protocol: ExposeProtocol::Http,
    subdomain: Some("my-dev".into()),
}).await?;
```

## Wire Protocol

All frames are JSON over WebSocket text messages, tagged by `type`:

| Direction | Frame | Purpose |
|-----------|-------|---------|
| Instance -> Cloud | `register` | Register app with capabilities and metadata |
| Cloud -> Instance | `registered` | Acknowledge with session_url |
| Instance -> Cloud | `event` | Stream data (chat.delta, exec.output, etc.) |
| Cloud -> Instance | `command` | Send command (chat.send, config.update, etc.) |
| Instance -> Cloud | `response` | Reply to a command (ok/error + data) |
| Bidirectional | `ping` / `pong` | Keep-alive heartbeat |

### App Kinds

```rust
pub enum AppKind {
    Dev,        // Development agent
    Node,       // Blockchain/AI node
    Desktop,    // Desktop application
    Bot,        // Bot instance
    Extension,  // Browser/IDE extension
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `reconnect` | Yes | Auto-reconnect with exponential backoff |
| `mdns` | No | LAN discovery via mDNS (`_hanzo._tcp.local.`) |
| `tls-rustls` | No | Use rustls instead of native-tls |

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | TunnelConfig, TunnelConnection, connect(), connect_and_register() |
| `src/protocol.rs` | Frame enum, all payload types, AppKind |
| `src/transport.rs` | WebSocket transport with reconnection loop |
| `src/auth.rs` | AuthToken (JWT vs API key auto-detection) |
| `src/expose.rs` | Service exposure (HTTP/WS/TCP tunneling) |
| `src/registry.rs` | Instance type, InvokeParams, InvokeResult |
| `src/discovery.rs` | mDNS advertisement and discovery |

## Auth

```rust
pub enum AuthToken {
    Jwt { token: String },      // JWT from hanzo.id
    ApiKey { key: String },     // sk-hanzo-... API key
}

// Auto-detects: 2+ dots = JWT, otherwise API key
let token = AuthToken::from_string("sk-hanzo-abc123");
```

## Transport Config

```rust
pub struct TransportConfig {
    pub url: String,                    // wss://api.hanzo.ai/v1/relay
    pub auth: AuthToken,
    pub initial_backoff: Duration,      // 1s
    pub max_backoff: Duration,          // 60s
    pub heartbeat_interval: Duration,   // 30s
    pub connect_timeout: Duration,      // 10s
}
```

## TunnelConnection API

```rust
conn.send_event(event, data).await?;       // Stream event to cloud
conn.recv_command().await;                   // Next command (None = closed)
conn.respond(id, ok, data).await?;          // Reply to command
conn.respond_error(id, msg).await?;         // Reply with error
conn.event_sender();                         // Clone sender for spawned tasks
conn.shutdown();                             // Graceful disconnect
conn.is_connected();                         // Check liveness
```

## Related Skills

- `hanzo/hanzo-bot.md` - Bot framework (uses relay for cloud management)
- `hanzo/hanzo-code.md` - Dev agent (registers via relay)
- `hanzo/hanzo-extension.md` - Browser/IDE extensions (tunnel back to cloud)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: tunnel, websocket, remote-management, cloud, rust, ngrok
**Prerequisites**: Rust toolchain, Tokio runtime
