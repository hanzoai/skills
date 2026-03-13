# Hanzo Desktop - Cross-Platform AI Agent Builder

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-node.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-ui.md`

## Overview

Hanzo Desktop is a **cross-platform desktop application** for creating and managing AI agents without code. Built with **Tauri** (Rust backend + React frontend) and managed as an **NX monorepo**, it provides a visual interface for multi-agent orchestration, MCP tool integration, and hybrid local/cloud AI inference.

The application embeds the **Hanzo Node** binary (from `github.com/hanzoai/node`) as a sidecar process, which provides the actual AI agent runtime, P2P networking, and inference capabilities.

Fork lineage: Originally based on `dcSpark/shinkai-local-ai-agents` (some `shinkai-*` lib names remain in the codebase).

### OSS Info

Repo: `github.com/hanzoai/desktop`. Package: `@hanzo/source` v1.1.34. Desktop app: `hanzo-desktop` v1.1.28. Apache License.

## When to use

- Building AI agents without writing code (drag-and-drop visual interface)
- Running local AI inference with privacy (Ollama integration)
- Orchestrating multiple agents that collaborate on complex workflows
- Connecting AI agents to external tools via MCP
- Cross-platform deployment (Windows, macOS, Linux)

## Hard requirements

1. **Node.js >= 20** with npm 10.x
2. **Rust toolchain** (for Tauri build)
3. **NX** (workspace orchestration, included as devDependency)
4. Side binaries: Hanzo Node v1.1.20 + Ollama v0.12.3 (downloaded via CI script)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/desktop` |
| Package name | `@hanzo/source` v1.1.34 |
| Desktop app | `hanzo-desktop` |
| Framework | Tauri (Rust + React) |
| Build system | NX + Vite |
| Frontend | React 19, TypeScript, Tailwind CSS 4 |
| State | Zustand (UI) + React Query (server) |
| Testing | Vitest + React Testing Library |
| Platforms | Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+) |
| Node.js | >= 20 |
| License | Apache-2.0 |

## Project structure

```
desktop/
├── package.json            # @hanzo/source v1.1.34 (workspace root)
├── nx.json                 # NX workspace config
├── tsconfig.base.json
├── eslint.config.mjs
├── mise.toml               # Runtime version manager
├── apps/
│   └── hanzo-desktop/      # Main Tauri desktop application
│       └── src-tauri/      # Rust backend + Tauri config
├── libs/
│   ├── hanzo-message-ts/   # Message protocols for Hanzo Node communication
│   ├── hanzo-node-state/   # React Query state management for node data
│   ├── hanzo-ui/           # Reusable React component library
│   ├── hanzo-artifacts/    # Styled UI primitives (Radix + Tailwind)
│   ├── hanzo-i18n/         # i18next internationalization (8 languages)
│   └── shinkai-node-state/ # Legacy state lib (upstream naming)
├── ci-scripts/             # Build, binary download, Ollama repo generation
├── scripts/                # Development utilities
├── tools/                  # NX plugins and dev tools
├── patches/                # npm patch-package overrides
├── docs/                   # Documentation
└── assets/                 # App icons and images
```

## Key features (verified from repo)

- **No-code agent builder**: Visual interface for creating specialized AI agents
- **Multi-agent orchestration**: Deploy teams of agents that collaborate and share context
- **MCP support**: Universal protocol compatibility (Claude, Cursor ecosystem)
- **Hybrid deployment**: Local inference (Ollama), cloud models, or both
- **Cross-platform**: Windows, macOS (Apple Silicon), Linux
- **Crypto-native**: Built-in support for decentralized payments and DeFi interactions
- **Internationalization**: 8 languages (EN, ES, ZH, ZH-HK, KO, JA, ID, TR)
- **Composio integration**: Pre-built app templates via generated repository

## Technology stack

| Layer | Technology |
|-------|-----------|
| Desktop shell | Tauri 2.x (Rust + WebView) |
| Frontend | React 19, TypeScript, Vite |
| Styling | Tailwind CSS 4, Radix UI, Shadcn/ui patterns |
| State management | Zustand (UI state), React Query (server state) |
| Build orchestration | NX 22.3 |
| AI runtime | Hanzo Node (embedded Rust sidecar) |
| Local inference | Ollama (embedded sidecar) |
| Testing | Vitest, React Testing Library, Playwright |
| i18n | i18next with AI-generated translations |
| Forms | React Hook Form + Zod |
| Charts | Recharts |
| Markdown | Streamdown |

## Development

### Setup

```bash
git clone https://github.com/hanzoai/desktop.git
cd desktop
nvm use
npm ci
```

### Download side binaries

The app embeds Hanzo Node and Ollama as sidecar binaries:

```bash
# macOS Apple Silicon
ARCH="aarch64-apple-darwin" \
HANZO_NODE_VERSION="v1.1.20" \
OLLAMA_VERSION="v0.12.3" \
npx ts-node ./ci-scripts/download-side-binaries.ts

# Linux
ARCH="x86_64-unknown-linux-gnu" \
HANZO_NODE_VERSION="v1.1.20" \
OLLAMA_VERSION="v0.12.3" \
npx ts-node ./ci-scripts/download-side-binaries.ts

# Windows
$ENV:ARCH="x86_64-pc-windows-msvc"
$ENV:HANZO_NODE_VERSION="v1.1.20"
$ENV:OLLAMA_VERSION="v0.12.3"
npx ts-node ./ci-scripts/download-side-binaries.ts
```

### Run development server

```bash
npx nx serve:tauri hanzo-desktop
```

### Build

```bash
npx nx build hanzo-desktop

# With increased memory for large builds
NODE_OPTIONS="--max_old_space_size=8192" npx nx build hanzo-desktop
```

### Test

```bash
npx nx test [project-name]
npx nx run-many --target=test
npx nx lint [project-name]
```

## Shared libraries

| Library | Purpose |
|---------|---------|
| `hanzo-message-ts` | TypeScript message protocols for communicating with Hanzo Node |
| `hanzo-node-state` | React Query hooks for node data (models, agents, jobs) |
| `hanzo-ui` | Reusable React components and design system |
| `hanzo-artifacts` | Styled UI primitives built on Radix and Tailwind CSS |
| `hanzo-i18n` | Internationalization utilities with AI-generated translations |

## Tauri plugins used

The desktop app uses these Tauri 2.x plugins (from package.json):

- `@tauri-apps/plugin-dialog` - Native file/folder dialogs
- `@tauri-apps/plugin-fs` - File system access
- `@tauri-apps/plugin-http` - HTTP client
- `@tauri-apps/plugin-log` - Logging
- `@tauri-apps/plugin-notification` - System notifications
- `@tauri-apps/plugin-opener` - Open URLs/files
- `@tauri-apps/plugin-os` - OS information
- `@tauri-apps/plugin-process` - Process management
- `@tauri-apps/plugin-shell` - Shell command execution
- `@tauri-apps/plugin-updater` - Auto-update

## Related Skills

- `hanzo/hanzo-node.md` - The Rust AI node that powers the desktop app backend
- `hanzo/hanzo-mcp.md` - MCP protocol integration
- `hanzo/hanzo-ui.md` - Component library (separate from the desktop-internal hanzo-ui lib)
- `hanzo/python-sdk.md` - Python SDK for programmatic access

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: desktop, tauri, react, agents, ai, cross-platform
**Prerequisites**: Node.js 20+, Rust toolchain, NX basics
