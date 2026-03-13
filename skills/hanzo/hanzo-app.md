# Hanzo App - AI-Powered Developer Platform and Desktop Assistant

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-chat.md`, `hanzo/hanzo-llm-gateway.md`, `hanzo/hanzo-studio.md`

## Overview

Hanzo App is an **AI-powered developer platform** that serves as both a web-based development environment (hanzo.app) and a macOS desktop assistant. Built with Next.js 15 and React 19 for the web frontend, with a native macOS layer via React Native and Xcode/CocoaPods. The web app provides a full IDE-like experience with AI chat, code editing (Monaco), project management, deployments, agents, and a template gallery. The desktop app adds system-level features like app launching, clipboard management, window management, and instant AI access via the Tab key.

### Why Hanzo App?

- **Browser-based IDE**: Monaco editor, file explorer, project management, deploy from browser
- **AI-first**: Integrated chat panel, model selector (Zen models), agents, code generation
- **Desktop assistant**: macOS native app with global hotkey (Tab), app search, clipboard manager
- **Full platform**: Auth, billing, settings, admin, enterprise features, community gallery
- **Template system**: Pre-built project templates with one-click deploy

### Tech Stack

- **Web Frontend**: Next.js 15.5 + React 19 + TypeScript 5.9
- **UI**: Radix UI primitives + Tailwind CSS 4 + Lucide icons
- **Editor**: Monaco Editor (via @monaco-editor/react)
- **Data**: TanStack React Query, better-sqlite3 (local), jose (JWT)
- **Bundling**: esbuild-wasm (in-browser bundling)
- **macOS Native**: React Native + Xcode + CocoaPods
- **Tooling**: Mise (bun, ruby, cocoapods), bun as JS runtime
- **Install**: `brew install --cask hanzo` (desktop) or visit hanzo.app (web)

### OSS Base

Repo: `hanzoai/app`. MIT License.

## When to use

- Building or deploying AI-powered applications through a web IDE
- Using AI chat and code generation alongside a code editor
- Managing projects, deployments, and templates from a unified dashboard
- Running a macOS productivity assistant with AI capabilities
- Prototyping with in-browser code execution (esbuild-wasm, QuickJS)

## Hard requirements

1. **Node.js / bun** for the web app
2. **macOS + Xcode + CocoaPods** for the native desktop app
3. **Mise** for development environment management (bun, ruby)

## Quick reference

| Item | Value |
|------|-------|
| Web URL | `https://hanzo.app` |
| Desktop Install | `brew install --cask hanzo` |
| Repo | `github.com/hanzoai/app` |
| Branch | `main` |
| Framework | Next.js 15.5 + React 19 |
| License | MIT |
| Version | 1.42.0 |

## One-file quickstart

### Web development

```bash
# Clone and install
git clone https://github.com/hanzoai/app.git
cd app
bun install

# Run development server
bun dev

# Build for production
bun run build
```

### macOS native development

```bash
# Requires Mise (https://mise.jdx.dev/) and Xcode
mise plugin add cocoapods
mise settings experimental=true
mise install

# Run the macOS app
bun macos
```

## Core Concepts

### Architecture

```
hanzoai/app/
  app/                    Next.js App Router pages
    (public)/             Public-facing pages
    actions/              Server actions
    admin/                Admin dashboard
    agents/               AI agent management
    api/                  API routes
    auth/                 Authentication pages
    billing/              Subscription / billing
    chat/                 AI chat interface
    dashboard/            Main dashboard
    deployments/          Deployment management
    dev/                  Developer tools / IDE
    docs/                 Documentation pages
    enterprise/           Enterprise features
    gallery/              Template gallery
    integrations/         Third-party integrations
    login/ signup/        Auth flows
    nodes/                Node management
    playground/           Code playground
    pricing/              Pricing pages
    profile/              User profile
    settings/             App settings
    templates/            Project templates
  components/             React components
    chat-panel/           AI chat UI
    editor/               Code editor (Monaco)
    file-explorer/        VFS file browser
    deployment-card/      Deployment management
    model-selector.tsx    Zen model picker
    monaco-editor/        Monaco wrapper
    workspace/            IDE workspace
    ui/                   Shared UI primitives
  macos/                  Native macOS layer
    hanzo-macOS/          Swift/ObjC native code
    hanzo.xcodeproj/      Xcode project
    Podfile               CocoaPods dependencies
  lib/                    Shared utilities
  public/                 Static assets
```

### Key Features

**Web IDE**:
- Monaco-based code editor with syntax highlighting
- In-browser JavaScript execution via QuickJS / esbuild-wasm
- File explorer with virtual filesystem
- Project templates and gallery
- Deploy to Hanzo Platform from the browser

**AI Integration**:
- Chat panel with Zen model selection
- Agent management and orchestration
- Code generation and debugging
- Context-aware assistance

**Desktop (macOS)**:
- Global hotkey (Tab) for instant AI access
- App search and launch
- Clipboard manager
- Window manager
- Calendar integration with menu bar appointments
- Custom AppleScript commands
- Wi-Fi password retrieval, IP display, process killer
- Emoji picker, notes scratchpad

### Environment Variables

```bash
# Configure via .env (see .env.example)
# API keys for AI providers
# Auth configuration
# Database URLs
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `bun macos` fails | Missing Mise/Xcode/CocoaPods | Install via `mise install` after setting up Mise |
| esbuild.wasm missing | postinstall script failed | Run `cp node_modules/esbuild-wasm/esbuild.wasm public/esbuild.wasm` |
| Build OOM | Large Next.js build | Set `NODE_OPTIONS=--max-old-space-size=4096` |

## Related Skills

- `hanzo/hanzo-chat.md` - AI chat (LibreChat fork)
- `hanzo/hanzo-llm-gateway.md` - LLM proxy for AI providers
- `hanzo/hanzo-studio.md` - Visual AI workflow builder (ComfyUI fork)
- `hanzo/hanzo-platform.md` - PaaS for deploying apps
- `hanzo/hanzo-extension.md` - Browser/IDE extensions

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: web-app, desktop, ide, ai-assistant, macos
**Prerequisites**: Node.js/bun, macOS for native app
