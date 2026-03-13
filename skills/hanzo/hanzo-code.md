# Hanzo Code - Open-Source AI Code Editor

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-extension.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-aci.md`

## Overview

Hanzo Code is an **open-source AI-native code editor** -- a fork of VS Code (microsoft/vscode) with integrated LLM chat, autocomplete, inline diffs, and agent tool use baked directly into the editor core. It is the open-source Cursor alternative. The custom AI code lives in `src/vs/workbench/contrib/code/browser/`. Ships as a standalone Electron desktop app for macOS, Windows, and Linux.

### Why Hanzo Code?

- **AI-native editor**: Chat sidebar, Cmd+K inline edits, autocomplete, and agent mode built into the editor -- not an extension
- **Multi-provider LLM**: Supports OpenAI, Anthropic, Google Gemini, Mistral, Groq, Ollama (local), and any OpenAI-compatible endpoint
- **Fast Apply**: Search/Replace-based code application that works on 1000+ line files without rewriting the whole file
- **MCP tool use**: Model Context Protocol integration for agent tool calling
- **Full VS Code compatibility**: VS Code extensions, themes, settings, and keybindings all work

### Tech Stack

- **Language**: TypeScript (Electron app -- main + browser processes)
- **UI Framework**: React 19 (bundled to browser process via custom build step)
- **Styling**: Tailwind CSS 3 (scoped)
- **Build**: Gulp + Webpack, npm scripts
- **Runtime**: Electron 34, Node 20.18.x
- **LLM SDKs**: `@anthropic-ai/sdk`, `openai`, `@google/genai`, `@mistralai/mistralai`, `groq-sdk`, `ollama`
- **MCP**: `@modelcontextprotocol/sdk ^1.11.2`
- **Testing**: Mocha (unit), Playwright (browser/e2e)

### OSS Base

Fork of [microsoft/vscode](https://github.com/microsoft/vscode) v1.94.0. Previously known as "Void Editor" -- references to `void` remain in internal code paths and product.json `applicationName`.

Repo: `github.com/hanzoai/code`

## When to use

- Building or contributing to the Hanzo Code editor
- Adding new LLM providers or models
- Extending the AI chat, autocomplete, or apply features
- Creating new agent tools or MCP integrations within the editor
- Packaging the editor for distribution

## Hard requirements

1. **Node.js 20.18.x** (see `.nvmrc`)
2. **npm** (not pnpm -- this is a VS Code fork and uses npm workspaces)
3. **Python** and **XCode** (macOS) or **Visual Studio 2022** (Windows) or **build-essential** (Linux) for native modules
4. **Electron 34** (installed via devDependencies)

## Quick reference

| Item | Value |
|------|-------|
| Website | `https://code.hanzo.ai` |
| Repo | `github.com/hanzoai/code` |
| Branch | `main` |
| License | MIT |
| Node Version | `20.18.2` |
| Package Manager | npm (not pnpm) |
| Editor Version | `1.94.0` (VS Code base) |
| Custom Code Path | `src/vs/workbench/contrib/code/browser/` |
| Data Folder | `~/.code-editor` |
| Bundle ID (macOS) | `com.hanzoai.code` |
| URL Protocol | `code-editor://` |

## One-file quickstart

### Build and run from source

```bash
git clone https://github.com/hanzoai/code.git
cd code

# Install dependencies
npm install

# Build React UI components
npm run buildreact

# Watch-compile the editor (wait for two checkmarks)
npm run watch

# In another terminal, launch the editor
./scripts/code.sh          # macOS/Linux
# ./scripts/code.bat       # Windows

# Optional: isolate test data
./scripts/code.sh --user-data-dir ./.tmp/user-data --extensions-dir ./.tmp/extensions
```

### Package for distribution

```bash
# macOS Apple Silicon
npm run gulp vscode-darwin-arm64

# macOS Intel
npm run gulp vscode-darwin-x64

# Windows
npm run gulp vscode-win32-x64

# Linux
npm run gulp vscode-linux-x64
```

Output folder appears as a sibling directory (e.g. `../VSCode-darwin-arm64/`).

## Core Concepts

### Architecture

Code is an Electron app with two processes:

```
┌───────────────────────────────┐
│       Browser Process         │
│  (HTML/CSS/React UI)          │
│                               │
│  src/.../code/browser/        │
│  ├── react/          (React)  │
│  ├── chatThreadService        │
│  ├── autocompleteService      │
│  ├── editCodeService          │
│  ├── toolsService             │
│  ├── inlineDiffsService       │
│  └── sidebarPane              │
└──────────┬────────────────────┘
           │ IPC channels
┌──────────▼────────────────────┐
│       Main Process            │
│  (Node.js, node_modules)      │
│                               │
│  sendLLMMessage (all providers│
│  Anthropic, OpenAI, Gemini,   │
│  Mistral, Groq, Ollama)       │
└───────────────────────────────┘
```

- `browser/` code can use `window` and DOM but cannot import `node_modules`
- `electron-main/` code can import `node_modules` but has no DOM
- `common/` code is shared between both processes
- LLM API calls run on `electron-main` to avoid CSP issues with local providers

### Key Services

| Service | File | Purpose |
|---------|------|---------|
| `chatThreadService` | 67KB | Chat sidebar message handling, streaming, agent mode |
| `editCodeService` | 89KB | Apply (Fast/Slow), Cmd+K, Edit tool -- all code modifications |
| `autocompleteService` | 33KB | Inline completions from LLM |
| `inlineDiffsService` | 64KB | Red/green diff rendering for pending changes |
| `toolsService` | 24KB | Agent tool registration and execution |
| `terminalToolService` | 14KB | Terminal command execution for agent |
| `voidSettingsService` | -- | Provider/model configuration (implicit dependency for all AI services) |
| `contextGatheringService` | 13KB | File and selection context for LLM prompts |
| `voidCommandBarService` | 28KB | Command palette integration |

### Apply System

Two modes:
- **Fast Apply**: LLM outputs `<<<<<<< ORIGINAL` / `=======` / `>>>>>>> UPDATED` search/replace blocks. Works on large files.
- **Slow Apply**: LLM rewrites the entire file content.

Key types:
- **DiffZone**: A `{startLine, endLine}` region showing red/green diffs, with an optional `llmCancelToken` for streaming
- **DiffArea**: Generalized line-number tracker
- **ModelSelection**: `{providerName, modelName}` pair
- **ChatMode**: `normal | gather | agent`

### VS Code Internals

- **Editor**: The Monaco editor instance (one per split pane, tabs share it)
- **Model** (`ITextModel`): Internal file content representation, shared across editors
- **URI**: File path identifier for models
- **Service**: Singleton class registered with `registerSingleton`, injected via `@IServiceName` decorator
- **Action/Command**: Registered functions callable from command palette (Cmd+Shift+P) or by ID

## Directory structure

```
code/
├── src/vs/workbench/contrib/code/browser/   # ALL custom Hanzo Code AI features
│   ├── react/                    # React UI (chat sidebar, settings)
│   ├── helpers/                  # Shared utility functions
│   ├── helperServices/           # Small helper singleton services
│   ├── prompt/                   # LLM prompt templates
│   ├── media/                    # CSS and assets
│   ├── chatThreadService.ts      # Chat + agent orchestration
│   ├── editCodeService.ts        # Apply, Cmd+K, Edit tool
│   ├── autocompleteService.ts    # Inline completions
│   ├── inlineDiffsService.ts     # Diff visualization
│   ├── toolsService.ts           # Agent tool system
│   └── ...                       # 35+ service files
├── src/vs/                       # VS Code core (mostly untouched)
├── extensions/                   # Built-in VS Code extensions
├── build/                        # Build scripts (Gulp tasks)
├── cli/                          # CLI tool
├── remote/                       # Remote development server
├── test/                         # Test infrastructure
├── product.json                  # Editor identity and branding
├── package.json                  # npm deps (v1.94.0)
└── .nvmrc                        # Node 20.18.2
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails with native module errors | Wrong Node version | Use Node 20.18.x (check `.nvmrc`) |
| `TypeError: Failed to fetch dynamically imported module` | Missing `.js` extension in imports | Ensure all TS imports end with `.js` |
| Missing styles after build | CSS not yet compiled | Wait a few seconds, then Cmd+R to reload |
| React UI not rendering | React not built | Run `npm run buildreact` before watch |
| CSP errors with local LLM | Provider called from browser process | LLM calls must go through electron-main IPC |
| Extensions gallery empty | Missing `extensionsGallery` in product.json | Already configured for VS Marketplace |

### Development rules (from .voidrules)

- Never modify files outside `src/vs/workbench/contrib/code/` without consulting first
- Never lazily cast to `any` -- find the correct type
- Do not add or remove semicolons -- follow existing convention
- Type naming: `bOfA` pattern (e.g. `toolNameOfToolId` for a map from tool ID to name)

## Related Skills

- `hanzo/hanzo-extension.md` - VS Code / browser extension (separate repo, not the editor itself)
- `hanzo/hanzo-aci.md` - Agent Computer Interface (Python backend for agent tools)
- `hanzo/hanzo-mcp.md` - Model Context Protocol tools
- `hanzo/hanzo-llm-gateway.md` - LLM proxy (api.hanzo.ai)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: editor, vscode, ai, llm, cursor-alternative
**Prerequisites**: Node.js 20, npm, platform build tools (XCode/VS/gcc)
