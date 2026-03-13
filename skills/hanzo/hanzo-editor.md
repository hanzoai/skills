# Hanzo Editor - Open-Source AI Code Editor

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-code.md`, `hanzo/hanzo-extension.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Editor (`hanzoai/code`) is an **open-source AI-native code editor** -- a VS Code fork (originally "Void Editor") with LLM chat, autocomplete, inline diffs, and agent tool use built directly into the editor core. It is the open-source Cursor alternative. Ships as a standalone Electron desktop app.

**Note**: This skill documents the same repo as `hanzo/hanzo-code.md`. The repo is `hanzoai/code`, not `hanzoai/editor` (which does not exist). This file exists as a convenience alias.

### What it actually is

- VS Code fork (v1.94.0 base) rebranded as "Code"
- Custom AI code lives in `src/vs/workbench/contrib/void/`
- Electron app: main process (Node.js, LLM API calls) + browser process (UI, React)
- Multi-provider LLM: OpenAI, Anthropic, Gemini, Mistral, Groq, Ollama (local)
- Fast Apply: search/replace-based diffs for large files
- MCP tool use for agent capabilities
- Full VS Code extension/theme/keybinding compatibility
- React 19 UI bundled to browser process

### Upstream references

- Product identity: `product.json` has `applicationName: "void"`, `dataFolderName: ".code-editor"`, `darwinBundleIdentifier: "com.hanzoai.code"`
- Internal docs still reference "Void" (`VOID_CODEBASE_GUIDE.md`, `.voidrules`, `void_icons/`)
- Upstream trusted domains include `voideditor.com`, `voideditor.dev`

## When to use

- Contributing to the Hanzo Code editor
- Adding new LLM providers or models
- Extending AI chat, autocomplete, or apply features
- Creating agent tools or MCP integrations
- Packaging the editor for distribution

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/code` |
| Branch | `main` |
| License | Apache-2.0 (+ MIT for VS Code portions) |
| Stack | TypeScript, Electron 34, React 19, Tailwind CSS 3 |
| Node | 20.18.x (`.nvmrc`) |
| Package Manager | npm (not pnpm -- VS Code convention) |
| Custom Code | `src/vs/workbench/contrib/void/` |
| Data Folder | `~/.code-editor` |
| Bundle ID | `com.hanzoai.code` |
| URL Protocol | `code-editor://` |
| Website | `code.hanzo.ai` |

## Quickstart

```bash
git clone https://github.com/hanzoai/code.git
cd code
npm install

# Build React UI
npm run buildreact

# Watch-compile (wait for two checkmarks)
npm run watch

# Launch editor (separate terminal)
./scripts/code.sh         # macOS/Linux
./scripts/code.bat        # Windows

# Package for distribution
npm run gulp vscode-darwin-arm64   # macOS ARM
npm run gulp vscode-darwin-x64     # macOS Intel
npm run gulp vscode-win32-x64     # Windows
npm run gulp vscode-linux-x64     # Linux
```

## Architecture

```
┌─────────────────────────────┐
│     Browser Process         │
│  (React UI, Monaco editor)  │
│                             │
│  src/.../void/browser/      │
│  chatThreadService          │
│  autocompleteService        │
│  editCodeService            │
│  toolsService               │
└────────────┬────────────────┘
             │ IPC channels
┌────────────▼────────────────┐
│     Main Process            │
│  (Node.js, LLM SDKs)       │
│                             │
│  sendLLMMessage             │
│  Anthropic, OpenAI, Gemini  │
│  Mistral, Groq, Ollama      │
└─────────────────────────────┘
```

## Directory structure

```
hanzoai/code/
  src/vs/workbench/contrib/void/browser/   # All custom AI features
    react/              # React UI (chat sidebar, settings)
    chatThreadService   # Chat + agent orchestration
    editCodeService     # Apply, Cmd+K, Edit tool
    autocompleteService # Inline completions
    inlineDiffsService  # Diff visualization
    toolsService        # Agent tool system
  extensions/           # Built-in VS Code extensions
  build/                # Gulp build scripts
  cli/                  # CLI tool
  remote/               # Remote dev server
  test/                 # Test infrastructure
  void_icons/           # Editor icons
  product.json          # Editor identity/branding
  package.json          # Dependencies
  .voidrules            # AI assistant rules for this codebase
```

## Development rules

From `.voidrules`:
- Never modify files outside `src/vs/workbench/contrib/void/` without consulting first
- Never cast to `any` -- find the correct type
- Do not change semicolons -- follow existing convention
- Type naming: `bOfA` pattern (e.g., `toolNameOfToolId`)

## Key services

| Service | Purpose |
|---------|---------|
| `chatThreadService` | Chat sidebar, streaming, agent mode |
| `editCodeService` | Fast/Slow Apply, Cmd+K, Edit tool |
| `autocompleteService` | Inline LLM completions |
| `inlineDiffsService` | Red/green diff rendering |
| `toolsService` | Agent tool registration and execution |
| `voidSettingsService` | Provider/model configuration |

## Related Skills

- `hanzo/hanzo-code.md` -- Comprehensive version of this skill (same repo)
- `hanzo/hanzo-extension.md` -- VS Code / browser extension (separate product)
- `hanzo/hanzo-mcp.md` -- MCP tools used by the editor's agent mode
- `hanzo/hanzo-llm-gateway.md` -- LLM proxy (api.hanzo.ai)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: editor, vscode, void, ai, cursor-alternative, code
**Prerequisites**: Node.js 20, npm, platform build tools (XCode/VS/gcc)
