# Hanzo Extensions - Browser & IDE Plugins

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-id.md`, `hanzo/hanzo-chat.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Extensions is a **monorepo of browser and IDE extensions** providing AI-powered development and browsing tools. Chrome, Firefox, Safari, VS Code, and JetBrains — all from a single pnpm workspace.

### Components

- **Browser Extension**: Chrome/Firefox/Safari — AI chat in browser
- **VS Code Extension**: @hanzo/extension — AI coding assistant
- **JetBrains Plugin**: IntelliJ/GoLand/PyCharm AI integration
- **AI Toolkit**: @hanzo/ai — shared AI primitives
- **ACI**: @hanzo/aci — Agent Computer Interface
- **MCP**: @hanzo/mcp — MCP server package
- **CLI Tools**: @hanzo/cli-tools — terminal integration

### Repo

`hanzoai/extension`. Local: ``github.com/hanzoai/extension``. Version: **1.8.0**.

## When to use

- Building or modifying browser/IDE extensions
- Adding AI features to developer tools
- Implementing browser-based OAuth with Hanzo ID
- Packaging extensions for multiple platforms

## Hard requirements

1. **pnpm** workspace manager
2. **Node.js 18+**
3. For Safari: macOS with Xcode
4. For JetBrains: gradle-wrapper.jar in repo

## Quick reference

| Package | Path | Published As |
|---------|------|-------------|
| Browser | `packages/browser/` | Chrome Web Store, Firefox AMO |
| VS Code | `packages/vscode/` | VS Code Marketplace, Open VSX |
| JetBrains | `packages/jetbrains/` | JetBrains Marketplace |
| AI | `packages/ai/` | `@hanzo/ai` (npm) |
| ACI | `packages/aci/` | `@hanzo/aci` (npm) |
| MCP | `packages/mcp/` | `@hanzo/mcp` (npm) |
| Tools | `packages/tools/` | `@hanzo/cli-tools` (npm) |
| Site | `apps/site/` | Marketing website |

## Browser Extension Auth Flow

**CRITICAL**: Uses implicit OAuth2 (`response_type=token`), NOT code flow.

```
1. Extension opens tab → hanzo.id/login/oauth/authorize?response_type=token&...
2. User logs in on Casdoor form
3. Redirect to hanzo.ai/callback?access_token=JWT&state=...
4. Extension catches redirect via chrome.tabs.onUpdated
5. Extracts token, closes tab
```

**LLM endpoint**: `api.hanzo.ai/v1/chat/completions` (NOT `llm.hanzo.ai` which is Cloud UI)

### Auth Implementation

```typescript
// packages/browser/src/auth.ts
async function startAuth(): Promise<string> {
  const state = crypto.randomUUID()

  const authUrl = new URL("https://hanzo.id/login/oauth/authorize")
  authUrl.searchParams.set("response_type", "token")
  authUrl.searchParams.set("client_id", "app-hanzo")
  authUrl.searchParams.set("redirect_uri", "https://hanzo.ai/callback")
  authUrl.searchParams.set("scope", "openid profile email")
  authUrl.searchParams.set("state", state)

  // Open auth tab
  const tab = await chrome.tabs.create({ url: authUrl.toString() })

  // Listen for redirect
  return new Promise((resolve) => {
    chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
      if (tabId !== tab.id || !info.url?.includes("callback")) return

      const hash = new URL(info.url).hash.substring(1)
      const params = new URLSearchParams(hash)
      const token = params.get("access_token")

      chrome.tabs.onUpdated.removeListener(listener)
      chrome.tabs.remove(tabId)
      resolve(token!)
    })
  })
}
```

### User Profile

```typescript
// /api/userinfo only returns `sub` — use /api/get-account
async function getProfile(token: string) {
  const res = await fetch("https://iam.hanzo.ai/api/get-account", {
    headers: { "Authorization": `Bearer ${token}` }
  })
  return res.json()  // { name, email, avatar, ... }
}
```

## Development

```bash
cd extension
pnpm install

# Browser extension
cd packages/browser
pnpm dev           # Watch mode for Chrome
pnpm build         # Production build

# VS Code extension
cd packages/vscode
pnpm dev           # Watch mode
pnpm package       # Create .vsix

# JetBrains plugin
cd packages/jetbrains
./gradlew buildPlugin  # Create .zip
```

## CI/CD

- **Tag `v*`** → test → build → release (all platforms)
- **`publish.yml`** on release → Chrome Web Store, Firefox AMO, VS Code Marketplace, Open VSX, JetBrains Marketplace, npm
- Safari builds require macOS runner
- JetBrains needs `gradle-wrapper.jar` committed to repo

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Code flow fails | Casdoor empty grant_type bug | Use implicit flow (response_type=token) |
| LLM returns 404 | Using llm.hanzo.ai | Use api.hanzo.ai/v1/chat/completions |
| Safari build fails | Missing Xcode | Build on macOS with Xcode installed |
| JetBrains build fails | Missing gradle wrapper | Commit gradle-wrapper.jar |

## Related Skills

- `hanzo/hanzo-id.md` - Auth provider
- `hanzo/hanzo-chat.md` - LLM API backend
- `hanzo/hanzo-mcp.md` - MCP tools
- `hanzo/hanzo-agent.md` - Agent SDK

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: extensions, browser, vscode, jetbrains, ide
**Prerequisites**: TypeScript, browser extension APIs, OAuth2
