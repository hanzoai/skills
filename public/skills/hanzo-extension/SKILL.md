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

`github.com/hanzoai/extension`. Version: **1.8.0**.

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

**The ONE way**: the `@hanzo/iam` SDK, OAuth2 authorization-code + **PKCE
`S256`** against the canonical `/v1/iam/oauth/*` endpoints. No implicit flow,
no hand-rolled OAuth, no legacy `/login/oauth/*`. See `hanzo/hanzo-id.md`.

```
1. Extension opens the SDK authorize URL (PKCE S256) in a tab / launchWebAuthFlow
2. User logs in on Hanzo IAM
3. Redirect to the registered callback with ?code=...&state=...
4. Extension catches the redirect and calls handleCallback() ONCE (single-use code)
5. SDK exchanges the code for tokens (PKCE) and stores them
```

**LLM endpoint**: `api.hanzo.ai/v1/chat/completions` (NOT `llm.hanzo.ai` which is Cloud UI)

### Auth Implementation

```typescript
// packages/browser/src/auth.ts
import { configureIam, getLoginUrl, handleCallback, getUser } from "@hanzo/iam/browser"

configureIam({
  serverUrl: "https://hanzo.id",       // issuer / brand host (not iam.hanzo.ai)
  clientId: "hanzo-extension",         // <org>-<app>
  redirectUri: chrome.identity.getRedirectURL("callback"),
  scope: "openid profile email",
})

async function startAuth(): Promise<void> {
  const authUrl = await getLoginUrl()  // PKCE S256 + state handled by the SDK
  const redirect = await chrome.identity.launchWebAuthFlow({ url: authUrl, interactive: true })
  // Single-use code: run the exchange exactly once.
  await handleCallback(redirect)
}
```

### User Profile

```typescript
// Read identity via the SDK — never a legacy /api/get-account call.
async function getProfile() {
  return getUser()  // { sub, email, name, owner, ... }
}
```

## Development

```bash
cd extension
pnpm install

# Browser extension
cd packages/browser
pnpm dev # Watch mode for Chrome
pnpm build # Production build

# VS Code extension
cd packages/vscode
pnpm dev # Watch mode
pnpm package # Create .vsix

# JetBrains plugin
cd packages/jetbrains
./gradlew buildPlugin # Create .zip
```

## CI/CD

- **Tag `v*`** → test → build → release (all platforms)
- **`publish.yml`** on release → Chrome Web Store, Firefox AMO, VS Code Marketplace, Open VSX, JetBrains Marketplace, npm
- Safari builds require macOS runner
- JetBrains needs `gradle-wrapper.jar` committed to repo

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Login fails / wrong path | Hardcoded/legacy OAuth path | Use `@hanzo/iam/browser` (PKCE S256, `/v1/iam/oauth/*`) — never hand-rolled implicit flow |
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
