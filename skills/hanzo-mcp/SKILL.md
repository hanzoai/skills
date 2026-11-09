---
name: hanzo-mcp
description: Progressive disclosure guide and index for Hanzo MCP (@hanzo/mcp, HIP-0300) and api.hanzo.ai/v1/mcp — load 13 unified tools first, then expand action schemas and 188 cloud capability groups on demand without context bloat.
---

# Hanzo MCP — Progressive Tool Disclosure (`@hanzo/mcp`, HIP-0300)

> Do not dump 200+ tool definitions into an agent's context window on turn 1.
> Hanzo MCP uses **3-level progressive disclosure** so an agent starts with a compact 13-tool surface (<1.5k tokens) and expands specific tool or cloud sub-schemas only when needed.

## Level 0: The 13 Unified Local + Cloud Tools (HIP-0300)

Install or run locally via `npx -y --package=@hanzo/mcp hanzo-mcp serve` or remotely via `POST https://api.hanzo.ai/v1/mcp`:

| Tool | Category | Actions (`action` param) | Progressive Schema URL |
|---|---|---|---|
| **`fs`** | Filesystem | `read`, `write`, `edit`, `list`, `stat`, `mkdir`, `rm`, `mv`, `cp`, `patch` | `https://hanzoskills.com/mcp/fs.json` |
| **`exec`** | Execution | `run`, `bg`, `ps`, `logs`, `kill`, `stdin`, `wait` | `https://hanzoskills.com/mcp/exec.json` |
| **`code`** | Semantic Code | `search`, `symbols`, `definition`, `references`, `outline`, `ast`, `diagnostics` | `https://hanzoskills.com/mcp/code.json` |
| **`git`** | Version Control | `status`, `diff`, `log`, `commit`, `branch`, `checkout`, `push`, `pr` | `https://hanzoskills.com/mcp/git.json` |
| **`test`** | Verification | `run`, `watch`, `coverage`, `single`, `list` | `https://hanzoskills.com/mcp/test.json` |
| **`fetch`** | Network & Docs | `get`, `post`, `head`, `scrape`, `markdown`, `openapi` | `https://hanzoskills.com/mcp/fetch.json` |
| **`workspace`** | Project Context | `detect`, `roots`, `packages`, `env`, `scripts`, `graph` | `https://hanzoskills.com/mcp/workspace.json` |
| **`computer`** | Browser & GUI | `tabs`, `navigate`, `snapshot`, `click`, `type`, `press`, `screenshot`, `evaluate`, `social.sessions`, `social.sync`, `social.post` | `https://hanzoskills.com/mcp/computer.json` |
| **`think`** | Reasoning | `thought`, `critic`, `decompose`, `verify`, `hypothesis` | `https://hanzoskills.com/mcp/think.json` |
| **`memory`** | Durable Brain | `recall`, `store`, `facts`, `graph`, `search`, `forget` | `https://hanzoskills.com/mcp/memory.json` |
| **`hanzo`** | Sovereign Cloud | `iam`, `kms`, `base`, `cloud`, `deploy`, `platform`, `billing`, `crm`, `cms`, `social`, `post`, `sandbox`, `describe` | `https://hanzoskills.com/mcp/hanzo.json` |
| **`plan`** | Architecture | `create`, `update`, `show`, `approve`, `step` | `https://hanzoskills.com/mcp/plan.json` |
| **`tasks`** | Task Tracking | `list`, `add`, `update`, `complete`, `assign` | `https://hanzoskills.com/mcp/tasks.json` |

## Level 1: Discovering Sub-Capabilities On Demand

Instead of pre-loading every cloud route or browser command:

1. **Fetch the compact JSON index**:
   ```bash
   curl -s https://hanzoskills.com/mcp/index.json
   ```
2. **Fetch a single tool's full schema when needed**:
   ```bash
   curl -s https://hanzoskills.com/mcp/computer.json
   curl -s https://hanzoskills.com/mcp/hanzo.json
   ```
3. **Or call `describe` on the live MCP endpoint**:
   ```bash
   curl -s https://api.hanzo.ai/v1/mcp \
     -H "Authorization: Bearer $(hanzo auth token)" \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"describe","arguments":{"capability":"sandbox"}}}'
   ```

## Level 2: Browser & Social ZAP Bridge (`computer` + `hanzo-browser`)

The `computer` MCP tool and `hanzo-browser` CLI connect over the local ZAP unix socket (`$XDG_RUNTIME_DIR/zap/zapd.sock`) to the signed-in Hanzo Browser Extension (`biingenefmanpecedoafkfajbnlgdmbl`):

```bash
# Lifecycle of persistent sandboxed Chromium (with 1Password, LastPass, Hanzo Extension)
hanzo-browser start              # Start or resume persistent browser on CDP 127.0.0.1:9222
hanzo-browser suspend            # Freeze browser in-place via SIGSTOP (0% CPU, preserves state)
hanzo-browser resume             # Thaw suspended browser via SIGCONT
hanzo-browser stop               # Stop browser cleanly (preserves ~/.hanzo/browser-sandbox/profile)
hanzo-browser status             # JSON status of PID, CDP URL, profile, and installed extensions

# Zero-API-key social session discovery, syncing, and posting via ZAP
hanzo-browser tabs               # List open browser tabs via Target.getTargets
hanzo-browser sessions           # Detect active logins across X, LinkedIn, YouTube, Reddit, etc.
hanzo-browser sync hanzo         # Sync connected social accounts to api.hanzo.ai/v1/social/accounts
hanzo-browser post x "Hello!"    # Publish directly through the logged-in browser session
```
