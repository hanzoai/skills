# Hanzo Operative - Computer Use Agent

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Operative enables **AI agents to control computers** — taking screenshots, moving the mouse, clicking, typing, and automating browser workflows. Built for Claude's computer use capabilities with Playwright integration.

### Why Hanzo Operative?

- **Full computer control**: Screen capture, mouse, keyboard, browser
- **Claude vision**: Uses Claude's multimodal understanding
- **Browser automation**: Playwright-based web interaction
- **Reproducible**: Record and replay workflows
- **MCP compatible**: Expose computer use as MCP tools

### Repo

`hanzoai/operative`. Local: ``github.com/hanzoai/operative``.

## When to use

- Automating web-based workflows
- AI-driven UI testing
- Screen scraping with AI understanding
- Building computer-use agents
- Automating repetitive desktop tasks

## Hard requirements

1. **Python 3.11+** with uv
2. **Playwright** installed (`playwright install`)
3. **ANTHROPIC_API_KEY** for Claude vision
4. Display server (X11/Wayland/macOS) for screen capture

## Quick reference

| Item | Value |
|------|-------|
| Install | `uv add hanzo-operative` |
| Repo | `github.com/hanzoai/operative` |
| Dev | `make dev` |
| Test | `uv run pytest -v` |

## One-file quickstart

### Basic Computer Use

```python
from hanzo_operative import Operative

async def main():
    op = Operative()

    # Take screenshot and analyze
    screenshot = await op.screenshot()
    analysis = await op.analyze(screenshot, "What windows are open?")
    print(analysis)

    # Click on element
    await op.click(x=500, y=300)

    # Type text
    await op.type("Hello from Hanzo Operative!")

    # Press keys
    await op.key("Enter")
```

### Browser Automation

```python
from hanzo_operative import Operative, Browser

async def main():
    op = Operative()
    browser = Browser()

    await browser.navigate("https://example.com")
    screenshot = await browser.screenshot()

    # AI-driven interaction
    action = await op.analyze(screenshot, "Click the login button")
    await browser.click(action.x, action.y)

    # Fill form
    await browser.fill("input[name=email]", "user@example.com")
    await browser.fill("input[name=password]", "secure-password")
    await browser.click("button[type=submit]")
```

### MCP Tool Exposure

```python
from hanzo_operative import Operative
from hanzo_operative.mcp import create_mcp_server

op = Operative()
server = create_mcp_server(op)

# Exposes tools: screenshot, click, type, navigate, analyze
await server.listen(8090)
```

## Core Concepts

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  AI Agent   │────▶│  Operative   │────▶│  Computer   │
│  (Claude)   │     │  (Python)    │     │  (Display)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │  Playwright  │──▶ Browser
                    │  PyAutoGUI   │──▶ Desktop
                    │  Screenshot  │──▶ Vision
                    └──────────────┘
```

### Workflow Recording

```python
from hanzo_operative import Operative, Recorder

async def main():
    op = Operative()
    recorder = Recorder()

    # Record human actions
    await recorder.start()
    # ... user performs actions ...
    workflow = await recorder.stop()

    # Replay with AI supervision
    await op.replay(workflow, supervise=True)
```

## Development

```bash
cd operative
make dev                # Start development
uv sync --all-extras    # Install dependencies
playwright install      # Install browsers
uv run pytest -v        # Run tests
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No display | Headless environment | Use `DISPLAY=:99 Xvfb :99 &` |
| Playwright not found | Not installed | `playwright install chromium` |
| Screenshot blank | Wrong display | Check DISPLAY env var |
| Click misaligned | Screen scaling | Account for DPI scaling |

## Related Skills

- `hanzo/hanzo-agent.md` - Agent framework (uses operative for computer use)
- `hanzo/hanzo-mcp.md` - MCP integration
- `hanzo/hanzo-extension.md` - Browser extension (different approach)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: computer-use, automation, browser, playwright
**Prerequisites**: Python, Playwright basics, Claude API
