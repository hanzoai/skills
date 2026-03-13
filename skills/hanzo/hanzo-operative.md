# Hanzo Operative - Computer Use Agent

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Operative enables **AI agents to control computers** — taking screenshots, moving the mouse, clicking, typing, and automating browser workflows. Fork of Anthropic's computer-use demo with Hanzo ecosystem integration.

### Why Hanzo Operative?

- **Full computer control**: Screen capture, mouse, keyboard, browser
- **Claude vision**: Uses Claude's multimodal understanding for screen analysis
- **Browser automation**: Playwright-based web interaction
- **Reproducible**: Record and replay workflows
- **MCP compatible**: Expose computer use as MCP tools
- **Multi-provider**: Works with Claude, Zen, or any vision-capable model

### OSS Base

Fork of **Anthropic computer-use demo** v0.1.1. Repo: `hanzoai/operative`.

## When to use

- Automating web-based workflows via AI
- AI-driven UI testing and QA
- Screen scraping with AI understanding
- Building computer-use agents for complex tasks
- RPA (Robotic Process Automation) with AI reasoning

## Hard requirements

1. **Python 3.11+** with uv (note: `>=3.14` in upstream is overly restrictive)
2. **Playwright** installed (`playwright install chromium`)
3. **ANTHROPIC_API_KEY** or **HANZO_API_KEY** for vision model
4. Display server (X11/Wayland/macOS) for screen capture

## Quick reference

| Item | Value |
|------|-------|
| Install | `uv add hanzo-operative` |
| Version | 0.1.1 |
| Repo | `github.com/hanzoai/operative` |
| Upstream | Anthropic computer-use demo |
| Dev | `make dev` |
| Test | `uv run pytest -v` |
| Python | 3.11+ |
| License | MIT |

## One-file quickstart

### Basic Computer Use

```python
import asyncio
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

    # Key combo
    await op.key("Control+C")

    # Mouse movement
    await op.move(x=100, y=200)

    # Scroll
    await op.scroll(direction="down", amount=3)

asyncio.run(main())
```

### Browser Automation

```python
from hanzo_operative import Operative, Browser

async def main():
    op = Operative()
    browser = Browser()

    # Navigate
    await browser.navigate("https://example.com")
    screenshot = await browser.screenshot()

    # AI-driven interaction — analyze what to click
    action = await op.analyze(screenshot, "Find and click the login button")
    await browser.click(action.x, action.y)

    # Fill form using CSS selectors
    await browser.fill("input[name=email]", "user@example.com")
    await browser.fill("input[name=password]", "secure-password")
    await browser.click("button[type=submit]")

    # Wait for navigation
    await browser.wait_for("text=Dashboard")

    # Extract text
    text = await browser.text("h1")
    print(f"Page title: {text}")

    await browser.close()

asyncio.run(main())
```

### AI-Driven Task Execution

```python
from hanzo_operative import Operative

async def main():
    op = Operative()

    # High-level task — AI decides the steps
    result = await op.execute_task(
        "Go to github.com/hanzoai, find the most starred repo, "
        "and tell me its name and star count"
    )
    print(result)  # AI navigates, reads, and reports

asyncio.run(main())
```

### MCP Tool Exposure

```python
from hanzo_operative import Operative
from hanzo_operative.mcp import create_mcp_server

op = Operative()
server = create_mcp_server(op)

# Exposes tools: screenshot, click, type, navigate, analyze, scroll, key
await server.serve_stdio()
```

### Workflow Recording

```python
from hanzo_operative import Operative, Recorder

async def main():
    op = Operative()
    recorder = Recorder()

    # Record human actions
    await recorder.start()
    # ... user performs actions (clicks, types, navigates) ...
    workflow = await recorder.stop()

    # Save workflow
    workflow.save("my_workflow.json")

    # Replay with AI supervision
    await op.replay(workflow, supervise=True)

    # Replay without supervision (faster)
    await op.replay(workflow, supervise=False)
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
                    │  Backends    │
                    ├──────────────┤
                    │ Playwright   │──▶ Browser (Chromium/Firefox)
                    │ PyAutoGUI    │──▶ Desktop (mouse/keyboard)
                    │ Pillow       │──▶ Screenshots (capture/crop)
                    │ Claude API   │──▶ Vision (analysis/OCR)
                    └──────────────┘
```

### Action Types

| Action | Method | Description |
|--------|--------|-------------|
| Screenshot | `op.screenshot()` | Capture full screen or region |
| Click | `op.click(x, y)` | Mouse click at coordinates |
| Double Click | `op.double_click(x, y)` | Double mouse click |
| Right Click | `op.right_click(x, y)` | Context menu click |
| Type | `op.type(text)` | Type text string |
| Key | `op.key(combo)` | Press key or combo |
| Move | `op.move(x, y)` | Move mouse cursor |
| Scroll | `op.scroll(dir, amt)` | Scroll up/down/left/right |
| Drag | `op.drag(x1, y1, x2, y2)` | Click-drag operation |
| Analyze | `op.analyze(img, prompt)` | AI vision analysis |
| Wait | `op.wait(seconds)` | Pause between actions |

### Configuration

```python
op = Operative(
    model="claude-sonnet-4-20250514",  # Vision model
    api_key=os.environ["ANTHROPIC_API_KEY"],
    screenshot_delay=0.5,        # Delay before screenshots
    action_delay=0.3,            # Delay between actions
    max_retries=3,               # Retry failed actions
    viewport_width=1920,         # Browser viewport
    viewport_height=1080,
)
```

## Development

```bash
git clone https://github.com/hanzoai/operative.git
cd operative
uv sync --all-extras         # Install dependencies
playwright install chromium  # Install browser
make dev                     # Start development
uv run pytest -v             # Run tests
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No display | Headless server | Use `DISPLAY=:99 Xvfb :99 &` |
| Playwright not found | Not installed | `playwright install chromium` |
| Screenshot blank | Wrong display | Check `DISPLAY` env var |
| Click misaligned | DPI scaling | Account for display scaling factor |
| Python version | `>=3.14` too strict | Use Python 3.11+ (upstream bug) |
| Vision API error | Wrong API key | Check `ANTHROPIC_API_KEY` |
| Timeout on action | Page still loading | Add `await op.wait(2)` before action |

## Related Skills

- `hanzo/hanzo-agent.md` - Agent framework (uses operative for computer use)
- `hanzo/hanzo-mcp.md` - MCP integration
- `hanzo/hanzo-extension.md` - Browser extension (different approach — extends existing browser)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: computer-use, automation, browser, playwright, vision
**Prerequisites**: Python 3.11+, Playwright, Claude API (or Hanzo API)
