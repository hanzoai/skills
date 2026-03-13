# Hanzo Operative - Computer Use Agent

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`, `hanzo/hanzo-extension.md`

## Overview

Hanzo Operative enables **AI agents to control computers** via X11 desktop automation — taking screenshots, moving the mouse, clicking, typing, and executing complex desktop workflows. Fork of Anthropic's computer-use demo with Hanzo ecosystem integration.

**NOTE**: Operative uses **X11/xdotool/Xvfb** for desktop automation, NOT Playwright. It controls the full desktop environment (any application), not just browsers. Has a Streamlit UI for interactive sessions and VNC for remote viewing.

### Why Hanzo Operative?

- **Full desktop control**: X11-based screen capture, mouse, keyboard
- **Claude vision**: Uses Claude's multimodal understanding for screen analysis
- **Any application**: Controls any desktop app, not just browsers
- **Streamlit UI**: Interactive web UI for monitoring and controlling sessions (port 8501)
- **VNC access**: Remote desktop viewing via VNC (5900) and noVNC (6080)
- **Dockerized**: Runs in container with Xvfb virtual display
- **MCP compatible**: Expose computer use as MCP tools

### OSS Base

Fork of **Anthropic computer-use demo** v0.1.1. Repo: `hanzoai/operative`.

## When to use

- Automating desktop workflows via AI vision
- AI-driven UI testing and QA across any application
- Screen scraping with AI understanding
- Building computer-use agents for complex multi-app tasks
- RPA (Robotic Process Automation) with AI reasoning
- Remote desktop automation via VNC

## Hard requirements

1. **Python 3.11+** with uv
2. **X11 display server** (Xvfb for headless, native X11 for desktop)
3. **xdotool** for mouse/keyboard simulation
4. **ANTHROPIC_API_KEY** or **HANZO_API_KEY** for vision model
5. Docker recommended (bundles Xvfb + xdotool + VNC)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/operative` |
| Upstream | Anthropic computer-use demo |
| UI | Streamlit (port 8501) |
| VNC | Port 5900 (native), 6080 (noVNC web) |
| Dev | `make dev` |
| Docker | `docker compose up` |
| Test | `uv run pytest -v` |
| Python | 3.11+ |
| License | MIT |

## One-file quickstart

### Docker (Recommended)

```bash
git clone https://github.com/hanzoai/operative.git
cd operative

# Set API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Start with Docker (includes Xvfb, xdotool, VNC, Streamlit)
docker compose up -d

# Access Streamlit UI
open http://localhost:8501

# Access VNC (view desktop)
open http://localhost:6080  # noVNC web client
# or connect VNC client to localhost:5900
```

### Native (Linux/macOS with X11)

```bash
git clone https://github.com/hanzoai/operative.git
cd operative
uv sync --all-extras

# Install system deps (Ubuntu/Debian)
sudo apt install xdotool xvfb scrot

# Start Xvfb if headless
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Run
uv run streamlit run app.py --server.port 8501
```

### Programmatic Usage

```python
import asyncio
from operative import Operative

async def main():
    op = Operative(
        display=":99",  # X11 display
        model="claude-sonnet-4-20250514",
    )

    # Take screenshot and analyze
    screenshot = await op.screenshot()
    analysis = await op.analyze(screenshot, "What windows are open?")
    print(analysis)

    # Mouse operations (via xdotool)
    await op.click(x=500, y=300)
    await op.move(x=100, y=200)
    await op.double_click(x=500, y=300)
    await op.right_click(x=500, y=300)

    # Keyboard operations (via xdotool)
    await op.type("Hello from Hanzo Operative!")
    await op.key("Return")
    await op.key("ctrl+c")
    await op.key("alt+Tab")

    # Scroll
    await op.scroll(direction="down", amount=3)

    # Drag
    await op.drag(x1=100, y1=100, x2=500, y2=500)

asyncio.run(main())
```

### AI-Driven Task Execution

```python
from operative import Operative

async def main():
    op = Operative()

    # High-level task — AI decides the steps using vision
    result = await op.execute_task(
        "Open Firefox, go to github.com/hanzoai, find the most starred repo, "
        "and tell me its name and star count"
    )
    print(result)  # AI navigates desktop, reads screen, and reports

asyncio.run(main())
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  AI Agent   │────▶│  Operative   │────▶│  X11 Display │
│  (Claude)   │     │  (Python)    │     │  (Xvfb/X11) │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │   Backends   │
                    ├──────────────┤
                    │ xdotool      │──▶ Mouse/Keyboard simulation
                    │ scrot/import │──▶ Screenshot capture
                    │ Claude API   │──▶ Vision analysis/OCR
                    │ Streamlit    │──▶ Web UI (port 8501)
                    │ VNC/noVNC    │──▶ Remote viewing (5900/6080)
                    └──────────────┘
```

## Action Types

| Action | Method | Backend | Description |
|--------|--------|---------|-------------|
| Screenshot | `op.screenshot()` | scrot/import | Capture full screen or region |
| Click | `op.click(x, y)` | xdotool | Left mouse click at coordinates |
| Double Click | `op.double_click(x, y)` | xdotool | Double mouse click |
| Right Click | `op.right_click(x, y)` | xdotool | Context menu click |
| Type | `op.type(text)` | xdotool | Type text string |
| Key | `op.key(combo)` | xdotool | Press key or combo (e.g., "ctrl+c") |
| Move | `op.move(x, y)` | xdotool | Move mouse cursor |
| Scroll | `op.scroll(dir, amt)` | xdotool | Scroll up/down/left/right |
| Drag | `op.drag(x1,y1,x2,y2)` | xdotool | Click-drag operation |
| Analyze | `op.analyze(img, prompt)` | Claude API | AI vision analysis |
| Wait | `op.wait(seconds)` | asyncio | Pause between actions |

## Docker Compose

```yaml
# compose.yml
services:
  operative:
    build: .
    ports:
      - "8501:8501"   # Streamlit UI
      - "5900:5900"   # VNC
      - "6080:6080"   # noVNC (web VNC client)
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DISPLAY=:99
      - RESOLUTION=1920x1080
    volumes:
      - ./recordings:/app/recordings
```

## Streamlit UI

The Streamlit interface (port 8501) provides:
- Live screenshot view of the virtual desktop
- Task input field for natural language commands
- Action history with screenshots at each step
- Manual mouse/keyboard controls
- Session recording and replay

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No display | Headless server | Use Docker (includes Xvfb) or `Xvfb :99 &` |
| xdotool not found | Not installed | `apt install xdotool` or use Docker |
| Screenshot blank | Wrong DISPLAY | Check `DISPLAY` env var matches Xvfb |
| Click misaligned | DPI scaling | Set `RESOLUTION` to match expected screen size |
| Vision API error | Wrong API key | Check `ANTHROPIC_API_KEY` env var |
| Timeout on action | App still loading | Add `await op.wait(2)` before action |
| VNC won't connect | Port not exposed | Check Docker port mapping for 5900/6080 |
| Streamlit error | Wrong Python | Requires Python 3.11+ |

## Related Skills

- `hanzo/hanzo-agent.md` - Agent framework (uses operative for computer use)
- `hanzo/hanzo-mcp.md` - MCP integration
- `hanzo/hanzo-extension.md` - Browser extension (different approach — extends existing browser)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: computer-use, automation, desktop, x11, xdotool, vision
**Prerequisites**: Python 3.11+, Docker (recommended), Claude API
