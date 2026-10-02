---
name: hanzo-repl
description: "Interactive AI REPL for Hanzo — Claude Code-like terminal experience with IPython, TUI, voice, and MCP tool integration."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-repl",
              "label": "Install Hanzo REPL (pip)",
            },
          ],
      },
  }
---

# Hanzo REPL — Interactive AI Terminal

`pip install hanzo-repl`

Claude Code-like interactive REPL with three interfaces, MCP tool access, and optional voice I/O.

## Quick Start

```bash
# Basic REPL
hanzo-repl

# IPython-enhanced REPL
hanzo-repl-ipython

# Textual TUI (rich terminal UI)
hanzo-repl-tui
```

## Options

```bash
hanzo-repl --mode=ipython    # IPython mode
hanzo-repl --model=gpt-4     # Select LLM model
hanzo-repl --debug            # Debug output
```

## Features

- **Three interfaces**: Basic REPL, IPython, Textual TUI
- **MCP tools**: Full access to 260+ hanzo-mcp tools
- **Multi-language**: Execute code in Python, JavaScript, TypeScript, Bash
- **Voice I/O**: Speech recognition and TTS (optional)

## Voice Support

```bash
pip install "hanzo-repl[voice]"

# Enables speech-to-text input and text-to-speech output
hanzo-repl --voice
```

## Python API

```python
from hanzo_repl import HanzoREPL

repl = HanzoREPL(model="gpt-4")
await repl.start()
```
