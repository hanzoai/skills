# Hanzo Agent SDK - Multi-Agent Orchestration Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-chat.md`, `hanzo/python-sdk.md`

## Overview

Hanzo Agent SDK is a **Python multi-agent framework** with OpenAI-compatible APIs. Build autonomous agents that plan, use tools, coordinate with other agents, and maintain memory. Designed for production agentic workflows.

### Why Hanzo Agent SDK?

- **Multi-agent**: Orchestrate teams of specialized agents
- **OpenAI compatible**: Drop-in replacement for OpenAI Agents SDK
- **Tool use**: Native MCP + custom tool integration
- **Memory**: Persistent agent memory across sessions
- **Planning**: Built-in planning and reasoning capabilities

### Repo

`hanzoai/agent`. Local: ``github.com/hanzoai/agent``.

## When to use

- Building autonomous AI agents
- Multi-agent coordination (supervisor, swarm patterns)
- Tool-using agents with MCP integration
- Production agentic workflows with memory
- Replacing OpenAI Agents SDK with self-hosted option

## Hard requirements

1. **Python 3.11+** with uv
2. **HANZO_API_KEY** or **OPENAI_API_KEY** for LLM access
3. `uv sync --all-extras` for full installation

## Quick reference

| Item | Value |
|------|-------|
| Install | `uv add hanzo-agent` |
| Repo | `github.com/hanzoai/agent` |
| Setup | `uv sync --all-extras` |
| Test | `uv run pytest -v` |
| Docs | `https://hanzo.ai/docs/agent` |

## One-file quickstart

### Simple Agent

```python
from hanzo_agent import Agent, Runner

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant that answers questions concisely.",
    model="zen-70b",
)

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)  # "Paris"
```

### Agent with Tools

```python
from hanzo_agent import Agent, Runner, function_tool

@function_tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"72°F and sunny in {location}"

@function_tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Top result for '{query}': ..."

agent = Agent(
    name="research-assistant",
    instructions="Help users with research. Use tools when needed.",
    model="zen-70b",
    tools=[get_weather, search_web],
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
```

### Multi-Agent (Supervisor Pattern)

```python
from hanzo_agent import Agent, Runner

coder = Agent(
    name="coder",
    instructions="Write clean, tested Python code.",
    model="zen-70b",
)

reviewer = Agent(
    name="reviewer",
    instructions="Review code for bugs, security issues, and style.",
    model="zen-70b",
)

supervisor = Agent(
    name="supervisor",
    instructions="""
    You coordinate a coding team:
    1. Delegate coding tasks to the coder agent
    2. Send code to the reviewer agent for review
    3. Iterate until code passes review
    """,
    model="zen-70b",
    handoffs=[coder, reviewer],
)

result = Runner.run_sync(supervisor, "Write a REST API for a todo app")
```

### Streaming

```python
import asyncio
from hanzo_agent import Agent, Runner

agent = Agent(name="writer", instructions="Write creatively.", model="zen-70b")

async def main():
    async for event in Runner.run_streamed(agent, "Write a haiku about code"):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

## Core Concepts

### Agent Architecture

```
┌───────────────────┐
│    Supervisor      │
│  (orchestrates)    │
├─────────┬─────────┤
│  Agent A │ Agent B │
│  (tools) │ (tools) │
├─────────┴─────────┤
│    Tool Registry   │
│  MCP + Custom      │
├───────────────────┤
│    Memory Store    │
│  (persistent)      │
├───────────────────┤
│    LLM Backend    │
│  (api.hanzo.ai)   │
└───────────────────┘
```

### Handoffs

Agents can delegate to other agents:

```python
agent_a = Agent(name="a", instructions="...", handoffs=[agent_b, agent_c])
# When agent_a decides it needs agent_b's expertise, it hands off automatically
```

### Guardrails

```python
from hanzo_agent import Agent, InputGuardrail

async def check_injection(input_text: str) -> bool:
    """Return True if input is safe."""
    return "ignore previous instructions" not in input_text.lower()

agent = Agent(
    name="safe-agent",
    instructions="...",
    input_guardrails=[InputGuardrail(fn=check_injection)],
)
```

### MCP Integration

```python
from hanzo_agent import Agent
from hanzo_agent.mcp import MCPServerStdio

mcp_server = MCPServerStdio(
    command="npx",
    args=["-y", "@hanzo/mcp"],
)

agent = Agent(
    name="mcp-agent",
    instructions="Use MCP tools to accomplish tasks.",
    mcp_servers=[mcp_server],
)
```

## Development

```bash
# Clone from github.com/hanzoai first
cd <project>
uv sync --all-extras   # Install all dependencies
uv run pytest -v       # Run tests
uv run ruff format .   # Format code
uv run mypy .          # Type check
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error | Missing extras | `uv sync --all-extras` |
| LLM timeout | Model too slow | Use faster model or increase timeout |
| Handoff loop | Agents delegating back and forth | Add clear stopping criteria |
| Memory overflow | Too much context | Implement context windowing |

## Related Skills

- `hanzo/hanzo-mcp.md` - MCP tools integration
- `hanzo/hanzo-chat.md` - LLM API backend
- `hanzo/python-sdk.md` - Python client library
- `hanzo/hanzo-operative.md` - Computer use agent

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: agents, multi-agent, orchestration, tools
**Prerequisites**: Python, async/await, LLM concepts
