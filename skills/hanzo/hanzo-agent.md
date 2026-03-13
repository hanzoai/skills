# Hanzo Agent SDK - Multi-Agent Orchestration Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mcp.md`, `hanzo/hanzo-chat.md`, `hanzo/python-sdk.md`

## Overview

Hanzo Agent SDK is a **Python multi-agent framework** — fork of OpenAI's Agents SDK with Hanzo AI integration. Build autonomous agents that plan, use tools, coordinate with other agents, and maintain memory. Designed for production agentic workflows.

### Why Hanzo Agent SDK?

- **Multi-agent**: Orchestrate teams of specialized agents
- **OpenAI compatible**: Fork of OpenAI Agents SDK, same patterns
- **Tool use**: Native MCP + custom function tools
- **Handoffs**: Agents delegate to specialists automatically
- **Guardrails**: Input/output validation and safety checks
- **Tracing**: Built-in observability for agent runs
- **Streaming**: Real-time token-by-token output

### OSS Base

Fork of **OpenAI Agents SDK**. Repo: `hanzoai/agent`.

**NOTE**: PyPI package name collision — `hanzoai/agent` publishes `hanzoai` v0.0.4 on PyPI, while `hanzoai/python-sdk` publishes `hanzoai` v2.2.0. Install as `hanzo-agent` to avoid conflict.

## When to use

- Building autonomous AI agents
- Multi-agent coordination (supervisor, swarm, pipeline patterns)
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
| PyPI | `hanzo-agent` (NOT `hanzoai` — name collision) |
| Version | 0.0.4 |
| Repo | `github.com/hanzoai/agent` |
| Upstream | OpenAI Agents SDK |
| Setup | `uv sync --all-extras` |
| Test | `uv run pytest -v` |
| Format | `uv run ruff format .` |
| Type check | `uv run mypy .` |
| License | MIT |

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

@function_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))  # Use safe eval in production

agent = Agent(
    name="research-assistant",
    instructions="Help users with research. Use tools when needed.",
    model="zen-70b",
    tools=[get_weather, search_web, calculate],
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
```

### Multi-Agent (Handoffs)

```python
from hanzo_agent import Agent, Runner

coder = Agent(
    name="coder",
    instructions="Write clean, tested Python code. Return only code.",
    model="zen-70b",
)

reviewer = Agent(
    name="reviewer",
    instructions="Review code for bugs, security issues, and style. Be thorough.",
    model="zen-70b",
)

supervisor = Agent(
    name="supervisor",
    instructions="""You coordinate a coding team:
    1. Delegate coding tasks to the coder
    2. Send code to the reviewer for review
    3. Iterate until code passes review
    Return the final approved code.""",
    model="zen-70b",
    handoffs=[coder, reviewer],
)

result = Runner.run_sync(supervisor, "Write a REST API for a todo app with FastAPI")
print(result.final_output)
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

### Guardrails

```python
from hanzo_agent import Agent, Runner, InputGuardrail, OutputGuardrail, GuardrailFunctionOutput

async def check_injection(ctx, agent, input_text: str) -> GuardrailFunctionOutput:
    """Block prompt injection attempts."""
    is_safe = "ignore previous instructions" not in input_text.lower()
    return GuardrailFunctionOutput(
        output_info={"safe": is_safe},
        tripwire_triggered=not is_safe,
    )

async def check_pii(ctx, agent, output_text: str) -> GuardrailFunctionOutput:
    """Block PII in output."""
    import re
    has_ssn = bool(re.search(r'\d{3}-\d{2}-\d{4}', output_text))
    return GuardrailFunctionOutput(
        output_info={"has_pii": has_ssn},
        tripwire_triggered=has_ssn,
    )

agent = Agent(
    name="safe-agent",
    instructions="Answer questions helpfully.",
    model="zen-70b",
    input_guardrails=[InputGuardrail(guardrail_function=check_injection)],
    output_guardrails=[OutputGuardrail(guardrail_function=check_pii)],
)
```

### MCP Integration

```python
from hanzo_agent import Agent, Runner
from hanzo_agent.mcp import MCPServerStdio, MCPServerSse

# Stdio transport (local process)
mcp_local = MCPServerStdio(
    command="npx",
    args=["-y", "@hanzo/mcp"],
)

# SSE transport (remote server)
mcp_remote = MCPServerSse(
    url="https://mcp.example.com/sse",
    headers={"Authorization": "Bearer token"},
)

agent = Agent(
    name="mcp-agent",
    instructions="Use available MCP tools to accomplish tasks.",
    model="zen-70b",
    mcp_servers=[mcp_local, mcp_remote],
)

async def main():
    async with mcp_local, mcp_remote:
        result = await Runner.run(agent, "Search for recent AI news")
        print(result.final_output)
```

### Tracing

```python
from hanzo_agent import Agent, Runner
from hanzo_agent.tracing import TracingProcessor, Span

class CustomTracer(TracingProcessor):
    def on_span_start(self, span: Span):
        print(f"Start: {span.name}")

    def on_span_end(self, span: Span):
        print(f"End: {span.name} ({span.duration_ms}ms)")

# Register globally
from hanzo_agent.tracing import set_tracing_processor
set_tracing_processor(CustomTracer())

# All agent runs now traced automatically
result = Runner.run_sync(agent, "Hello")
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
│  MCP + Functions   │
├───────────────────┤
│    Guardrails      │
│  Input + Output    │
├───────────────────┤
│    Tracing Layer   │
│  (spans, events)   │
├───────────────────┤
│    LLM Backend    │
│  (api.hanzo.ai)   │
└───────────────────┘
```

### Orchestration Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Single agent** | One agent, one task | Simple Q&A, single-tool tasks |
| **Handoff** | Agent delegates to specialist | Domain-specific expertise needed |
| **Supervisor** | Manager coordinates team | Complex multi-step tasks |
| **Pipeline** | Sequential agent chain | Transform/refine/review flows |
| **Swarm** | Peer agents collaborate | Exploration, brainstorming |

### Agent Configuration

```python
agent = Agent(
    name="my-agent",                    # Required: unique name
    instructions="...",                  # System prompt
    model="zen-70b",                     # LLM model
    tools=[tool1, tool2],               # Function tools
    mcp_servers=[server1],              # MCP tool servers
    handoffs=[agent_b, agent_c],        # Agents to delegate to
    input_guardrails=[guard1],          # Input validation
    output_guardrails=[guard2],         # Output validation
    output_type=MyPydanticModel,        # Structured output (Pydantic)
    model_settings=ModelSettings(       # LLM settings
        temperature=0.7,
        max_tokens=4096,
    ),
)
```

## Development

```bash
git clone https://github.com/hanzoai/agent.git
cd agent
uv sync --all-extras    # Install all dependencies
uv run pytest -v        # Run tests
uv run ruff format .    # Format code
uv run ruff check .     # Lint
uv run mypy .           # Type check
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error | Missing extras | `uv sync --all-extras` |
| PyPI collision | Wrong `hanzoai` package | Install `hanzo-agent` specifically |
| LLM timeout | Model too slow | Use faster model or increase timeout |
| Handoff loop | Circular delegation | Add clear stopping criteria in instructions |
| Memory overflow | Too much context | Implement context windowing |
| MCP connection fail | Server not running | Check MCP server process |

## Related Skills

- `hanzo/hanzo-mcp.md` - MCP tools integration (13 HIP-0300 tools)
- `hanzo/hanzo-chat.md` - LLM API backend
- `hanzo/python-sdk.md` - Python client library
- `hanzo/hanzo-operative.md` - Computer use agent

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: agents, multi-agent, orchestration, tools, openai-agents
**Prerequisites**: Python 3.11+, async/await, LLM concepts
