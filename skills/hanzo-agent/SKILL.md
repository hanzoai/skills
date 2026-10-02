---
name: hanzo-agent
description: "Build multi-agent AI systems with Hanzo Agent SDK. Create agents with tools, handoffs, structured outputs, networks, and orchestration workflows."
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
              "package": "hanzo-agent",
              "label": "Install Hanzo Agent SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo Agent — Multi-Agent Framework

`pip install hanzo-agent`

## Quick Start

```python
from hanzoai import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"

agent = Agent(
    name="Assistant",
    instructions="You help users with weather questions.",
    tools=[get_weather],
    model="gpt-4"
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

## Async Usage

```python
import asyncio
from hanzoai import Agent, Runner

agent = Agent(name="Helper", instructions="Be helpful")

async def main():
    result = await Runner.run(agent, "Hello!")
    print(result.final_output)

asyncio.run(main())
```

## Tools

### Function Tools

```python
from hanzoai import function_tool

@function_tool
def search_docs(query: str, limit: int = 10) -> str:
    """Search documentation."""
    return f"Found {limit} results for: {query}"
```

### MCP Tools

```python
from hanzoai.mcp import MCPServerStdio

server = MCPServerStdio(command="hanzo-mcp", args=["--tools", "fs,shell"])

agent = Agent(
    name="Coder",
    instructions="Write code",
    mcp_servers=[server]
)
```

## Agent Handoffs

```python
researcher = Agent(
    name="Researcher",
    instructions="Research topics thoroughly",
    tools=[search_tool]
)

writer = Agent(
    name="Writer",
    instructions="Write clear, engaging content",
    handoffs=[researcher]  # Can delegate to researcher
)

result = await Runner.run(writer, "Write about quantum computing")
```

## Structured Output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

agent = Agent(
    name="Summarizer",
    instructions="Summarize the input",
    output_type=Summary
)

result = await Runner.run(agent, "Long article text here...")
summary: Summary = result.final_output
print(summary.title, summary.key_points)
```

## Multi-Agent Networks

```python
from hanzoai import Agent, Runner

triage = Agent(
    name="Triage",
    instructions="Route to the right specialist",
    handoffs=["coder", "writer", "analyst"]
)

coder = Agent(name="Coder", instructions="Write code")
writer = Agent(name="Writer", instructions="Write prose")
analyst = Agent(name="Analyst", instructions="Analyze data")

result = await Runner.run(
    triage,
    "Write a Python script to sort a list",
    agents=[coder, writer, analyst]
)
```

## Guardrails

```python
from hanzoai import Agent, InputGuardrail, GuardrailFunctionOutput

@InputGuardrail
async def no_pii(ctx, agent, input):
    """Block requests containing PII."""
    has_pii = check_for_pii(input)
    return GuardrailFunctionOutput(
        output_info={"has_pii": has_pii},
        tripwire_triggered=has_pii
    )

agent = Agent(
    name="Safe Agent",
    instructions="Help users",
    input_guardrails=[no_pii]
)
```

## Tracing & Observability

```python
from hanzoai import trace

@trace
async def my_workflow():
    result = await Runner.run(agent, "input")
    return result

# Integrations: Logfire, AgentOps, Braintrust, Scorecard
```

## Environment Variables

```bash
HANZO_API_KEY=sk-...          # Required for LLM calls
HANZO_BASE_URL=...            # API endpoint
OPENAI_API_KEY=sk-...         # For OpenAI models
ANTHROPIC_API_KEY=sk-ant-...  # For Claude models
```
