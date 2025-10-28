# Hanzo Python SDK

**Category**: Hanzo Ecosystem
**Skill Level**: Beginner to Advanced
**Prerequisites**: Python 3.8+, basic understanding of LLM APIs

## Overview

The Hanzo Python SDK provides a unified interface to foundational AI models with automatic routing, privacy-first local inference, and seamless cloud fallback. Instead of managing multiple API clients and implementing routing logic manually, the SDK gives you a single interface that "just works" - automatically selecting the best model based on your requirements and constraints.

**Core Philosophy**: Use high-level SDK abstractions instead of managing individual model APIs.

## Key Features

### 🔒 Privacy-First Architecture
- **Local-first inference**: Automatically uses Hanzo Node when available
- **Selective cloud routing**: Only routes to cloud when absolutely necessary
- **Data sovereignty**: Keep sensitive data on your infrastructure

### 🎯 Automatic Model Routing
- **Complexity analysis**: Routes based on query complexity
- **Cost optimization**: Balances quality vs cost automatically
- **Latency optimization**: Uses fastest available model
- **Quality-focused**: Routes to best model for critical queries

### 🔌 Multi-Provider Support
- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **Local**: Llama 3, Mistral, Qwen (via Hanzo Node)
- **Extensible**: Add custom providers easily

### 📊 Built-in Observability
- **Cost tracking**: Per-request and aggregate costs
- **Latency monitoring**: Response times and bottlenecks
- **Token usage**: Input/output token tracking
- **Cache analytics**: Hit rates and savings

## Installation

```bash
# From PyPI (when published)
pip install hanzo

# From source (current)
cd ~/work/hanzo/sdk/python
pip install -e .

# With all optional dependencies
pip install hanzo[all]
```

## Quick Start

### Basic Usage (Local Inference)

```python
from hanzo import Hanzo

# Initialize with local inference only
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'  # Hanzo Node URL
)

# Simple chat completion
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain quantum computing in simple terms.'}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Hybrid Mode (Local + Cloud with Auto-Routing)

```python
from hanzo import Hanzo

# Initialize with hybrid mode
hanzo = Hanzo(
    inference_mode='hybrid',  # Local + cloud
    auto_route=True,          # Enable automatic routing
    cost_optimize=True,       # Optimize for cost
    
    # Cloud provider credentials (optional - falls back to env vars)
    openai_api_key='sk-...',
    anthropic_api_key='sk-ant-...'
)

# SDK automatically routes based on query complexity
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Write a sorting algorithm'}]
    # No model specified - SDK chooses best option
)

# Check which model was used and why
print(f"Model used: {response.model}")
print(f"Provider: {response.provider}")
print(f"Reasoning: {response.routing_reason}")
print(f"Cost: ${response.cost:.4f}")
```

## Configuration Patterns

### 1. Privacy-First (Local Only)

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080',
    fallback_on_error=False  # Never use cloud, even on errors
)

# All inference happens locally - perfect for sensitive data
response = hanzo.chat.completions.create(
    model='llama-3-8b',
    messages=[{
        'role': 'user',
        'content': 'Process this sensitive patient data: ...'
    }]
)
```

### 2. Cost-Optimized

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    auto_route=True,
    
    # Cost optimization strategy
    routing_strategy='cost_optimized',
    max_cost_per_request=0.01,  # $0.01 per request max
    prefer_local=True,           # Try local first
    
    # Budget limits
    daily_budget=10.0,    # $10/day
    monthly_budget=200.0  # $200/month
)

# SDK prioritizes cheap models unless quality is critical
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Simple classification task'}],
    quality_threshold='good'  # acceptable, good, excellent
)
```

### 3. Latency-Optimized

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    auto_route=True,
    
    # Latency optimization
    routing_strategy='latency_optimized',
    max_latency_ms=500,     # 500ms timeout
    prefer_streaming=True,   # Stream responses when possible
    
    # Pre-warm models
    warm_models=['llama-3-8b', 'gpt-3.5-turbo']
)

# SDK uses fastest available model
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Quick response needed'}],
    stream=True  # Stream for perceived latency reduction
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='')
```

### 4. Quality-Focused

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    auto_route=True,
    
    # Quality optimization
    routing_strategy='quality_focused',
    min_quality_threshold='excellent',
    allow_expensive_models=True,
    
    # Always try best models first
    preferred_models=['gpt-4', 'claude-3-opus', 'llama-3-70b']
)

# SDK uses highest quality model available
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Critical analysis needed'}],
    temperature=0.2  # Low temp for consistent quality
)
```

## Advanced Features

### Custom Routing Logic

```python
from hanzo import Hanzo
from hanzo.routing import RoutingStrategy

class CustomRoutingStrategy(RoutingStrategy):
    """Route based on custom business logic."""
    
    def select_model(self, messages, options):
        # Example: Route by user role
        user_role = options.get('user_role', 'free')
        
        if user_role == 'enterprise':
            return 'gpt-4'  # Best model for enterprise
        elif user_role == 'pro':
            return 'llama-3-70b'  # Good local model
        else:
            return 'llama-3-8b'  # Free tier
    
    def should_use_local(self, messages, options):
        # Always use local for PII data
        content = ' '.join(m['content'] for m in messages)
        return self.contains_pii(content)

hanzo = Hanzo(
    inference_mode='hybrid',
    routing_strategy=CustomRoutingStrategy()
)
```

### Caching and Memoization

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    
    # Enable caching
    enable_cache=True,
    cache_backend='redis',  # redis, memory, disk
    cache_ttl=3600,         # 1 hour
    
    # Semantic caching (cache similar queries)
    semantic_cache=True,
    similarity_threshold=0.95
)

# First call hits the model
response1 = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'What is 2+2?'}]
)

# Second call hits the cache (instant response)
response2 = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'What is 2+2?'}]
)

# Similar query also hits cache
response3 = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'What is two plus two?'}]
)

print(f"Cache hit: {response3.from_cache}")
```

### Batch Processing

```python
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='local')

# Process multiple requests in parallel
requests = [
    {'role': 'user', 'content': f'Summarize document {i}'}
    for i in range(100)
]

# Batch API automatically parallelizes
responses = hanzo.chat.completions.batch_create(
    requests=requests,
    model='llama-3-8b',
    max_concurrent=10  # Process 10 at a time
)

for i, response in enumerate(responses):
    print(f"Document {i}: {response.choices[0].message.content[:50]}...")
```

### Function Calling

```python
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='hybrid')

# Define tools
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current weather for a location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'City name'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit']
                    }
                },
                'required': ['location']
            }
        }
    }
]

response = hanzo.chat.completions.create(
    model='gpt-4',
    messages=[{'role': 'user', 'content': 'What\'s the weather in Paris?'}],
    tools=tools,
    tool_choice='auto'
)

# Check if model wants to call a function
tool_call = response.choices[0].message.tool_calls[0]
if tool_call.function.name == 'get_weather':
    args = json.loads(tool_call.function.arguments)
    print(f"Model wants weather for: {args['location']}")
```

## Integration with Hanzo Ecosystem

### With Hanzo Node (Local Inference)

```python
from hanzo import Hanzo

# SDK automatically discovers local Hanzo Node
hanzo = Hanzo(
    inference_mode='local',
    # No node_url needed - uses service discovery
)

# Check which models are available locally
local_models = hanzo.models.list(provider='local')
for model in local_models:
    print(f"{model.id}: {model.context_length} tokens, {model.parameters}B params")

# Download new model
hanzo.models.download('qwen-2-7b')

# Start inference
response = hanzo.chat.completions.create(
    model='qwen-2-7b',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### With Hanzo MCP (Agentic Workflows)

```python
from hanzo import Hanzo
from hanzo.mcp import MCPClient

# Initialize SDK with MCP integration
hanzo = Hanzo(inference_mode='hybrid')

# Connect to MCP server
mcp = MCPClient('http://localhost:8081')

# Use MCP tools in SDK
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Search the codebase for auth logic'}],
    tools=mcp.get_tools(),  # Automatically import MCP tools
    tool_choice='auto'
)

# SDK can invoke MCP tools automatically
if response.tool_calls:
    results = mcp.execute_tool_calls(response.tool_calls)
    print(results)
```

### With @hanzo/ui (Web Applications)

```python
from hanzo import Hanzo
from fastapi import FastAPI, WebSocket

app = FastAPI()
hanzo = Hanzo(inference_mode='hybrid')

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Receive message from @hanzo/ui frontend
        data = await websocket.receive_json()
        
        # Stream response back
        response = hanzo.chat.completions.create(
            messages=data['messages'],
            stream=True
        )
        
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                await websocket.send_json({'content': content})
```

## Monitoring and Observability

### Built-in Analytics

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    enable_analytics=True
)

# Make some requests
for i in range(100):
    hanzo.chat.completions.create(
        messages=[{'role': 'user', 'content': f'Request {i}'}]
    )

# Get analytics
analytics = hanzo.analytics.get_summary()

print(f"Total requests: {analytics.total_requests}")
print(f"Total cost: ${analytics.total_cost:.2f}")
print(f"Average latency: {analytics.avg_latency_ms:.0f}ms")
print(f"Cache hit rate: {analytics.cache_hit_rate:.1%}")
print(f"Local vs cloud: {analytics.local_ratio:.1%} local")

# Model-specific stats
for model, stats in analytics.by_model.items():
    print(f"{model}: {stats.requests} requests, ${stats.cost:.2f}")
```

### Prometheus Metrics

```python
from hanzo import Hanzo
from prometheus_client import start_http_server

hanzo = Hanzo(
    inference_mode='hybrid',
    enable_metrics=True,
    metrics_port=9090
)

# Metrics automatically exported:
# - hanzo_requests_total
# - hanzo_request_duration_seconds
# - hanzo_tokens_total
# - hanzo_cost_dollars_total
# - hanzo_cache_hits_total
# - hanzo_errors_total

# Start Prometheus endpoint
start_http_server(9090)

# Your application continues...
```

### Custom Logging

```python
from hanzo import Hanzo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hanzo')

hanzo = Hanzo(
    inference_mode='hybrid',
    logger=logger,
    log_requests=True,
    log_responses=True,
    log_routing_decisions=True
)

# Logs include:
# - Request parameters
# - Routing decisions with reasoning
# - Response metadata (tokens, cost, latency)
# - Cache hits/misses
# - Error traces
```

## Error Handling and Resilience

### Automatic Retries

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    
    # Retry configuration
    max_retries=3,
    retry_delay=1.0,         # Exponential backoff starting at 1s
    retry_on=['rate_limit', 'timeout', 'server_error'],
    
    # Fallback configuration
    fallback_on_error=True,
    fallback_order=['local', 'openai', 'anthropic']
)

# SDK automatically retries and falls back
try:
    response = hanzo.chat.completions.create(
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
except hanzo.errors.AllProvidersFailed as e:
    print(f"All providers failed: {e.failures}")
```

### Circuit Breaking

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    
    # Circuit breaker per provider
    circuit_breaker={
        'failure_threshold': 5,   # Open after 5 failures
        'timeout': 60,            # Try again after 60s
        'half_open_requests': 1   # Test with 1 request
    }
)

# SDK tracks provider health and disables failing providers temporarily
```

## Cost Management

### Budget Limits

```python
from hanzo import Hanzo

hanzo = Hanzo(
    inference_mode='hybrid',
    
    # Budget enforcement
    daily_budget=10.0,
    monthly_budget=200.0,
    
    # Budget notifications
    budget_alerts={
        'thresholds': [0.5, 0.8, 0.9, 1.0],
        'webhook': 'https://api.example.com/budget-alert'
    }
)

# SDK raises BudgetExceeded when limits reached
try:
    response = hanzo.chat.completions.create(
        messages=[{'role': 'user', 'content': 'Query'}]
    )
except hanzo.errors.BudgetExceeded as e:
    print(f"Budget exceeded: {e.budget_type} limit of ${e.limit}")
```

### Cost Estimation

```python
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='hybrid')

# Estimate cost before making request
estimate = hanzo.estimate_cost(
    messages=[{'role': 'user', 'content': 'Long prompt...' * 1000}],
    model='gpt-4',
    max_tokens=1000
)

print(f"Estimated cost: ${estimate.cost:.4f}")
print(f"Input tokens: {estimate.input_tokens}")
print(f"Max output tokens: {estimate.max_output_tokens}")

if estimate.cost < 0.10:
    # Proceed with request
    response = hanzo.chat.completions.create(...)
```

## Comparison: Manual vs Hanzo SDK

### Manual Implementation (100+ lines)

```python
# Multiple API clients
from openai import OpenAI
from anthropic import Anthropic
import requests

openai_client = OpenAI(api_key='...')
anthropic_client = Anthropic(api_key='...')

def infer(messages, model=None):
    # Manual routing logic
    if not model:
        # Analyze query complexity (custom implementation needed)
        complexity = analyze_complexity(messages)
        model = select_model_by_complexity(complexity)
    
    # Try local first
    try:
        response = requests.post(
            'http://localhost:8080/v1/chat/completions',
            json={'messages': messages, 'model': model},
            timeout=30
        )
        if response.ok:
            return parse_local_response(response.json())
    except:
        pass  # Fall back to cloud
    
    # Try OpenAI
    if model.startswith('gpt'):
        try:
            return openai_client.chat.completions.create(
                messages=messages,
                model=model
            )
        except:
            pass
    
    # Try Anthropic
    if model.startswith('claude'):
        try:
            return anthropic_client.messages.create(
                messages=messages,
                model=model,
                max_tokens=1000
            )
        except:
            pass
    
    raise Exception("All providers failed")

# Manual cost tracking (custom implementation)
# Manual caching (custom implementation)
# Manual retry logic (custom implementation)
# Manual metrics (custom implementation)
```

### Hanzo SDK (3 lines)

```python
from hanzo import Hanzo

hanzo = Hanzo(inference_mode='hybrid', auto_route=True)

response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Query'}]
)
# Automatic routing, cost tracking, caching, retries, metrics included
```

### Feature Comparison

| Feature | Manual | Hanzo SDK |
|---------|--------|-----------|
| Setup | 50+ lines | 2 lines |
| Provider Management | Manual | Automatic |
| Routing Logic | Custom implementation | Built-in + customizable |
| Local Inference | Manual HTTP | Automatic |
| Cost Tracking | Custom implementation | Built-in |
| Caching | Custom implementation | Built-in |
| Retry Logic | Custom implementation | Built-in |
| Circuit Breaker | Custom implementation | Built-in |
| Metrics | Custom implementation | Built-in |
| Budget Management | Not included | Built-in |
| Streaming | Partial | Full support |
| Function Calling | Manual parsing | Automatic |
| MCP Integration | Not included | Native |

**Lines of Code**: ~500 (manual) vs ~10 (Hanzo SDK)
**Features**: 50% coverage (manual) vs 100% (Hanzo SDK)
**Maintenance**: High (manual) vs Low (Hanzo SDK)

## Best Practices

### 1. Use Hybrid Mode for Production

```python
# ✅ Good - flexible and resilient
hanzo = Hanzo(
    inference_mode='hybrid',
    prefer_local=True,
    fallback_on_error=True
)

# ❌ Avoid - single point of failure
hanzo = Hanzo(inference_mode='cloud_only')
```

### 2. Enable Caching

```python
# ✅ Good - reduces cost and latency
hanzo = Hanzo(
    enable_cache=True,
    cache_backend='redis',
    semantic_cache=True
)

# ❌ Avoid - wastes money on duplicate requests
hanzo = Hanzo(enable_cache=False)
```

### 3. Set Budget Limits

```python
# ✅ Good - prevents runaway costs
hanzo = Hanzo(
    daily_budget=10.0,
    max_cost_per_request=0.10
)

# ❌ Avoid - no cost control
hanzo = Hanzo()  # No budget limits
```

### 4. Use Streaming for Long Responses

```python
# ✅ Good - better UX
response = hanzo.chat.completions.create(
    messages=[...],
    stream=True
)
for chunk in response:
    display(chunk.choices[0].delta.content)

# ❌ Avoid - wait for entire response
response = hanzo.chat.completions.create(messages=[...])
display(response.choices[0].message.content)
```

### 5. Monitor and Alert

```python
# ✅ Good - proactive monitoring
hanzo = Hanzo(
    enable_metrics=True,
    enable_analytics=True,
    budget_alerts={...}
)

# ❌ Avoid - reactive troubleshooting
hanzo = Hanzo()  # No monitoring
```

## Troubleshooting

### SDK Not Finding Local Hanzo Node

```python
# Check if Hanzo Node is running
import requests
try:
    response = requests.get('http://localhost:8080/health')
    print(f"Hanzo Node status: {response.json()}")
except:
    print("Hanzo Node not running. Start with: hanzo-node start")

# Specify URL explicitly
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'
)
```

### High Latency

```python
# Enable latency optimization
hanzo = Hanzo(
    inference_mode='hybrid',
    routing_strategy='latency_optimized',
    prefer_streaming=True,
    warm_models=['llama-3-8b']  # Pre-warm frequently used models
)

# Use smaller models for simple queries
response = hanzo.chat.completions.create(
    messages=[...],
    max_tokens=100,  # Limit output length
    temperature=0    # Deterministic = faster
)
```

### High Costs

```python
# Enable cost optimization
hanzo = Hanzo(
    inference_mode='hybrid',
    routing_strategy='cost_optimized',
    prefer_local=True,
    max_cost_per_request=0.01,
    daily_budget=5.0
)

# Use local models when possible
hanzo.set_default_model('llama-3-8b')

# Enable aggressive caching
hanzo.configure_cache(
    semantic_cache=True,
    cache_ttl=86400  # 24 hours
)
```

## Next Steps

1. **Read Hanzo Node Documentation** - Understanding the local inference layer
2. **Explore Hanzo MCP** - Agentic workflows with the SDK
3. **Check @hanzo/ui** - Building UI with Python SDK backend
4. **See llm-model-routing.md** - Advanced routing patterns

## Related Skills

- **hanzo-node.md** - Local AI inference infrastructure
- **hanzo-mcp.md** - Model Context Protocol integration
- **hanzo-ui.md** - Frontend components
- **llm-model-routing.md** - Model selection strategies
- **dspy-setup.md** - DSPy integration with Hanzo SDK

---

**Remember**: Use the Hanzo SDK for unified AI model access with automatic routing, caching, and cost management - don't implement these features manually.
