---
name: ml-multi-model-orchestration
description: Coordinating multiple LLMs in pipelines, ensembles, and cascading workflows
---

# Multi-Model Orchestration

**Scope**: Pipeline patterns, ensemble methods, specialist routing, cascading, context management, error handling
**Lines**: ~400
**Last Updated**: 2025-10-26

## When to Use This Skill

Activate this skill when:
- Building workflows that combine strengths of multiple models
- Implementing consensus mechanisms for critical decisions
- Cascading from fast models to accurate models based on confidence
- Designing specialist systems where different models handle different subtasks
- Chaining models in sequential pipelines (planning -> execution -> review)
- Managing context and state across multi-model calls
- Handling failures with fallback models
- Optimizing for both quality and cost through model mixing

## Core Concepts

### Orchestration Patterns

**1. Pipeline (Sequential)**:
- Models execute in sequence, each using previous output
- Example: Planner -> Coder -> Reviewer -> Refiner
- **Benefit**: Specialization per stage
- **Cost**: Cumulative latency

**2. Ensemble (Parallel)**:
- Multiple models process same input, outputs combined
- Example: GPT-4o + Claude + Gemini -> Vote or merge
- **Benefit**: Higher accuracy, reduced bias
- **Cost**: 3x+ API costs

**3. Specialist Routing**:
- Route to model specialized for subtask
- Example: Code->Claude, Math->DeepSeek R1, General->GPT-4o
- **Benefit**: Optimal model per task
- **Cost**: Routing overhead

**4. Cascade (Conditional)**:
- Start with cheap/fast model, escalate if needed
- Example: GPT-4o mini -> (if unsure) -> GPT-4o
- **Benefit**: Cost savings with quality backstop
- **Cost**: Complexity in confidence detection

**5. Hybrid (Combined)**:
- Mix patterns: Pipeline + Ensemble, Cascade + Specialist
- Example: Route by topic -> Ensemble -> Pipeline
- **Benefit**: Maximum flexibility
- **Cost**: Complex orchestration logic

### Context Management

**Challenges**:
- Passing large context between models (token costs)
- Maintaining conversation state across calls
- Avoiding context window overflow
- Preserving semantic coherence

**Solutions**:
- Prompt chaining (minimal context passing)
- Shared memory/cache layer (Redis, database)
- Summarization between stages (lossy but efficient)
- Context compression (extract key information)

### Error Handling Strategies

**Fallback Chains**:
```
Primary model -> (on error) -> Fallback 1 -> Fallback 2 -> Default response
```

**Retry with Backoff**:
```
Attempt 1 (immediate) -> Attempt 2 (+1s) -> Attempt 3 (+3s) -> Fail gracefully
```

**Graceful Degradation**:
```
Try ensemble -> (on partial failure) -> Use successful responses -> (on total failure) -> Single model
```

### Arize Phoenix Multi-Model Tracing

**Capabilities**:
- Span-level tracing across model calls
- Routing decision visibility
- Cost/latency breakdown per model
- Error rate tracking per model
- A/B test comparison

**Span Structure**:
```
Root Span (orchestration)
├─ Span: routing_decision
├─ Span: model_call_1 (claude-3.5-sonnet)
│  ├─ Attribute: tokens_used
│  ├─ Attribute: latency_ms
│  └─ Attribute: cost_usd
├─ Span: model_call_2 (gpt-4o)
└─ Span: result_aggregation
```

---

## Implementation Patterns

### Pipeline Pattern: Plan-Code-Review

```python
import openai
from typing import Dict, Any

class CodeGenerationPipeline:
    """Multi-stage pipeline: planning -> coding -> review"""

    def __init__(self):
        self.planner_model = "gpt-4o"
        self.coder_model = "claude-3-5-sonnet-20241022"
        self.reviewer_model = "gpt-4o"

    def plan(self, requirement: str) -> str:
        """Stage 1: Planning with GPT-4o"""
        response = openai.ChatCompletion.create(
            model=self.planner_model,
            messages=[{
                "role": "system",
                "content": "You are a software architect. Create a detailed implementation plan."
            }, {
                "role": "user",
                "content": f"Plan implementation for: {requirement}"
            }],
            temperature=0.7
        )
        return response.choices[0].message.content

    def code(self, plan: str) -> str:
        """Stage 2: Coding with Claude (best at code)"""
        response = openai.ChatCompletion.create(
            model=self.coder_model,
            messages=[{
                "role": "system",
                "content": "You are an expert programmer. Implement the plan precisely."
            }, {
                "role": "user",
                "content": f"Implement this plan:\n\n{plan}"
            }],
            temperature=0.2
        )
        return response.choices[0].message.content

    def review(self, code: str, plan: str) -> Dict[str, Any]:
        """Stage 3: Review with GPT-4o"""
        response = openai.ChatCompletion.create(
            model=self.reviewer_model,
            messages=[{
                "role": "system",
                "content": "You are a code reviewer. Assess quality, correctness, and adherence to plan."
            }, {
                "role": "user",
                "content": f"Review this code against the plan.\n\nPlan:\n{plan}\n\nCode:\n{code}"
            }],
            temperature=0.3
        )

        review_text = response.choices[0].message.content

        # Extract approval (simple heuristic)
        approved = "approved" in review_text.lower() or "looks good" in review_text.lower()

        return {
            "approved": approved,
            "review": review_text,
            "code": code
        }

    def execute(self, requirement: str) -> Dict[str, Any]:
        """Execute full pipeline"""
        print("Stage 1: Planning...")
        plan = self.plan(requirement)

        print("Stage 2: Coding...")
        code = self.code(plan)

        print("Stage 3: Review...")
        result = self.review(code, plan)

        return {
            "requirement": requirement,
            "plan": plan,
            "code": result["code"],
            "review": result["review"],
            "approved": result["approved"]
        }

# Usage
pipeline = CodeGenerationPipeline()
result = pipeline.execute("Build a REST API for user authentication with JWT")

print(f"Approved: {result['approved']}")
print(f"Code:\n{result['code'][:500]}...")
```

### Ensemble Pattern: Consensus Voting

```python
import asyncio
from typing import List, Dict
from collections import Counter

class EnsembleOrchestrator:
    """Run multiple models in parallel and vote on results"""

    def __init__(self, models: List[str] = None):
        self.models = models or [
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "gemini-2.5-pro"
        ]

    async def call_model(self, model: str, prompt: str) -> str:
        """Call single model asynchronously"""
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    async def ensemble_vote(
        self,
        prompt: str,
        extract_answer_fn = None
    ) -> Dict[str, Any]:
        """Run ensemble and vote on answers"""

        # Call all models in parallel
        tasks = [
            self.call_model(model, prompt)
            for model in self.models
        ]
        responses = await asyncio.gather(*tasks)

        # Extract answers (default: use full response)
        if extract_answer_fn is None:
            extract_answer_fn = lambda x: x.strip().lower()

        answers = [extract_answer_fn(r) for r in responses]

        # Vote
        vote_counts = Counter(answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts[majority_answer]

        # Check consensus (>50%)
        consensus = majority_count > len(self.models) / 2

        return {
            "models": self.models,
            "responses": responses,
            "answers": answers,
            "majority_answer": majority_answer,
            "vote_counts": dict(vote_counts),
            "consensus": consensus,
            "confidence": majority_count / len(self.models)
        }

# Usage
async def main():
    ensemble = EnsembleOrchestrator()

    # Multiple choice question
    prompt = """
    What is the capital of Australia?
    A) Sydney
    B) Melbourne
    C) Canberra
    D) Brisbane

    Answer with just the letter (A, B, C, or D).
    """

    def extract_letter(response: str) -> str:
        # Extract first letter A-D
        import re
        match = re.search(r'[ABCD]', response.upper())
        return match.group(0) if match else ""

    result = await ensemble.ensemble_vote(prompt, extract_answer_fn=extract_letter)

    print(f"Majority answer: {result['majority_answer']}")
    print(f"Consensus: {result['consensus']} ({result['confidence']:.0%})")
    print(f"Vote counts: {result['vote_counts']}")

asyncio.run(main())
```

### Cascade Pattern: Confidence-Based Escalation

```python
import re
from typing import Optional, Dict, Any

class CascadeOrchestrator:
    """Start with cheap model, escalate to expensive if uncertain"""

    def __init__(
        self,
        fast_model: str = "gpt-4o-mini",
        strong_model: str = "gpt-4o",
        confidence_threshold: float = 0.8
    ):
        self.fast = fast_model
        self.strong = strong_model
        self.threshold = confidence_threshold

    def extract_confidence(self, response: str) -> Optional[float]:
        """Extract confidence from response (if model provides it)"""
        # Look for patterns like "Confidence: 85%" or "(confidence: 0.85)"
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)\%?",
            r"\(confidence[:\s]+(\d+(?:\.\d+)?)\)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                conf = float(match.group(1))
                return conf / 100 if conf > 1 else conf

        return None

    def call_with_confidence(
        self,
        model: str,
        prompt: str,
        request_confidence: bool = True
    ) -> Dict[str, Any]:
        """Call model and extract confidence"""

        if request_confidence:
            prompt += "\n\nProvide your confidence level (0-100%) at the end."

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        text = response.choices[0].message.content
        confidence = self.extract_confidence(text)

        return {
            "model": model,
            "response": text,
            "confidence": confidence,
            "tokens": response.usage.total_tokens
        }

    def cascade(self, prompt: str) -> Dict[str, Any]:
        """Execute cascade: fast -> strong if needed"""

        print(f"Trying fast model: {self.fast}")
        fast_result = self.call_with_confidence(self.fast, prompt)

        # Check if confident enough
        if fast_result["confidence"] and fast_result["confidence"] >= self.threshold:
            print(f"Fast model confident ({fast_result['confidence']:.0%}), using result")
            return {
                **fast_result,
                "escalated": False,
                "total_tokens": fast_result["tokens"]
            }

        # Escalate to strong model
        print(f"Low confidence ({fast_result['confidence']:.0%}), escalating to {self.strong}")
        strong_result = self.call_with_confidence(self.strong, prompt)

        return {
            **strong_result,
            "escalated": True,
            "fast_result": fast_result,
            "total_tokens": fast_result["tokens"] + strong_result["tokens"]
        }

# Usage
cascade = CascadeOrchestrator(confidence_threshold=0.8)

result = cascade.cascade("What are the ethical implications of AGI development?")

print(f"Final model: {result['model']}")
print(f"Escalated: {result['escalated']}")
print(f"Total tokens: {result['total_tokens']}")
```

### Specialist Routing with Context Passing

```python
from typing import List, Dict, Any

class SpecialistOrchestrator:
    """Route subtasks to specialist models"""

    def __init__(self):
        self.specialists = {
            "code": "claude-3-5-sonnet-20241022",
            "math": "deepseek-r1-distilled-70b",
            "general": "gpt-4o",
            "fast": "gemini-2.5-flash"
        }
        self.conversation_history: List[Dict] = []

    def detect_task_type(self, query: str) -> str:
        """Simple task detection (use semantic routing for production)"""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["code", "function", "implement", "debug"]):
            return "code"
        if any(kw in query_lower for kw in ["calculate", "solve", "math", "equation"]):
            return "math"
        if len(query) < 50:
            return "fast"

        return "general"

    def call_specialist(
        self,
        task_type: str,
        query: str,
        include_history: bool = False
    ) -> Dict[str, Any]:
        """Route to specialist and maintain context"""

        model = self.specialists[task_type]

        # Build messages with optional history
        messages = []
        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": query})

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "task_type": task_type,
            "model": model,
            "query": query,
            "answer": answer,
            "tokens": response.usage.total_tokens
        }

    def multi_turn_conversation(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Handle multi-turn conversation with specialist routing"""

        results = []
        for query in queries:
            task_type = self.detect_task_type(query)
            print(f"Query: {query[:60]}... -> {task_type} ({self.specialists[task_type]})")

            result = self.call_specialist(
                task_type,
                query,
                include_history=True  # Maintain context
            )
            results.append(result)

        return results

# Usage
orchestrator = SpecialistOrchestrator()

conversation = [
    "Write a function to calculate Fibonacci numbers",
    "Now optimize it for large inputs",
    "What's the time complexity?",
    "Calculate the 50th Fibonacci number using your implementation"
]

results = orchestrator.multi_turn_conversation(conversation)

for r in results:
    print(f"\n[{r['task_type']}] {r['model']}")
    print(f"Answer: {r['answer'][:200]}...")
```

### Phoenix Observability Integration

```python
import phoenix as px
from phoenix.trace import using_project
from opentelemetry import trace

# Start Phoenix
px.launch_app()

tracer = trace.get_tracer(__name__)

@using_project("multi-model-orchestration")
def orchestrated_pipeline(requirement: str):
    """Pipeline with full observability"""

    with tracer.start_as_current_span("pipeline") as root_span:
        root_span.set_attribute("requirement", requirement)

        # Stage 1: Planning
        with tracer.start_as_current_span("stage_planning") as plan_span:
            plan_span.set_attribute("model", "gpt-4o")

            plan_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"Plan: {requirement}"
                }]
            )

            plan = plan_response.choices[0].message.content
            plan_span.set_attribute("tokens", plan_response.usage.total_tokens)
            plan_span.set_attribute("cost_usd", calculate_cost(plan_response, "gpt-4o"))

        # Stage 2: Coding
        with tracer.start_as_current_span("stage_coding") as code_span:
            code_span.set_attribute("model", "claude-3-5-sonnet")

            code_response = openai.ChatCompletion.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": f"Implement:\n{plan}"
                }]
            )

            code = code_response.choices[0].message.content
            code_span.set_attribute("tokens", code_response.usage.total_tokens)
            code_span.set_attribute("cost_usd", calculate_cost(code_response, "claude-3-5-sonnet"))

        # Stage 3: Review
        with tracer.start_as_current_span("stage_review") as review_span:
            review_span.set_attribute("model", "gpt-4o")

            review_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"Review:\nPlan: {plan}\nCode: {code}"
                }]
            )

            review = review_response.choices[0].message.content
            review_span.set_attribute("tokens", review_response.usage.total_tokens)
            review_span.set_attribute("cost_usd", calculate_cost(review_response, "gpt-4o"))

        # Calculate total metrics
        total_cost = (
            calculate_cost(plan_response, "gpt-4o") +
            calculate_cost(code_response, "claude-3-5-sonnet") +
            calculate_cost(review_response, "gpt-4o")
        )
        total_tokens = (
            plan_response.usage.total_tokens +
            code_response.usage.total_tokens +
            review_response.usage.total_tokens
        )

        root_span.set_attribute("total_cost_usd", total_cost)
        root_span.set_attribute("total_tokens", total_tokens)

        return {
            "plan": plan,
            "code": code,
            "review": review,
            "cost": total_cost,
            "tokens": total_tokens
        }

def calculate_cost(response, model: str) -> float:
    """Calculate cost based on token usage"""
    pricing = {
        "gpt-4o": {"input": 5.0, "output": 20.0},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gemini-2.5-flash": {"input": 0.10, "output": 0.40}
    }

    model_pricing = pricing.get(model, pricing["gpt-4o"])
    usage = response.usage

    cost = (
        (usage.prompt_tokens / 1_000_000) * model_pricing["input"] +
        (usage.completion_tokens / 1_000_000) * model_pricing["output"]
    )

    return cost
```

---

## Quick Reference

### Orchestration Pattern Selection

```
Use Case                          | Recommended Pattern
----------------------------------|------------------------------------
Sequential specialization         | Pipeline
Critical decision (safety)        | Ensemble with voting
Cost optimization                 | Cascade (cheap -> expensive)
Domain-specific routing           | Specialist routing
Complex multi-step workflows      | Hybrid (Pipeline + Specialist)
High-reliability requirements     | Ensemble + Fallback chains
```

### Context Passing Strategies

```
Strategy           | Cost   | Latency | Use Case
-------------------|--------|---------|---------------------------
Full history       | High   | High    | Short conversations (<10 turns)
Summarization      | Medium | Medium  | Long conversations
Key extraction     | Low    | Low     | Factual pipelines
Shared state (DB)  | Low    | Medium  | Multi-user systems
No context         | Lowest | Lowest  | Independent tasks
```

### Error Handling Patterns

```python
# Pattern 1: Fallback chain
try:
    response = call_model("claude-3-5-sonnet", prompt)
except Exception:
    try:
        response = call_model("gpt-4o", prompt)
    except Exception:
        response = call_model("gpt-4o-mini", prompt)

# Pattern 2: Retry with exponential backoff
for attempt in range(3):
    try:
        response = call_model("gpt-4o", prompt)
        break
    except Exception as e:
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)

# Pattern 3: Graceful degradation
results = []
for model in ["gpt-4o", "claude-3-5-sonnet", "gemini-2.5-pro"]:
    try:
        results.append(call_model(model, prompt))
    except Exception:
        pass  # Continue with other models

if results:
    return aggregate(results)
else:
    return default_response()
```

---

## Anti-Patterns

❌ **Passing full context between all stages**: Wastes tokens and money
✅ Summarize or extract key information between pipeline stages

❌ **Ensemble without voting mechanism**: Can't resolve disagreements
✅ Implement majority voting or weighted consensus

❌ **Cascade without confidence scores**: Can't decide when to escalate
✅ Prompt models for confidence or use logprobs for uncertainty estimation

❌ **No fallback on model failures**: Single point of failure
✅ Implement fallback chains with multiple model options

❌ **Ignoring cost in orchestration**: Ensemble costs 3x per query
✅ Track costs per pattern; use ensembles only for critical decisions

❌ **Synchronous calls in ensemble**: 3x latency penalty
✅ Use async/parallel calls for ensemble and specialist routing

❌ **No observability in multi-model flows**: Can't debug or optimize
✅ Use Arize Phoenix or similar for span-level tracing

❌ **Hard-coded model names in orchestration**: Brittle to model updates
✅ Use configuration or model registry for flexibility

---

## Hanzo SDK: Simplified Multi-Model Orchestration

Instead of manually implementing orchestration patterns, use **Hanzo SDK** for automatic multi-model coordination with built-in routing, caching, and failover.

### Pipeline Pattern with Hanzo

```python
from hanzo import Hanzo

# Initialize with different strategies for different stages
hanzo_fast = Hanzo(routing_strategy='latency_optimized')
hanzo_smart = Hanzo(routing_strategy='quality_focused')

async def research_pipeline(topic: str) -> str:
    # Stage 1: Fast outline generation
    outline = await hanzo_fast.chat.completions.create_async(
        messages=[{
            'role': 'user',
            'content': f'Create a research outline for: {topic}'
        }]
    )
    
    # Stage 2: Quality content generation
    content = await hanzo_smart.chat.completions.create_async(
        messages=[{
            'role': 'user',
            'content': f'Write detailed content for: {outline.content}'
        }]
    )
    
    return content.choices[0].message.content

# Hanzo automatically:
# - Routed stage 1 to fastest model (Gemini Flash)
# - Routed stage 2 to best model (GPT-4)
# - Handled retries and fallbacks
# - Tracked costs per stage
```

### Ensemble Pattern with Hanzo

```python
from hanzo import Hanzo

async def ensemble_decision(query: str) -> str:
    # Hanzo can query multiple models in parallel
    hanzo = Hanzo(inference_mode='hybrid')
    
    # Parallel calls to different models
    responses = await hanzo.chat.completions.batch_create_async([
        {'messages': [{'role': 'user', 'content': query}], 'model': 'gpt-4'},
        {'messages': [{'role': 'user', 'content': query}], 'model': 'claude-3-5-sonnet'},
        {'messages': [{'role': 'user', 'content': query}], 'model': 'gemini-2.5-pro'}
    ])
    
    # Extract answers
    answers = [r.choices[0].message.content for r in responses]
    
    # Aggregate with another model
    final = await hanzo.chat.completions.create_async(
        messages=[{
            'role': 'user',
            'content': f'Synthesize these expert opinions:\n\n' + '\n\n'.join(answers)
        }],
        model='gpt-4'  # Use best model for synthesis
    )
    
    return final.choices[0].message.content
```

### Specialist Routing with Hanzo MCP

```python
from hanzo import Hanzo
from hanzo.mcp import MCPClient

# Hanzo can use MCP tools for specialized tasks
hanzo = Hanzo(
    inference_mode='hybrid',
    mcp_servers=['http://localhost:8081']
)

mcp = MCPClient('http://localhost:8081')

async def route_by_intent(query: str) -> str:
    # First, classify intent
    intent = await hanzo.chat.completions.create_async(
        messages=[{
            'role': 'user',
            'content': f'Classify this query intent (code/search/analyze): {query}'
        }],
        model='gpt-4o-mini'  # Fast classifier
    )
    
    if 'code' in intent.content.lower():
        # Use code specialist model
        return await hanzo.chat.completions.create_async(
            messages=[{'role': 'user', 'content': query}],
            model='claude-3-5-sonnet'  # Best for code
        )
    elif 'search' in intent.content.lower():
        # Use MCP search tool
        tools = mcp.get_tools()
        return await hanzo.chat.completions.create_async(
            messages=[{'role': 'user', 'content': query}],
            tools=tools,
            tool_choice='auto'
        )
    else:
        # General analysis
        return await hanzo.chat.completions.create_async(
            messages=[{'role': 'user', 'content': query}],
            model='gpt-4'  # Best for analysis
        )
```

### Cascade Pattern with Hanzo

```python
from hanzo import Hanzo

hanzo_local = Hanzo(inference_mode='local')
hanzo_cloud = Hanzo(inference_mode='cloud')

async def cascade_query(query: str, complexity_threshold: float = 0.7) -> str:
    # Try fast local model first
    try:
        response = await hanzo_local.chat.completions.create_async(
            messages=[{'role': 'user', 'content': query}],
            model='llama-3-8b',
            timeout=5  # Fast timeout
        )
        
        # Check confidence (if model provides it)
        if hasattr(response, 'confidence') and response.confidence > complexity_threshold:
            return response.choices[0].message.content
    except TimeoutError:
        pass  # Fall through to cloud
    
    # Escalate to cloud model
    response = await hanzo_cloud.chat.completions.create_async(
        messages=[{'role': 'user', 'content': query}],
        model='gpt-4'  # Best model for complex queries
    )
    
    return response.choices[0].message.content
```

### Comparison: Manual Orchestration vs Hanzo SDK

| Pattern | Manual Implementation | Hanzo SDK |
|---------|----------------------|-----------|
| **Pipeline** | 50+ lines, manual routing | 10-15 lines, auto-routing |
| **Ensemble** | Custom parallel execution | Built-in batch API |
| **Specialist** | Manual intent classification | Automatic or MCP-based |
| **Cascade** | Custom timeout/retry logic | Built-in with fallbacks |
| **Cost Tracking** | Custom per-pattern | Automatic all patterns |
| **Caching** | Manual between stages | Automatic semantic cache |
| **Observability** | Manual OpenTelemetry | Built-in metrics |
| **Error Handling** | Custom per model | Automatic retry/circuit breaker |

### Advanced: Multi-Model Workflow

```python
from hanzo import Hanzo
from hanzo.workflows import Pipeline

# Define reusable workflow
class ResearchWorkflow(Pipeline):
    def __init__(self):
        super().__init__()
        
        # Different Hanzo instances for different stages
        self.fast_hanzo = Hanzo(routing_strategy='latency_optimized')
        self.smart_hanzo = Hanzo(routing_strategy='quality_focused')
        self.local_hanzo = Hanzo(inference_mode='local')
    
    async def research(self, topic: str) -> dict:
        # Stage 1: Generate outline (fast)
        outline = await self.fast_hanzo.chat.completions.create_async(
            messages=[{'role': 'user', 'content': f'Outline: {topic}'}]
        )
        
        # Stage 2: Parallel research on each section
        sections = outline.content.split('\n')
        research_tasks = [
            self.local_hanzo.chat.completions.create_async(
                messages=[{'role': 'user', 'content': f'Research: {section}'}]
            )
            for section in sections
        ]
        research_results = await asyncio.gather(*research_tasks)
        
        # Stage 3: Synthesize with best model
        synthesis = await self.smart_hanzo.chat.completions.create_async(
            messages=[{
                'role': 'user',
                'content': f'Synthesize:\n\n' + '\n\n'.join([r.content for r in research_results])
            }]
        )
        
        return {
            'outline': outline.content,
            'research': [r.content for r in research_results],
            'synthesis': synthesis.content,
            'total_cost': sum([
                outline.cost,
                sum(r.cost for r in research_results),
                synthesis.cost
            ])
        }

# Use workflow
workflow = ResearchWorkflow()
result = await workflow.research('Quantum Computing Applications')
print(f"Total cost: ${result['total_cost']:.4f}")
```

**Key Benefits**:
- **10x less code**: Hanzo SDK handles routing, retries, caching automatically
- **Built-in observability**: Automatic cost tracking and performance metrics
- **Privacy-first**: Local models via Hanzo Node before cloud escalation
- **Production-ready**: Circuit breakers, retries, fallbacks included
- **Type-safe**: Full TypeScript support with Python compatibility

**See Also**:
- **python-sdk.md** - Complete Hanzo SDK documentation
- **hanzo-mcp.md** - MCP tool integration for workflows
- **hanzo-node.md** - Local model inference
- **llm-model-routing.md** - Automatic routing strategies

---

## Related Skills

- `llm-model-routing.md` - Routing strategies and RouteLLM framework
- `llm-model-selection.md` - Choosing models for capabilities and cost
- `api-error-handling.md` - Robust error handling across providers
- `observability-distributed-tracing.md` - OpenTelemetry and Phoenix
- `modal-web-endpoints.md` - Deploying orchestration as FastAPI services
- `redis-caching.md` - Shared state and caching for multi-model systems
- **`python-sdk.md`** - Hanzo SDK for multi-model orchestration
- **`hanzo-mcp.md`** - MCP tools for specialized tasks
- **`hanzo-node.md`** - Local inference infrastructure

---

**Last Updated**: 2025-10-28
**Format Version**: 1.0 (Atomic)
