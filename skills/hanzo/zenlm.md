# ZenLM - Next-Generation Local AI Models

**Category**: Hanzo Ecosystem
**Difficulty**: Beginner
**Prerequisites**: Python 3.10+, basic terminal knowledge
**Related Skills**: hanzo-gym.md, python-sdk.md, hanzo-desktop.md

## Overview

**ZenLM** is a groundbreaking collaboration between [Hanzo AI](https://hanzo.ai) (Techstars-backed GenAI lab) and [Zoo Labs Foundation](https://zoo.ngo) (501(c)(3) environmental non-profit), building AI models that run entirely on your device—no cloud, no subscriptions, no surveillance.

**Mission**: "Democratize AI while protecting our planet"

**Why ZenLM?**
- **Ultra-Efficient**: 4B parameters achieving 70B-class performance
- **Truly Private**: 100% local processing, your data never leaves your device
- **Environmentally Responsible**: 95% less energy than cloud AI
- **Free Forever**: Apache 2.0 licensed, no premium tiers or API fees

## Model Family

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| **zen-nano-instruct** | 4B | Ultra-fast instruction following | Chat, coding, Q&A |
| **zen-nano-thinking** | 4B | Chain-of-thought reasoning | Math, logic, analysis |
| **zen-nano-instruct-4bit** | 1.5GB | Quantized for mobile/edge | iOS, Android, Pi |
| **zen-nano-thinking-4bit** | 1.5GB | Quantized reasoning | Mobile reasoning |
| **zen-eco** | 4B | Efficient instruct/thinking | Production use |
| **zen-agent** | 4B | Tool-calling & function execution | Sub-agents, APIs |
| **zen-director** | 5B | Text-to-video generation | Video content |
| **zen-musician** | 7B | Music generation with lyrics | Music creation |

**GitHub Organization**: [github.com/zoolm](https://github.com/zoolm)
**HuggingFace Organization**: [huggingface.co/zoolm](https://huggingface.co/zoolm)

## Performance Benchmarks

| Benchmark | Zen-Nano-4B | GPT-3.5 | Llama-7B | Advantage |
|-----------|-------------|---------|----------|-----------|
| MMLU | 68.2% | 70.0% | 63.4% | Competitive |
| HellaSwag | 79.1% | 85.5% | 78.3% | Near parity |
| HumanEval | 52.4% | 48.1% | 31.2% | **+10% vs GPT-3.5** |
| Speed (tok/s) | 50+ | 20* | 30 | **2.5x faster** |
| Memory (GB) | 3.8 | 12+* | 13.5 | **70% less** |
| Energy/1K tok | 0.02 Wh | 0.4 Wh* | 0.3 Wh | **95% reduction** |

*Cloud-based, network latency not included

## Quick Start

### Installation: Transformers (Python)

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load zen-nano-instruct
model = AutoModelForCausalLM.from_pretrained(
    "zoolm/zen-nano-instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zoolm/zen-nano-instruct")

# Generate response
input_text = "Explain quantum computing in simple terms"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Installation: MLX (Apple Silicon)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

# Load zen-nano-instruct (optimized for M-series)
model, tokenizer = load("zoolm/zen-nano-instruct")

# Generate at 50+ tokens/sec on M1 Pro
response = generate(
    model,
    tokenizer,
    prompt="Write a haiku about AI",
    max_tokens=100,
    temp=0.7
)
print(response)
```

### Installation: llama.cpp (Universal)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Download GGUF version (1.5GB quantized)
huggingface-cli download zoolm/zen-nano-instruct-4bit \
    --include "*.gguf" \
    --local-dir ./models

# Run inference (50+ tokens/sec on CPU!)
./llama-cli \
    -m ./models/zen-nano-instruct-Q4_K_M.gguf \
    -p "Your prompt here" \
    -n 512 \
    --temp 0.7
```

## Model Details

### zen-nano-instruct (4B)

**Best for**: General chat, coding, question answering

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zoolm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zoolm/zen-nano-instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Write a Python function to reverse a string"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

### zen-nano-thinking (4B)

**Best for**: Math, logic, transparent reasoning

```python
model = AutoModelForCausalLM.from_pretrained("zoolm/zen-nano-thinking")
tokenizer = AutoTokenizer.from_pretrained("zoolm/zen-nano-thinking")

# Enable chain-of-thought
messages = [
    {"role": "user", "content": "Solve: If 3 apples cost $5, how much do 7 apples cost?"}
]

# Model will show reasoning steps
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0])

# Output shows thinking process:
# <think>
# Cost per apple = $5 / 3 = $1.67
# For 7 apples: 7 * $1.67 = $11.67
# </think>
# Answer: 7 apples cost $11.67
```

### zen-agent (4B)

**Best for**: Tool calling, function execution, sub-agents

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model = AutoModelForCausalLM.from_pretrained("zoolm/zen-agent")
tokenizer = AutoTokenizer.from_pretrained("zoolm/zen-agent")

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"}
]

# Model generates function call
text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False
)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0])

# Parse function call and execute
# function_call = {"name": "get_weather", "arguments": {"location": "San Francisco"}}
```

## Running on Different Hardware

### Desktop/Laptop (16GB RAM)

```bash
# Full 4B model (3.8GB memory)
python -c "
from mlx_lm import load, generate
model, tokenizer = load('zoolm/zen-nano-instruct')
print(generate(model, tokenizer, prompt='Hello!'))
"
```

### Mobile/Edge (4GB RAM)

```bash
# 4-bit quantized (1.5GB memory)
python -c "
from mlx_lm import load, generate
model, tokenizer = load('zoolm/zen-nano-instruct-4bit')
print(generate(model, tokenizer, prompt='Hello!'))
"
```

### Raspberry Pi 4 (4GB)

```bash
# llama.cpp for ARM
./llama-cli \
    -m zen-nano-instruct-Q4_K_M.gguf \
    -p "Hello from Raspberry Pi!" \
    -n 100 \
    --threads 4
```

### Browser (WebGPU)

```javascript
import { pipeline } from '@xenova/transformers';

// Load zen-nano in browser
const generator = await pipeline(
  'text-generation',
  'zoolm/zen-nano-instruct',
  { device: 'webgpu' }
);

const output = await generator('Explain neural networks', {
  max_length: 200
});
console.log(output[0].generated_text);
```

## Quantization Options

### GGUF Formats (llama.cpp)

| Quantization | Size | Quality | Speed | Use Case |
|--------------|------|---------|-------|----------|
| Q2_K | 1.2GB | 85% | Fastest | Mobile/Edge |
| Q4_K_M | 1.5GB | 95% | Fast | **Recommended** |
| Q5_K_M | 1.8GB | 97% | Medium | Balanced |
| Q8_0 | 2.4GB | 99% | Slower | Quality-focused |

```bash
# Download specific quantization
huggingface-cli download zoolm/zen-nano-instruct-4bit \
    --include "*Q4_K_M.gguf" \
    --local-dir ./models
```

### MLX Formats (Apple Silicon)

```bash
# 4-bit MLX (optimized for M-series)
pip install mlx-lm
python -m mlx_lm.convert \
    --hf-path zoolm/zen-nano-instruct \
    --mlx-path ./zen-nano-mlx \
    --quantize
```

### AWQ/GPTQ (NVIDIA GPUs)

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

# 4-bit GPTQ for NVIDIA
model = AutoGPTQForCausalLM.from_quantized(
    "zoolm/zen-nano-instruct-gptq",
    device="cuda:0"
)
```

## Integration with Hanzo Ecosystem

### Use with Hanzo Python SDK

```python
from hanzo import Hanzo

# Initialize with ZenLM local inference
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080',
    local_model='zoolm/zen-nano-instruct'
)

# Automatic routing: local for small tasks, cloud for complex
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

print(f"Model used: {response.model}")  # zen-nano-instruct (local)
print(f"Cost: ${response.cost:.4f}")    # $0.0000 (local)
print(f"Latency: {response.latency}ms") # 20ms (local)
```

### Use with Hanzo Dev (Terminal Agent)

```bash
# Hanzo Dev uses zen-nano locally
dev "write a Python function to calculate fibonacci"

# Response generated by zen-nano-instruct running locally
# No API keys, no cloud costs, instant response
```

### Deploy with Hanzo Desktop

Hanzo Desktop includes zen-nano for instant AI assistance:

```bash
# Press Tab → AI chat powered by zen-nano
# Context-aware, privacy-first, runs entirely on your Mac
```

## Fine-tuning with Hanzo Gym

Train custom ZenLM models:

```bash
cd /Users/z/work/zen/gym

# Fine-tune zen-nano for your domain
llamafactory-cli train \
    --stage sft \
    --model_name_or_path zoolm/zen-nano-instruct \
    --dataset your_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir ./zen-nano-custom \
    --flash_attn fa2 \
    --use_unsloth true

# Export to GGUF for deployment
llamafactory-cli export \
    --model_name_or_path ./zen-nano-custom \
    --export_dir ./zen-nano-gguf \
    --export_quantization_bit 4
```

See **hanzo-gym.md** for comprehensive training guide.

## Comparison: Cloud AI vs ZenLM

### Cloud API (GPT-3.5)

```python
import openai

# Requires API key, internet, and monthly subscription
client = openai.OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Cost: $0.002/1K tokens ($20-100/month typical)
# Latency: 200-500ms (network + processing)
# Privacy: Data sent to cloud, subject to terms of service
# Energy: 0.4 Wh per 1K tokens
```

### ZenLM Local (zen-nano)

```python
from mlx_lm import load, generate

# No API key, no internet, no subscription
model, tokenizer = load("zoolm/zen-nano-instruct")

response = generate(
    model,
    tokenizer,
    prompt="Hello!",
    max_tokens=100
)

# Cost: $0.00 (free forever)
# Latency: 20-50ms (local processing)
# Privacy: 100% local, data never leaves device
# Energy: 0.02 Wh per 1K tokens (95% reduction)
```

**Result**: ZenLM is **10x faster**, **100% private**, **infinitely cheaper**, and **95% more energy efficient** than cloud AI.

## Environmental Impact

### Carbon Savings

**Cloud AI (1 year of moderate use)**:
- Inference energy: ~200 kWh/year
- Data center overhead: ~400 kWh/year
- Total: 600 kWh = 240 kg CO₂

**ZenLM (1 year of moderate use)**:
- Inference energy: ~10 kWh/year
- No data center overhead
- Total: 10 kWh = 4 kg CO₂

**Savings**: 236 kg CO₂/year per user = **95% reduction**

### Zoo Labs Foundation Partnership

For every ZenLM download, Zoo Labs Foundation (501(c)(3)) plants trees and protects rainforest:

- **100K+ downloads** → 50 acres protected
- **Carbon-negative operations** → Net removal of CO₂
- **Sustainable AI** → Built on renewable energy

Learn more: [zoolabs.org](https://zoolabs.io)

## Use Cases

### 1. Privacy-Critical Applications

```python
# Medical records, financial data, legal documents
# Data never leaves your device

model = AutoModelForCausalLM.from_pretrained("zoolm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zoolm/zen-nano-instruct")

# Process sensitive data locally
sensitive_text = "Patient diagnosis: ..."
inputs = tokenizer(f"Summarize: {sensitive_text}", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
summary = tokenizer.decode(outputs[0])
```

### 2. Offline AI Applications

```python
# Airplanes, remote locations, intermittent connectivity

from mlx_lm import load, generate

# Works 100% offline
model, tokenizer = load("zoolm/zen-nano-instruct-4bit")
response = generate(model, tokenizer, prompt="Help with code offline")
```

### 3. Cost-Optimized Production

```python
# Save $1000s/month on API costs

from hanzo import Hanzo

# Local for 80% of queries, cloud for 20%
hanzo = Hanzo(
    inference_mode='hybrid',
    local_model='zoolm/zen-nano-instruct',
    cost_optimize=True
)

# Automatically routes based on complexity
response = hanzo.chat.completions.create(
    messages=[{'role': 'user', 'content': query}]
)

# Result: 80% cost reduction while maintaining quality
```

### 4. Edge AI / IoT

```bash
# Raspberry Pi, edge servers, mobile devices

# Install llama.cpp on ARM
./llama-cli \
    -m zen-nano-instruct-Q4_K_M.gguf \
    -p "Process sensor data: ..." \
    --threads 4
```

### 5. AI Sub-Agents

```python
# Lightweight agents for tool calling

model = AutoModelForCausalLM.from_pretrained("zoolm/zen-agent")

# Each sub-agent runs zen-agent locally (4GB memory)
# Coordinate 10+ agents on single machine
```

## Deployment Options

### Docker Container

```dockerfile
FROM python:3.10-slim

# Install ZenLM
RUN pip install transformers torch mlx-lm

# Download model
RUN python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('zoolm/zen-nano-instruct')
"

EXPOSE 8080
CMD ["python", "server.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zenlm-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: zenlm
        image: zoolm/zen-nano-instruct:latest
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Serverless (Cloudflare Workers)

```javascript
// zen-nano-instruct compiled to WASM
import { generate } from '@zoolm/zen-nano-wasm';

export default {
  async fetch(request) {
    const { prompt } = await request.json();
    const response = await generate(prompt);
    return new Response(JSON.stringify({ response }));
  }
}
```

## Community & Support

### Get Involved

- ⭐ **Star on GitHub**: [github.com/zoolm](https://github.com/zoolm)
- 💬 **Join Discord**: [discord.gg/zoolm](https://discord.gg/zoolm)
- 🐦 **Follow Twitter**: [@zoolm_ai](https://twitter.com/zenlm_ai)
- 📧 **Newsletter**: [zoolm.org/newsletter](https://zoolm.org/newsletter)

### For Developers

- Report issues
- Submit PRs
- Improve docs
- Create tutorials

### For Organizations

- Deploy models (free forever)
- Sponsor development (tax-deductible)
- Partner on research
- Join sustainability program

### For Everyone

- Use ZenLM for free
- Save energy
- Protect privacy
- Support open AI

## Related Skills

- **hanzo-gym.md**: Train and fine-tune ZenLM models
- **python-sdk.md**: Deploy ZenLM with Hanzo SDK
- **hanzo-desktop.md**: Run ZenLM on Mac with mining
- **hanzo-dev.md**: Use ZenLM for terminal AI assistance

## Additional Resources

- **Website**: [zoolm.org](https://zoolm.org)
- **GitHub**: [github.com/zoolm](https://github.com/zoolm)
- **HuggingFace**: [huggingface.co/zoolm](https://huggingface.co/zoolm)
- **Documentation**: [docs.zoolm.org](https://docs.zoolm.org)
- **Research Papers**: [zoolm.org/research](https://zoolm.org/research)

## Citation

If you use ZenLM in your research:

```bibtex
@misc{zenlm2025,
  title={ZenLM: Next-Generation Local AI Models},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zoolm}}
}
```

---

**ZenLM** - Building AI that's local, private, and free — for everyone, forever.

**A Hanzo AI × Zoo Labs Foundation project**
© 2025 Zen LM • Apache 2.0 License