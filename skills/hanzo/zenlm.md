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

**GitHub Organization**: [github.com/zenlm](https://github.com/zenlm)
**HuggingFace Organization**: [huggingface.co/zenlm](https://huggingface.co/zenlm)

## Performance

Zen models are optimized for local inference on consumer hardware:

- **Memory**: ~3.8 GB for full 4B model, ~1.5 GB quantized (4-bit)
- **Speed**: 50+ tokens/sec on Apple Silicon (M1+), 30+ on modern CPUs
- **Offline**: No internet required after initial download

See model cards on HuggingFace for specific benchmark results.

## Quick Start

### Installation: Transformers (Python)

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load zen-nano-instruct
model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-nano-instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct")

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
model, tokenizer = load("zenlm/zen-nano-instruct")

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
huggingface-cli download zenlm/zen-nano-instruct-4bit \
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

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct")

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
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-thinking")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-thinking")

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

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-agent")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-agent")

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
model, tokenizer = load('zenlm/zen-nano-instruct')
print(generate(model, tokenizer, prompt='Hello!'))
"
```

### Mobile/Edge (4GB RAM)

```bash
# 4-bit quantized (1.5GB memory)
python -c "
from mlx_lm import load, generate
model, tokenizer = load('zenlm/zen-nano-instruct-4bit')
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
  'zenlm/zen-nano-instruct',
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
huggingface-cli download zenlm/zen-nano-instruct-4bit \
    --include "*Q4_K_M.gguf" \
    --local-dir ./models
```

### MLX Formats (Apple Silicon)

```bash
# 4-bit MLX (optimized for M-series)
pip install mlx-lm
python -m mlx_lm.convert \
    --hf-path zenlm/zen-nano-instruct \
    --mlx-path ./zen-nano-mlx \
    --quantize
```

### AWQ/GPTQ (NVIDIA GPUs)

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

# 4-bit GPTQ for NVIDIA
model = AutoGPTQForCausalLM.from_quantized(
    "zenlm/zen-nano-instruct-gptq",
    device="cuda:0"
)
```

## Integration with Hanzo Ecosystem

### Use via Hanzo LLM Gateway

```python
from hanzoai import Hanzo

client = Hanzo()
response = client.chat.completions.create(
    model="zen-nano-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with OpenAI SDK (drop-in)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.hanzo.ai/v1",
    api_key=os.environ["HANZO_API_KEY"],
)
response = client.chat.completions.create(
    model="zen-nano-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Fine-tuning with Hanzo Gym

Train custom ZenLM models using LLaMA Factory:

```bash
# Fine-tune zen-nano for your domain
llamafactory-cli train \
    --stage sft \
    --model_name_or_path zenlm/zen-nano-instruct \
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

## Local vs Cloud

| | Local (ZenLM) | Cloud API |
|--|---|---|
| Cost | Free (Apache 2.0) | Per-token pricing |
| Privacy | 100% local, data stays on device | Data sent to provider |
| Latency | 20-50ms (no network) | 200-500ms (network + queue) |
| Offline | Works fully offline | Requires internet |
| Quality | Competitive at 4B scale | Larger models available |

## Zoo Labs Foundation

ZenLM is co-developed with Zoo Labs Foundation (501(c)(3)), focused on sustainable and accessible AI research.

## Use Cases

### 1. Privacy-Critical Applications

```python
# Medical records, financial data, legal documents
# Data never leaves your device

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct")

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
model, tokenizer = load("zenlm/zen-nano-instruct-4bit")
response = generate(model, tokenizer, prompt="Help with code offline")
```

### 3. Cost-Optimized Production

```python
# Use zen-nano via gateway for cost-optimized production
from hanzoai import Hanzo

client = Hanzo()
response = client.chat.completions.create(
    model="zen-nano-instruct",  # Cheapest Zen model
    messages=[{"role": "user", "content": query}],
)
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

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-agent")

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
AutoModelForCausalLM.from_pretrained('zenlm/zen-nano-instruct')
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
        image: zenlm/zen-nano-instruct:latest
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Serverless

WASM deployment is planned but not yet available. For now, use container-based serverless (Cloud Run, Lambda with container support).

## Community

- **GitHub**: [github.com/zenlm](https://github.com/zenlm)
- **HuggingFace**: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- **Discord**: [discord.gg/zenlm](https://discord.gg/zenlm)
- **Docs**: [zenlm.org](https://zenlm.org)

## Related Skills

- **hanzo-gym.md**: Train and fine-tune ZenLM models
- **python-sdk.md**: Deploy ZenLM with Hanzo SDK
- **hanzo-desktop.md**: Hanzo Desktop app
- **hanzo-dev.md**: Terminal AI coding agent

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: zenlm, models, local-ai, inference, edge
**Prerequisites**: Python 3.10+, basic terminal knowledge