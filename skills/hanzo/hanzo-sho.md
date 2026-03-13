# Hanzo Sho - Text Diffusion Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-kensho.md`, `hanzo/hanzo-ml.md`, `hanzo/hanzo-engine.md`

## Overview

Sho is a **text diffusion engine** that generates text using masked diffusion rather than autoregressive token prediction. The core model (Genjo, 8B parameters) uses a Transformer Encoder with bidirectional attention and an iterative masked denoising process to produce coherent text. This approach allows parallel token prediction with progressive refinement, producing high-quality output competitive with autoregressive models on standard benchmarks.

Sho serves as the foundation for integration with the Enso diffusion Mixture of Experts (MoE) architecture, targeting 16B+ parameter models with expert specialization.

### Key Innovation

Unlike autoregressive models (LLaMA) that generate one token at a time left-to-right, Sho:
- Predicts **all masked tokens simultaneously** at each step
- Uses **iterative denoising**: selectively unmasks the most confident predictions
- Employs a **varying masking ratio** (0 to 1) during training as an upper bound on negative log-likelihood
- Supports **classifier-free guidance** for improved benchmark performance
- Uses **semi-autoregressive block generation** for variable-length output

## When to use

- Research into non-autoregressive text generation
- Applications where parallel token prediction is advantageous
- Benchmarking diffusion-based language models against autoregressive baselines
- Exploring hybrid diffusion + MoE architectures

## Hard requirements

1. **Python 3.8+** with PyTorch
2. **transformers==4.38.2** (specific version required for model loading)
3. **GPU recommended** for inference (bfloat16 support)
4. **lm-evaluation-harness** for benchmarking

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/sho` |
| Branch | `main` |
| Language | Python |
| Parameters | 8B |
| Architecture | Transformer Encoder (bidirectional) |
| Mask Token ID | 126336 |
| Generate | `python generate.py` |
| Chat | `python chat.py` |
| Evaluate | `python eval_llada.py` |
| Gradio Demo | `python app.py` |
| License | See repo |

## Model Variants

| Variant | Purpose | HuggingFace |
|---------|---------|-------------|
| Genjo-8B-Base | Foundation model | `GSAI-ML/Genjo-8B-Base` |
| Genjo-8B-Instruct | Chat/instruction following | `GSAI-ML/Genjo-8B-Instruct` |

## Architecture

```
┌──────────────────────────────────────────┐
│            Sho Diffusion Process          │
├──────────────────────────────────────────┤
│                                          │
│  Input: [prompt tokens] [MASK MASK ...]  │
│                                          │
│  For each denoising step:                │
│    1. Transformer Encoder (bidirectional)│
│       - Full self-attention (no causal)  │
│       - Same params as decoder (8B)      │
│                                          │
│    2. Predict all masked positions       │
│       - Gumbel noise sampling            │
│       - Optional CFG guidance            │
│                                          │
│    3. Confidence-based unmasking         │
│       - Score: softmax probability       │
│       - Unmask top-k most confident      │
│       - Or random selection              │
│                                          │
│    4. Re-mask remaining positions        │
│       - Linear schedule across steps     │
│                                          │
│  Output: fully unmasked response         │
└──────────────────────────────────────────┘
```

### Comparison with Autoregressive Models

| Feature | Sho (Diffusion) | Autoregressive (LLaMA) |
|---------|-----------------|------------------------|
| Architecture | Transformer Encoder | Transformer Decoder |
| Attention | Bidirectional | Unidirectional (causal) |
| Training | Masked diffusion, varying ratio | Next-token prediction |
| Generation | Parallel predict + iterative denoise | Sequential token-by-token |
| KV-Cache | Not applicable | Yes (faster inference) |
| In-context Learning | Yes | Yes |

### Generation Algorithm

The generation function implements block-based semi-autoregressive diffusion:

1. **Initialize** fully masked response sequence
2. **Divide** into blocks of `block_length` tokens
3. **Per block**, run `steps` denoising iterations:
   - Forward pass through encoder to get logits for all positions
   - Apply Gumbel noise (temperature-controlled) for sampling
   - Apply classifier-free guidance if `cfg_scale > 0`
   - Select tokens to unmask based on confidence (`low_confidence`) or randomly
   - Number of tokens unmasked per step is pre-computed for uniform distribution

Key parameters:
- `steps`: Total denoising steps (default 128)
- `gen_length`: Maximum generated tokens (default 128)
- `block_length`: Semi-autoregressive block size (default 128)
- `temperature`: Sampling temperature (0 = greedy)
- `cfg_scale`: Classifier-free guidance strength (0 = disabled)
- `remasking`: Strategy - `low_confidence` or `random`

## One-file quickstart

### Text Generation

```python
import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/Genjo-8B-Base", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "GSAI-ML/Genjo-8B-Base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

prompt = "The history of artificial intelligence begins with"
input_ids = torch.tensor(
    tokenizer(prompt)["input_ids"], device=device
).unsqueeze(0)

output = generate(
    model, input_ids,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)

result = tokenizer.batch_decode(
    output[:, input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]
print(result)
```

### Chat with Instruct Model

```python
import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/Genjo-8B-Instruct", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "GSAI-ML/Genjo-8B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

messages = [{"role": "user", "content": "Explain diffusion models in 3 sentences."}]
chat_input = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
input_ids = torch.tensor(
    tokenizer(chat_input)["input_ids"], device=device
).unsqueeze(0)

output = generate(
    model, input_ids,
    steps=128, gen_length=256, block_length=64,
)
print(tokenizer.batch_decode(
    output[:, input_ids.shape[1]:], skip_special_tokens=True
)[0])
```

## Training

### Pre-training (Core Code)

```python
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

# Training step
noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits
token_loss = F.cross_entropy(
    logits[masked_indices], input_ids[masked_indices], reduction="none"
) / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
```

### SFT (Supervised Fine-Tuning)

Same as pre-training but noise is only applied to the response portion; the prompt remains unmasked.

## Evaluation

Integration with `lm-evaluation-harness` via the `GenjoEvalHarness` class:

```bash
# Conditional likelihood estimation
accelerate launch eval_llada.py \
    --tasks gpqa_main_n_shot --num_fewshot 5 --model genjo_dist

# Conditional generation
accelerate launch eval_llada.py \
    --tasks bbh --model genjo_dist \
    --model_args gen_length=1024,steps=1024,block_length=1024
```

Evaluated on: BBH, GSM8K, Math, HumanEval, MBPP.

## Project Structure

```
sho/
├── generate.py              # Core generation: add_gumbel_noise, generate()
├── get_log_likelihood.py    # Monte Carlo log-likelihood estimation
├── eval_llada.py            # lm-evaluation-harness integration
├── eval_llada.sh            # Evaluation benchmark scripts
├── chat.py                  # Terminal chat interface (Instruct model)
├── app.py                   # Gradio web demo with denoising visualization
├── integration.py           # Integration utilities
├── demo_integration.py      # Demo integration
├── GUIDELINES.md            # Architecture and training details
├── EVAL.md                  # Evaluation documentation
├── LLM.md                   # Detailed project documentation
├── visualization/
│   ├── generate.py          # Visualization generation
│   ├── html_to_png.py       # Export visualizations
│   └── visualization_paper.py
└── imgs/                    # Benchmark comparison charts
```

## Future Directions

- **Enso MoE integration**: Scale to 16B+ with expert specialization via DiT-MoE
- **Semi-autoregressive sampling**: Reduce fixed context length limitations
- **Consistency distillation**: Fewer denoising steps without quality loss
- **DeepSpeed training**: Efficient large-scale training

## Related Skills

- `hanzo/hanzo-kensho.md` - Image diffusion (same diffusion paradigm, different modality)
- `hanzo/hanzo-ml.md` - Rust ML framework
- `hanzo/hanzo-engine.md` - Model serving
- `hanzo/hanzo-jin.md` - Visual self-supervised learning

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: text-diffusion, language-model, masked-diffusion, non-autoregressive, generation
**Prerequisites**: Python, PyTorch, transformers, diffusion model concepts
