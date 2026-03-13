# Hanzo Kensho - Image Generation Foundation Model

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-ml.md`, `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`

## Overview

Kensho is a **17B parameter image generation foundation model** using a Mixture of Experts (MoE) Diffusion Transformer architecture. It generates high-quality images from text prompts at multiple resolutions, achieving state-of-the-art scores on DPG-Bench (85.89) and GenEval benchmarks.

The architecture combines a DiT (Diffusion Transformer) backbone with a MoE feed-forward network (4 routed experts, 2 activated per token) and uses Llama 3.1-8B-Instruct as a text encoder for deep prompt understanding. Generation uses flow matching schedulers for fast, high-fidelity output.

### Zen MoDE Architecture

Kensho implements the Zen MoDE (Mixture of Diverse Experts) methodology applied to image diffusion:
- **MoE gating** with softmax scoring and top-k expert selection
- **Load balancing loss** for uniform expert utilization across distributed training
- **SwiGLU feed-forward** blocks within both joint and single transformer layers
- **Flash Attention** for memory-efficient cross-modal attention between image patches and text embeddings

## When to use

- Text-to-image generation at production quality
- High-resolution image synthesis (up to 1360x768, multiple aspect ratios)
- Applications requiring deep prompt understanding (complex scenes, text rendering)
- Research into MoE diffusion architectures

## Hard requirements

1. **Python 3.10+** with PyTorch 2.5+
2. **CUDA 12.4+** with Flash Attention installed
3. **GPU VRAM**: ~40GB (full model in bfloat16)
4. **Llama 3.1-8B-Instruct** access (HuggingFace gated model)

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/kensho` |
| Branch | `main` |
| Language | Python |
| Parameters | 17B |
| Architecture | MoE Diffusion Transformer |
| Precision | bfloat16 |
| Install | `pip install -r requirements.txt && pip install -U flash-attn --no-build-isolation` |
| Inference | `python inference.py --model_type full` |
| License | See repo |

## Model Variants

| Variant | Inference Steps | Guidance Scale | Flow Shift | Scheduler |
|---------|----------------|----------------|------------|-----------|
| Full | 50 | 5.0 | 3.0 | FlowUniPCMultistep |
| Dev | 28 | 0.0 | 6.0 | FlashFlowMatchEuler |
| Fast | 16 | 0.0 | 3.0 | FlashFlowMatchEuler |

## Supported Resolutions

| Resolution | Aspect |
|------------|--------|
| 1024 x 1024 | Square |
| 768 x 1360 | Portrait |
| 1360 x 768 | Landscape |
| 880 x 1168 | Portrait |
| 1168 x 880 | Landscape |
| 1248 x 832 | Landscape |
| 832 x 1248 | Portrait |

## Architecture Details

```
Text Prompt
    |
    v
┌──────────────────────────────┐
│ Llama 3.1-8B-Instruct        │ (frozen text encoder)
│ + CLIP/T5 text encoders      │
└──────────┬───────────────────┘
           |
           v
┌──────────────────────────────┐
│ Kensho DiT (17B params)      │
│                              │
│  PatchEmbed(4ch -> 1024dim)  │
│  + RoPE positional encoding  │
│  + Timestep embedding        │
│                              │
│  Joint Transformer Blocks:   │
│    - Cross-attention (img+txt)│
│    - adaLN modulation        │
│    - MoE FFN (SwiGLU)        │
│      4 experts, top-2 routing│
│      + load balancing loss   │
│                              │
│  Single Transformer Blocks:  │
│    - Self-attention (img)    │
│    - adaLN modulation        │
│    - MoE FFN (SwiGLU)        │
│                              │
│  Output projection           │
└──────────────────────────────┘
           |
           v
     Generated Image
```

### MoE Gate Implementation

The expert routing uses a softmax-based gating mechanism:
- Input projected through learned weight matrix to `num_routed_experts` logits
- Softmax scoring with top-k selection (k=2 by default)
- Auxiliary load balancing loss (alpha=0.01) for training stability
- All-gather across distributed ranks for balanced expert utilization

## One-file quickstart

```python
import torch
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Load text encoder
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False
)
text_encoder = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load pipeline
scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False
)
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    "HiDream-ai/HiDream-I1-Full",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe = HiDreamImagePipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Full",
    scheduler=scheduler,
    tokenizer_4=tokenizer,
    text_encoder_4=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda", torch.bfloat16)
pipe.transformer = transformer

# Generate
image = pipe(
    "A landscape painting of mountains at sunset",
    height=1024, width=1024,
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("output.png")
```

## Dependencies

```
torch>=2.5.1
torchvision>=0.20.1
diffusers>=0.32.1
transformers>=4.47.1
einops>=0.7.0
accelerate>=1.2.1
flash-attn (manual install)
```

## Project Structure

```
kensho/
├── hi_diffusers/
│   ├── __init__.py
│   ├── models/
│   │   ├── attention.py              # HiDreamAttention + FeedForwardSwiGLU
│   │   ├── attention_processor.py    # Flash Attention processor
│   │   ├── embeddings.py             # PatchEmbed, RoPE, TimestepEmbed
│   │   ├── moe.py                    # MoEGate + MOEFeedForwardSwiGLU
│   │   └── transformers/
│   │       └── transformer_hidream_image.py  # Main DiT model
│   ├── pipelines/
│   │   └── hidream_image/
│   │       ├── pipeline_hidream_image.py     # Inference pipeline
│   │       └── pipeline_output.py
│   └── schedulers/
│       ├── flash_flow_match.py       # FlashFlowMatchEulerDiscreteScheduler
│       └── fm_solvers_unipc.py       # FlowUniPCMultistepScheduler
├── inference.py                      # CLI inference script
├── gradio_demo.py                    # Gradio web UI
├── requirements.txt
└── assets/
```

## Benchmark Results (DPG-Bench)

| Model | Overall | Global | Entity | Attribute | Relation | Other |
|-------|---------|--------|--------|-----------|----------|-------|
| DALL-E 3 | 83.50 | 90.97 | 89.61 | 88.39 | 90.58 | 89.83 |
| Flux.1-dev | 83.79 | 85.80 | 86.79 | 89.98 | 90.04 | 89.90 |
| SD3-Medium | 84.08 | 87.90 | 91.01 | 88.83 | 80.70 | 88.68 |
| **Kensho** | **85.89** | 76.44 | 90.22 | 89.48 | 93.74 | 91.83 |

## Related Skills

- `hanzo/hanzo-ml.md` - Rust ML framework for inference optimization
- `hanzo/hanzo-engine.md` - Model serving infrastructure
- `hanzo/hanzo-candle.md` - Candle tensor operations
- `hanzo/hanzo-sho.md` - Text diffusion (complementary modality)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: image-generation, diffusion, moe, transformer, text-to-image
**Prerequisites**: Python, PyTorch, CUDA, diffusion model concepts
