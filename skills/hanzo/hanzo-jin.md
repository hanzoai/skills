# Hanzo Jin - Visual Self-Supervised Learning Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Jin is a **research-stage visual self-supervised learning framework** implementing Joint Embedding Predictive Architectures (JEPA). Python + PyTorch. Implements I-JEPA, Saccade JEPA (novel variant), and Self-Distillation MAE.

**NOTE**: Jin is vision-only. The "multimodal" roadmap (text + audio + 3D) exists in grant proposals but is not yet implemented in code. Current codebase is image-patch JEPA training only — no inference API, no published weights.

### Why Jin?

- **JEPA architecture**: Self-supervised visual representation learning
- **Novel Saccade JEPA**: Inspired by mammalian saccadic eye movement
- **Self-Distillation MAE**: Masked autoencoder with DINO-style centering
- **Energy I-JEPA**: Alternative using Hopfield-based energy attention
- **Research tool**: Benchmarking, visualization, attention maps

### OSS Base

Originated from **LumenPallidium/jepa**, adopted under Zen LM org. Repo: `github.com/hanzoai/jin` (redirects to `zenlm/jin`). Package name: `jin-tac`.

## When to use

- Research into visual self-supervised learning
- Training JEPA models on image datasets (ImageNet)
- Experimenting with novel JEPA variants
- Benchmarking visual representation quality
- NOT for production multimodal inference (not yet implemented)

## Hard requirements

1. **Python 3.8+** with PyTorch
2. **GPU recommended** for training (ImageNet-scale)

## Quick reference

| Item | Value |
|------|-------|
| Language | Python (PyTorch) |
| Package | `jin-tac` |
| Repo | `github.com/hanzoai/jin` (→ `zenlm/jin`) |
| Train | `python jepa/train.py` |
| Config | `config/training.yml` |

## Model Variants (Implemented)

| Model | Class | Backbone | Description |
|-------|-------|----------|-------------|
| I-JEPA | `ViTJepa` | ViT (Transformer) | Paper implementation (arXiv:2301.08243) |
| Energy I-JEPA | `EnergyIJepa` | Energy Transformer | Hopfield-based energy attention |
| Saccade JEPA | `SaccadeJepa` | ConvNeXT tiny | Novel: mammalian saccadic eye movement |
| Self-Distillation MAE | `SelfDistillMAE` | ViT + cross-attention | Masked autoencoder with DINO centering |

## Architecture

```
┌─────────────────────────────────────────┐
│           Jin JEPA Architecture          │
├──────────────────────────────────────────┤
│                                          │
│  Image ──▶ Patcher ──▶ Patch Embeddings │
│                    ┌──────────┐          │
│  Context patches ──│ Context  │          │
│                    │ Encoder  │── EMA ──▶ Target Encoder
│                    └────┬─────┘          │
│                         │                │
│                    ┌────┴─────┐          │
│                    │Predictor │          │
│                    └────┬─────┘          │
│                         │                │
│           Predict target embeddings      │
│           from context embeddings        │
│                                          │
│  Loss: MSE(predicted, target_stopped)    │
│  + VICReg (variance + covariance)        │
│  + Cycle consistency (Saccade only)      │
└──────────────────────────────────────────┘
```

## One-file quickstart

```python
# Training (the only supported mode)
import yaml
from jepa.train import train
from jepa.jepa import ViTJepa

# Load config
with open("config/training.yml") as f:
    config = yaml.safe_load(f)

# Create model
model = ViTJepa(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

# Train on ImageNet
train(model, config)
```

### Training Configuration (config/training.yml)

```yaml
model:
  type: "vit_jepa"        # or "saccade_jepa", "energy_ijepa", "self_distill_mae"
  image_size: 224
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12

training:
  dataset: "imagenet"
  batch_size: 128
  gradient_accumulation: 128
  learning_rate: 1.5e-4
  warmup_epochs: 40
  total_epochs: 300
  weight_decay: 0.05
  ema_momentum: 0.996      # Target encoder EMA

  schedule:
    type: "cosine"
    min_lr: 1e-6
```

### Saccade JEPA (Novel Variant)

```python
from jepa.jepa import SaccadeJepa

model = SaccadeJepa(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    # Uses ConvNeXT tiny backbone
    # NeRF-like positional encoding of rotation/translation affine transforms
    # VICReg loss + cycle consistency
)

# Cycle consistency: forward-backward saccade prediction must reconstruct original
# Mimics mammalian visual system's saccadic eye movements
```

## Evaluation

| Method | Purpose |
|--------|---------|
| Linear probes | Evaluate frozen representations |
| KNN (k-nearest neighbors) | Non-parametric evaluation |
| Correlation dimension | Representation geometry |
| UMAP visualization | Embedding space visualization |
| Attention map dashboard | Interactive Dash visualization |

## Dependencies

- `torch`, `torchvision` — core framework
- `einops` — tensor operations
- `pyyaml` — config parsing
- `tqdm`, `numpy`, `matplotlib` — utilities
- `sklearn` (optional) — KNN evaluation
- `umap-learn` (optional) — visualization
- `energy_transformer` (optional) — for Energy I-JEPA variant

## Project Structure

```
jin/
├── jepa/
│   ├── jepa.py          # Core models (ViTJepa, SaccadeJepa, EnergyIJepa)
│   ├── masked_autoencoder.py  # Self-Distillation MAE
│   ├── train.py         # Training loop
│   ├── patcher.py       # Image patch embedding (Conv, Hybrid, Conv3d)
│   ├── saccade.py       # Saccade cropper (NeRF positional encoding)
│   └── vicreg.py        # VICReg loss terms
├── config/
│   └── training.yml     # Training configuration
├── papers/              # Research papers and grant proposals
├── pyproject.toml
└── LLM.md
```

## Roadmap (Aspirational — NOT in code)

Grant proposals describe future multimodal expansion:
- Text encoder (not implemented)
- Audio encoder (not implemented)
- Diffusion Transformer MoE (not implemented)
- Cross-modal alignment (not implemented)

## Related Skills

- `hanzo/hanzo-engine.md` - Inference engine (future: serve trained Jin models)
- `hanzo/hanzo-candle.md` - Rust ML framework
- `hanzo/zenlm.md` - Text-only Zen models
- `hanzo/hanzo-gym.md` - Training infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: jepa, self-supervised, vision, research
**Prerequisites**: Python, PyTorch, self-supervised learning concepts
