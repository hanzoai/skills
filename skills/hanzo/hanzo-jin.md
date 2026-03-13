# Hanzo Jin - Unified Multimodal AI Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Jin is a **unified multimodal AI framework** combining text, vision, audio, and 3D understanding in a single architecture. Rust core with Python bindings for maximum performance and PyTorch interop.

### Why Jin?

- **True multimodal**: Text + vision + audio + 3D in one model
- **Cross-modal reasoning**: Understand relationships across modalities
- **Rust performance**: Core inference in Rust via Candle
- **Python API**: PyTorch-compatible training and inference
- **Flexible**: Use any modality combination per request

## When to use

- Building multimodal AI applications (text + images + audio)
- Vision-language model development and inference
- Audio processing and transcription with visual context
- 3D scene understanding
- Cross-modal search and retrieval

## Hard requirements

1. **Python 3.11+** with PyTorch 2.x
2. **CUDA 12+** or **Metal** (Apple Silicon) for GPU acceleration
3. **16GB+ VRAM** for 7B model, 8GB for smaller variants

## Quick reference

| Item | Value |
|------|-------|
| Language | Rust (core) + Python (API) |
| Framework | Candle (Rust) + PyTorch (Python) |
| Repo | `github.com/hanzoai/jin` |
| Modalities | Text, Vision, Audio, 3D |
| Model sizes | 1.5B, 7B, 13B |
| Build | `cargo build --release` (Rust), `uv sync` (Python) |
| Test | `cargo test` / `uv run pytest` |

## One-file quickstart

### Python API

```python
from jin import JinModel

# Load model
model = JinModel.from_pretrained("hanzo/jin-7b")

# Text generation
response = model.generate(
    text="Explain quantum computing in simple terms",
    max_tokens=500,
)
print(response.text)

# Text + Image (Vision-Language)
response = model.generate(
    text="Describe this image in detail. What objects are present?",
    image="photo.jpg",            # Path or URL
    max_tokens=500,
)
print(response.text)

# Text + Audio (Speech Understanding)
response = model.generate(
    text="Transcribe this audio and summarize the key points",
    audio="recording.wav",        # Path or URL
    max_tokens=500,
)
print(response.text)

# Multi-modal (Image + Audio + Text)
response = model.generate(
    text="What is happening in this scene?",
    image="video_frame.jpg",
    audio="ambient_sound.wav",
    max_tokens=500,
)
print(response.text)
```

### Image Analysis

```python
from jin import JinModel

model = JinModel.from_pretrained("hanzo/jin-7b")

# Object detection
objects = model.detect(image="scene.jpg")
for obj in objects:
    print(f"{obj.label}: {obj.confidence:.2f} at {obj.bbox}")

# Image captioning
caption = model.caption(image="photo.jpg")
print(caption)

# Visual Q&A
answer = model.vqa(
    image="chart.png",
    question="What is the trend shown in this chart?"
)
print(answer)

# Image similarity
score = model.similarity(
    image_a="photo1.jpg",
    image_b="photo2.jpg",
)
print(f"Similarity: {score:.3f}")
```

### Audio Processing

```python
from jin import JinModel

model = JinModel.from_pretrained("hanzo/jin-7b")

# Transcription
transcript = model.transcribe(audio="speech.wav")
print(transcript.text)
print(transcript.segments)  # Timestamped segments

# Audio classification
labels = model.classify_audio(audio="sound.wav")
for label in labels:
    print(f"{label.name}: {label.score:.2f}")

# Audio-visual alignment
alignment = model.align(
    audio="narration.wav",
    image="slide.png",
)
print(f"Alignment score: {alignment.score:.3f}")
```

### Rust Core API

```rust
use hanzo_jin::{Model, Config, Modality};

let model = Model::load(Config {
    model_path: "hanzo/jin-7b",
    device: Device::cuda_if_available(0)?,
    dtype: DType::BF16,
})?;

// Text generation
let output = model.generate(&[
    Modality::Text("Describe this image".to_string()),
    Modality::Image(image_tensor),
], max_tokens: 500)?;

println!("{}", output.text());
```

## Supported Modalities

| Modality | Input Types | Processing |
|----------|-------------|------------|
| Text | String, tokens | Transformer encoder/decoder |
| Vision | JPEG, PNG, WebP, tensor | Vision Transformer (ViT) |
| Audio | WAV, MP3, FLAC, tensor | Whisper-style encoder |
| 3D | Point clouds, meshes | 3D tokenizer |

## Model Variants

| Model | Parameters | VRAM | Modalities | Use Case |
|-------|-----------|------|------------|----------|
| jin-1.5b | 1.5B | 4GB | Text + Vision | Edge, mobile |
| jin-7b | 7B | 16GB | All 4 | General purpose |
| jin-13b | 13B | 32GB | All 4 | Maximum quality |

## Architecture

```
┌──────────────────────────────────────────┐
│              Jin Architecture             │
├──────────┬───────────┬───────┬───────────┤
│  Text    │  Vision   │ Audio │    3D     │
│ Tokenizer│   ViT     │Whisper│ 3D Tokens │
├──────────┴───────────┴───────┴───────────┤
│           Modality Alignment              │
│      (cross-attention, projection)        │
├──────────────────────────────────────────┤
│           Transformer Core               │
│      (shared decoder, cross-modal)       │
├──────────────────────────────────────────┤
│          Output Heads                    │
│   Text │ BBox │ Caption │ Transcript    │
└──────────────────────────────────────────┘
```

## Development

```bash
# Rust core
git clone https://github.com/hanzoai/jin.git
cd jin
cargo build --release
cargo test

# Python bindings
cd python
uv sync --all-extras
uv run pytest -v
```

## Related Skills

- `hanzo/hanzo-engine.md` - Inference engine (can serve Jin models)
- `hanzo/hanzo-candle.md` - Rust ML framework (Jin's tensor backend)
- `hanzo/zenlm.md` - Text-only Zen models
- `hanzo/hanzo-gym.md` - Training infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: multimodal, vision, audio, 3d, ai
**Prerequisites**: Python, PyTorch, ML fundamentals
