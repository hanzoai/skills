# Hanzo Jin - Unified Multimodal AI Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-candle.md`, `hanzo/zenlm.md`

## Overview

Jin is a **unified multimodal AI framework** combining text, vision, audio, and 3D understanding in a single architecture. Rust + Python for maximum performance with PyTorch integration.

## When to use

- Building multimodal AI applications (text + images + audio)
- Vision-language model development
- Audio processing and transcription
- Cross-modal understanding and generation

## Quick reference

| Item | Value |
|------|-------|
| Tech | Rust, Python, PyTorch |
| Repo | `github.com/hanzoai/jin` |
| Modalities | Text, Vision, Audio, 3D |

## One-file quickstart

```python
from jin import JinModel

model = JinModel.from_pretrained("hanzo/jin-7b")

# Text + Image
response = model.generate(
    text="Describe this image in detail",
    image="photo.jpg",
    max_tokens=500,
)

# Text + Audio
transcription = model.generate(
    text="Transcribe this audio",
    audio="recording.wav",
)
```

## Related Skills

- `hanzo/hanzo-engine.md` - Inference engine
- `hanzo/hanzo-candle.md` - Rust ML framework
- `hanzo/zenlm.md` - Text-only Zen models

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: multimodal, vision, audio, ai
**Prerequisites**: Python, PyTorch, ML fundamentals
