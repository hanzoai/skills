# Hanzo Mugen - Audio Generation Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-koe.md`, `hanzo/hanzo-engine.md`, `hanzo/hanzo-ml.md`

## Overview

Mugen is a **PyTorch framework for deep learning research on audio generation**. It provides training and inference code for multiple state-of-the-art generative audio models: text-to-music (MusicGen), text-to-sound (AudioGen), neural audio codecs (EnCodec), multi-band diffusion decoding, non-autoregressive generation (MAGNeT), audio watermarking (AudioSeal), and joint conditioning (JASCO).

The core architecture is a **transformer language model** operating over quantized audio tokens from EnCodec, with text conditioning via T5 or CLAP embeddings. Model scales range from ~300M to ~3.3B parameters.

### Models Included

| Model | Task | Description |
|-------|------|-------------|
| **MusicGen** | Text-to-music | Controllable music generation with melody conditioning |
| **AudioGen** | Text-to-sound | General audio/sound effect generation |
| **EnCodec** | Audio codec | High-fidelity neural audio tokenizer |
| **Multi-Band Diffusion** | Decoder | Diffusion-based EnCodec decoder for improved quality |
| **MAGNeT** | Text-to-music/sound | Non-autoregressive masked generation |
| **AudioSeal** | Watermarking | Imperceptible audio watermarking |
| **MusicGen Style** | Style-to-music | Text + style conditioning |
| **JASCO** | Multi-conditioned | Chords, melodies, drums + text conditioning |

## When to use

- Text-conditioned music or sound effect generation
- Training custom audio generation models on proprietary data
- Neural audio compression/tokenization (EnCodec)
- Audio watermarking for generated content
- Research into transformer-based audio generative models

## Hard requirements

1. **Python 3.9+** with PyTorch 2.1
2. **GPU recommended** for training and fast inference
3. **ffmpeg** installed (system or conda)
4. **xformers < 0.0.23** for memory-efficient attention

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/mugen` |
| Branch | `main` |
| Language | Python |
| Package | `audiocraft` v1.4.0a2 |
| Install | `pip install -e .` |
| Train | Hydra-based: `dora run` |
| Test | `make tests` |
| Lint | `make linter` |
| License | MIT (code), CC-BY-NC 4.0 (weights) |

## Architecture

### Transformer Language Model

The core generative model is an autoregressive transformer operating on EnCodec audio tokens:

```
Text Prompt: "epic orchestral music with drums"
    |
    v
┌────────────────────────────────┐
│ Conditioning                    │
│  T5 / CLAP text embeddings     │
│  + optional: chroma, chords,   │
│    drums, melody, style        │
│  ConditionFuser (cross-attn    │
│    or prepend or input_interp) │
└──────────┬─────────────────────┘
           |
           v
┌────────────────────────────────┐
│ Streaming Transformer LM       │
│  Codebook pattern: parallel    │
│  n_q codebook streams (8)      │
│  card: 1024 (codebook size)    │
│                                │
│  Configurable scale:           │
│    small:  dim=512,  8 layers  │
│    base:   dim=1024, 24 layers │
│    medium: dim=1536, 32 layers │
│    large:  dim=2048, 48 layers │
│                                │
│  Positional: sin/rope/sin_rope │
│  Attention: causal, streaming  │
│  Optional: xformers flash attn │
│  CFG: classifier-free guidance │
└──────────┬─────────────────────┘
           |
           v
┌────────────────────────────────┐
│ EnCodec Decoder                │
│  (or Multi-Band Diffusion)     │
│  Audio tokens -> waveform      │
│  16kHz (sound) / 32kHz (music) │
└────────────────────────────────┘
```

### Model Scale Configurations

| Scale | dim | Heads | Layers | ~Params |
|-------|-----|-------|--------|---------|
| small | 512 | 8 | 8 | ~300M |
| base | 1024 | 16 | 24 | ~1.5B |
| medium | 1536 | 24 | 32 | ~2.2B |
| large | 2048 | 32 | 48 | ~3.3B |

### EnCodec (Neural Audio Codec)

- SEANet encoder/decoder architecture
- Residual Vector Quantization (RVQ) with configurable codebooks (n_q=8 default)
- 1024 codes per codebook
- Supports 16kHz and 32kHz sample rates
- Streaming capable for real-time applications

## One-file quickstart

### Music Generation

```python
import torchaudio
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=8)

# Text-conditioned
wav = model.generate(["epic cinematic orchestral soundtrack"])

# Melody-conditioned
melody, sr = torchaudio.load("melody.wav")
wav = model.generate_with_chroma(
    ["epic cinematic version"],
    melody[None].expand(1, -1, -1),
    sr,
)

# Save output
torchaudio.save("output.wav", wav[0].cpu(), model.sample_rate)
```

### Sound Effect Generation

```python
from audiocraft.models import AudioGen

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=5)

wav = model.generate(["sirens and a humming engine approach and pass"])
```

### Audio Compression

```python
from audiocraft.models import EncodecModel

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

wav = torch.randn(1, 1, 24000)  # 1 second at 24kHz
encoded = model.encode(wav)
decoded = model.decode(encoded)
```

## Training

Training uses Facebook's Dora experiment manager with Hydra configs:

```bash
# Install with training deps
pip install -e '.[dev]'

# Run training (example: MusicGen base at 32kHz)
dora run solver=musicgen/musicgen_base_32khz

# Custom training config
dora run solver=musicgen/musicgen_base_32khz \
    dataset.batch_size=8 \
    optim.epochs=100 \
    transformer_lm.dim=1024 \
    transformer_lm.num_layers=24
```

### Hydra Config Structure

```
config/
├── augmentations/default.yaml
├── conditioner/
│   ├── text2music.yaml       # T5 text conditioning
│   ├── text2sound.yaml       # Text for sound generation
│   ├── chroma2music.yaml     # Chromagram conditioning
│   ├── chords2music.yaml     # Chord conditioning
│   ├── drums2music.yaml      # Drum track conditioning
│   ├── style2music.yaml      # Style embedding conditioning
│   └── jasco_chords_drums.yaml
├── dset/audio/               # Dataset configs
├── model/
│   ├── encodec/              # Codec model configs
│   └── lm/
│       ├── default.yaml      # Base LM config
│       └── model_scale/      # small/base/medium/large
└── config.yaml               # Root config
```

## Dependencies

```
torch==2.1.0
torchaudio>=2.0.0,<2.1.2
torchvision==0.16.0
av==11.0.0
einops
flashy>=0.0.1
hydra-core>=1.1
xformers<0.0.23
transformers>=4.31.0
demucs
librosa
soundfile
gradio
encodec
torchdiffeq
```

## Project Structure

```
mugen/
├── audiocraft/
│   ├── __init__.py               # v1.4.0a2
│   ├── models/
│   │   ├── musicgen.py           # MusicGen model
│   │   ├── audiogen.py           # AudioGen model
│   │   ├── encodec.py            # EnCodec codec
│   │   ├── multibanddiffusion.py # Multi-band diffusion decoder
│   │   ├── magnet.py             # MAGNeT model
│   │   ├── jasco.py              # JASCO multi-conditioned
│   │   ├── watermark.py          # AudioSeal watermarking
│   │   ├── lm.py                 # Core transformer LM
│   │   ├── lm_magnet.py          # MAGNeT LM variant
│   │   ├── flow_matching.py      # Flow matching model
│   │   ├── builders.py           # Model factory
│   │   └── loaders.py            # Checkpoint loading
│   ├── modules/
│   │   ├── transformer.py        # StreamingTransformer
│   │   ├── conditioners.py       # Text/audio/chroma conditioners
│   │   ├── codebooks_patterns.py # Parallel/delay codebook patterns
│   │   ├── seanet.py             # SEANet encoder/decoder
│   │   ├── conv.py               # Causal/streaming convolutions
│   │   ├── rope.py               # Rotary positional embeddings
│   │   └── streaming.py          # Streaming inference support
│   ├── solvers/
│   │   ├── musicgen.py           # MusicGen training solver
│   │   ├── audiogen.py           # AudioGen training solver
│   │   ├── compression.py        # EnCodec training solver
│   │   ├── diffusion.py          # Diffusion training solver
│   │   ├── magnet.py             # MAGNeT training solver
│   │   ├── jasco.py              # JASCO training solver
│   │   └── watermark.py          # Watermark training solver
│   ├── data/                     # Dataset loading and processing
│   ├── losses/                   # Training losses (balancer, STFT, loudness)
│   ├── metrics/                  # FAD, KLD, CLAP, PESQ, ViSQOL
│   ├── adversarial/              # GAN discriminators (MPD, MSD, MSSTFTD)
│   ├── optim/                    # Optimizers, schedulers, EMA, FSDP
│   ├── quantization/             # RVQ implementation
│   ├── grids/                    # Dora experiment grids
│   ├── utils/                    # Utilities, caching, checkpoints
│   └── train.py                  # Training entry point
├── config/                       # Hydra configs
├── assets/                       # Example audio files
├── setup.py
├── Makefile
└── requirements.txt
```

## Related Skills

- `hanzo/hanzo-koe.md` - Text-to-dialogue model (speech-specific)
- `hanzo/hanzo-engine.md` - Model serving infrastructure
- `hanzo/hanzo-ml.md` - Rust ML framework
- `hanzo/hanzo-candle.md` - Candle audio model support (Whisper, EnCodec)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: audio, music-generation, sound-effects, codec, tts, watermarking
**Prerequisites**: Python, PyTorch, audio processing, Hydra config
