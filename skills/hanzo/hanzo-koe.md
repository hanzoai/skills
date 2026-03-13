# Hanzo Koe - Text-to-Dialogue Model

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-mugen.md`, `hanzo/hanzo-engine.md`, `hanzo/hanzo-ml.md`

## Overview

Koe is a **1.6B parameter text-to-speech model** that directly generates realistic multi-speaker dialogue from text transcripts. It supports two-speaker conversations with speaker tags (`[S1]`, `[S2]`), non-verbal sounds (laughter, coughs, sighs), and voice cloning from audio prompts.

The architecture uses an **encoder-decoder transformer** with RoPE positional embeddings. The encoder processes text tokens, and the decoder autoregressively generates 9-channel audio codes that are decoded by the Descript Audio Codec (DAC) into 44.1kHz waveforms.

### Key Capabilities

- **Multi-speaker dialogue**: Generate conversations between two speakers from tagged transcripts
- **Non-verbal generation**: Produce laughter, coughs, throat clearing, sighs, gasps, etc.
- **Voice cloning**: Condition output on reference audio (5-10 seconds) for speaker consistency
- **Real-time generation**: Up to 2.2x realtime factor with torch.compile on RTX 4090

## When to use

- Generating dialogue audio from scripts with speaker annotations
- Voice cloning and style transfer from reference audio
- Adding non-verbal vocalizations to synthesized speech
- Multi-speaker TTS without per-speaker fine-tuning

## Hard requirements

1. **Python 3.10+** with PyTorch 2.6
2. **GPU**: CUDA or MPS (Apple Silicon with torch.compile disabled)
3. **VRAM**: ~10GB (float16/bfloat16), ~13GB (float32)
4. **HF_TOKEN**: Required for downloading model config from HuggingFace Hub

## Quick reference

| Item | Value |
|------|-------|
| Repo | `github.com/hanzoai/koe` |
| Branch | `main` |
| Language | Python |
| Parameters | 1.6B |
| Package | `nari-tts` |
| Sample Rate | 44.1kHz |
| Audio Channels | 9 (DAC codebooks) |
| Install | `pip install git+https://github.com/hanzoai/koe.git` |
| Run UI | `uv run app.py` |
| License | Apache 2.0 |

## Architecture

```
Text Input: "[S1] Hello. [S2] Hi there. (laughs)"
    |
    v
┌─────────────────────────────┐
│ Text Tokenizer               │
│  src_vocab_size: 128         │
│  text_length: padded to 128 │
└──────────┬──────────────────┘
           |
           v
┌─────────────────────────────┐
│ Encoder (Transformer)        │
│  n_layer layers              │
│  n_head attention heads      │
│  head_dim per head           │
│  RoPE positional encoding    │
│  RMSNorm normalization       │
│  DenseGeneral (tensordot)    │
└──────────┬──────────────────┘
           |
           v
┌─────────────────────────────┐
│ Decoder (Transformer)        │
│  Cross-attention to encoder  │
│  9-channel audio code output │
│  tgt_vocab_size: 1028        │
│  Special tokens:             │
│    EOS=1024, PAD=1025,       │
│    BOS=1026                  │
│  Delay pattern: [0,8..15]    │
└──────────┬──────────────────┘
           |
           v
┌─────────────────────────────┐
│ Descript Audio Codec (DAC)   │
│  Decode 9 codebook streams   │
│  Output: 44.1kHz waveform    │
└─────────────────────────────┘
```

### Audio Delay Pattern

The decoder uses a delay pattern across 9 channels: `[0, 8, 9, 10, 11, 12, 13, 14, 15]`. This staggers codebook generation to improve coherence across the multi-stream output. Token positions are shifted via pre-computed delay/revert indices.

### Inference Sampling

Token generation uses configurable sampling:
- **Temperature**: Controls randomness (0.0 = greedy argmax)
- **Top-p**: Nucleus sampling with cumulative probability threshold
- **Top-k**: Limits candidate tokens to top-k logits
- **EOS suppression**: EOS token masked unless it has the highest logit

## One-file quickstart

```python
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Koe is a text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now."

output = model.generate(text, use_torch_compile=True, verbose=True)
model.save_audio("dialogue.mp3", output)
```

### Voice Cloning

```python
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

# Provide transcript of reference audio before generation text
text = "[S1] This is the reference speaker talking. [S2] And this is another voice. [S1] Now generate new content in my voice."

output = model.generate(
    text,
    audio_prompt="reference_audio.mp3",  # 5-10 seconds
    use_torch_compile=True,
)
model.save_audio("cloned.mp3", output)
```

### Apple Silicon (MPS)

```python
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
# torch.compile not supported on MPS
output = model.generate(text, use_torch_compile=False, verbose=True)
```

## Non-Verbal Tags

Recognized tags (place in transcript where desired):

`(laughs)`, `(clears throat)`, `(sighs)`, `(gasps)`, `(coughs)`, `(singing)`, `(sings)`, `(mumbles)`, `(beep)`, `(groans)`, `(sniffs)`, `(claps)`, `(screams)`, `(inhales)`, `(exhales)`, `(applause)`, `(burps)`, `(humming)`, `(sneezes)`, `(chuckle)`, `(whistles)`

## Performance

| Precision | Realtime Factor (compile) | Realtime Factor (no compile) | VRAM |
|-----------|---------------------------|------------------------------|------|
| bfloat16 | 2.1x | 1.5x | ~10GB |
| float16 | 2.2x | 1.3x | ~10GB |
| float32 | 1.0x | 0.9x | ~13GB |

Benchmarked on RTX 4090.

## Dependencies

```
descript-audio-codec>=1.0.0
gradio>=5.25.2
huggingface-hub>=0.30.2
numpy>=2.2.4
pydantic>=2.11.3
safetensors>=0.5.3
soundfile>=0.13.1
torch==2.6.0
torchaudio==2.6.0
triton==3.2.0 (Linux)
```

## Project Structure

```
koe/
├── dia/
│   ├── __init__.py
│   ├── audio.py          # Delay pattern indices, audio processing
│   ├── config.py          # DiaConfig (Pydantic): data, encoder, decoder, model
│   ├── layers.py          # DiaModel: DenseGeneral, transformer blocks, attention
│   ├── model.py           # Dia: main model class, generate(), save_audio()
│   ├── state.py           # Inference state: KV cache, decoder/encoder state
│   └── static/images/     # Banner assets
├── docker/
│   ├── Dockerfile.cpu
│   └── Dockerfile.gpu
├── example/
│   ├── benchmark.py
│   ├── simple.py
│   ├── simple-mac.py
│   ├── simple_batch.py
│   ├── voice_clone.py
│   └── voice_clone_batch.py
├── app.py                 # Gradio web UI
├── cli.py                 # CLI interface
├── pyproject.toml
└── uv.lock
```

## Generation Guidelines

- Input text should correspond to 5-20 seconds of audio for best quality
- Always begin with `[S1]` and alternate between `[S1]` and `[S2]`
- Use non-verbal tags sparingly; overuse causes artifacts
- For voice cloning: provide 5-10 second reference audio with matching transcript
- End text with the second-to-last speaker tag to improve trailing audio quality
- 1 second of audio is approximately 86 tokens

## Related Skills

- `hanzo/hanzo-mugen.md` - Audio generation (music, sound effects)
- `hanzo/hanzo-engine.md` - Model serving infrastructure
- `hanzo/hanzo-ml.md` - Rust ML framework

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: tts, dialogue, speech-synthesis, voice-cloning, audio
**Prerequisites**: Python, PyTorch, audio processing concepts
