---
name: hanzo-studio
description: "Visual AI engine for image, video, audio, and 3D generation. Node-based Stable Diffusion workflows with Flux, SDXL, SD3, ControlNet, and LoRA support."
metadata: { "bot": { "requires": { "bins": ["curl"] } } }
---

# Hanzo Studio — Visual AI Engine

Node-based AI engine for image, video, audio, and 3D generation. Built on ComfyUI with support for Flux, SDXL, SD3, Stable Cascade, and more.

## Quick Start

```bash
# Docker
docker run -p 8188:8188 ghcr.io/hanzoai/studio:latest --listen

# With GPU
docker run --gpus all -p 8188:8188 ghcr.io/hanzoai/studio:latest --listen
```

## API Usage

```python
import requests
import json

STUDIO_URL = "http://localhost:8188"

# Queue a workflow
workflow = {
    "prompt": {
        "3": {"class_type": "KSampler", "inputs": {...}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a beautiful sunset"}},
        ...
    }
}

response = requests.post(f"{STUDIO_URL}/prompt", json=workflow)
prompt_id = response.json()["prompt_id"]

# Check status
status = requests.get(f"{STUDIO_URL}/history/{prompt_id}").json()
```

## WebSocket (Real-time Progress)

```python
import websocket

ws = websocket.create_connection(f"ws://localhost:8188/ws")
# Receive progress updates as workflow executes
while True:
    result = ws.recv()
    data = json.loads(result)
    if data["type"] == "progress":
        print(f"Progress: {data['data']['value']}/{data['data']['max']}")
    elif data["type"] == "executed":
        print("Done!")
        break
```

## Supported Models

### Image Generation

- **Flux** (Dev, Schnell) — Latest diffusion model
- **SDXL** — Stable Diffusion XL
- **SD3** — Stable Diffusion 3
- **Stable Cascade** — Multi-stage generation
- **Pixart**, **HunyuanDiT**, **AuraFlow**

### Video Generation

- **Stable Video Diffusion**
- **Mochi**, **LTX-Video**
- **Hunyuan Video**

### Audio Generation

- **Stable Audio**
- **ACE Step**

### 3D Generation

- **Hunyuan3D 2.0**

## Features

- Node-based visual workflow editor
- GPU auto-offloading (works with 1GB VRAM)
- LoRA, ControlNet, T2I-Adapter support
- Inpainting and outpainting
- Area composition
- Model merging
- Async queue system
- Workflow save/load/share

## Port

- API/WebUI: `8188`

## Environment Variables

```bash
COMFYUI_PORT=8188
COMFYUI_MODELS_PATH=/models     # Model storage path
```
