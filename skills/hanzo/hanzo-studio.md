# Hanzo Studio - Visual AI Workflow Engine

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-engine.md`, `hanzo/hanzo-node.md`, `hanzo/hanzo-flow.md`

## Overview

Hanzo Studio is a **visual node-based AI workflow engine** for building, testing, and deploying AI pipelines. Fork of ComfyUI with Hanzo branding, custom nodes, and cloud deployment. Live at `studio.hanzo.ai`.

### Why Hanzo Studio?

- **Visual workflows**: Drag-and-drop AI pipeline builder
- **Custom nodes**: Extend with Python — any model, any tool
- **API mode**: Run workflows programmatically via REST API
- **Self-hostable**: Docker, K8s, or bare metal
- **White-label**: Full branding customization

### OSS Base

Fork of **ComfyUI** (`comfyanonymous/ComfyUI`). Repo: `hanzoai/studio`.

## When to use

- Building visual AI workflows (image gen, text processing, pipelines)
- Creating custom AI node integrations
- Running ComfyUI workflows with Hanzo branding
- Deploying visual AI tools for non-technical users
- White-labeling a visual AI platform

## Hard requirements

1. **Python 3.10+** with pip
2. **Docker** for containerized deployment
3. Port **8188** available

## Quick reference

| Item | Value |
|------|-------|
| UI | `https://studio.hanzo.ai` |
| Port | 8188 |
| Image | `ghcr.io/hanzoai/studio:latest` |
| Repo | `github.com/hanzoai/studio` |
| Branch | `main` |
| Upstream | `comfyanonymous/ComfyUI` |

## One-file quickstart

### Docker

```bash
docker run -d --name hanzo-studio \
  -p 8188:8188 \
  --cpus=1 --memory=2g \
  ghcr.io/hanzoai/studio:latest \
  --cpu --listen 0.0.0.0
```

### API mode (run workflow)

```bash
curl -X POST http://localhost:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {
        "class_type": "KSampler",
        "inputs": {
          "seed": 42,
          "steps": 20,
          "cfg": 7.0,
          "sampler_name": "euler",
          "scheduler": "normal"
        }
      }
    }
  }'
```

## Core Concepts

### Branding Approach

**CRITICAL**: Never do blanket `sed 's/ComfyUI/Hanzo Studio/g'` on minified JS — it breaks class definitions, property assignments, and dynamic imports.

Use `branding/patch_frontend.py` (Python, context-aware):
- Only touches display strings, URLs, and quoted contexts
- Logo SVGs replaced directly (comfy-logo → hanzo-logo.svg)
- Favicon generated as ICO with H mark bitmap

### Custom Node Development

```python
# custom_nodes/hanzo_inference.py
class HanzoInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["zen-70b", "zen-32b", "zen-14b"],),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Hanzo AI"

    def inference(self, prompt, model, temperature):
        import requests
        resp = requests.post("https://api.hanzo.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['HANZO_API_KEY']}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": temperature})
        return (resp.json()["choices"][0]["message"]["content"],)

NODE_CLASS_MAPPINGS = {"HanzoInference": HanzoInference}
NODE_DISPLAY_NAME_MAPPINGS = {"HanzoInference": "Hanzo AI Inference"}
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-studio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hanzo-studio
  template:
    spec:
      containers:
      - name: studio
        image: ghcr.io/hanzoai/studio:latest
        args: ["--cpu", "--listen", "0.0.0.0"]
        ports:
        - containerPort: 8188
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: "1"
            memory: 2Gi
```

## White-Label

1. Fork `hanzoai/studio`
2. Edit `branding/patch_frontend.py` with your logo/colors
3. Run `python branding/patch_frontend.py` during Docker build
4. Deploy with your domain

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Broken class names in UI | Used sed on minified JS | Use patch_frontend.py only |
| OOM on large workflows | Insufficient memory | Increase K8s memory limit |
| Custom nodes not loading | Wrong directory | Place in `custom_nodes/` |

## Related Skills

- `hanzo/hanzo-engine.md` - Rust inference engine for backends
- `hanzo/hanzo-flow.md` - Alternative workflow builder
- `hanzo/hanzo-chat.md` - LLM API for custom nodes

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: comfyui, visual-ai, workflows, studio
**Prerequisites**: Python, Docker, AI pipeline concepts
