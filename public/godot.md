---
name: godot
description: "Build Godot 4 games, VR/XR experiences, and serverless game backends powered by Hanzo AI (https://api.hanzo.ai/v1) using the official Hanzo Godot SDK (hanzoai/godot-sdk v1.1.0, pure GDScript) or Hanzo C# SDK (Hanzo 8.5.1027 for Godot 4 .NET). Load when wiring Kai zero-output-token NPC brains and AI Directors (/v1/decisions), Zen 6 character dialogue (/v1/chat/completions), Zen Voice spatial 3D MP3 radio synthesis (/v1/audio/speech -> AudioStreamMP3), revisioned Cloud KV saves and checkpoints (/v1/kv/{bucket}/{key}), Cloud Agents (/v1/agent), or Cloud MCP tools (/v1/mcp) into a Godot 4 project without stalling frame budgets."
license: MIT
---

# Godot 4 SDK & Game Backend

Power an entire Godot 4 game backend directly from `https://api.hanzo.ai/v1` with zero custom middleware servers. One `sk-` API key gives a Godot 4 project calibrated NPC & AI Director decisions via **Kai** (`/v1/decisions`), dynamic character dialogue via **Zen 6** (`/v1/chat/completions`), spatial 3D radio and character speech via **Zen Voice** (`/v1/audio/speech` -> `AudioStreamMP3`), revisioned cloud saves via **Cloud KV** (`/v1/kv/{bucket}/{key}`), persistent **Cloud Agents** (`/v1/agent`), and **Cloud MCP** tooling (`/v1/mcp`).

## Docs & Repositories

Read the live docs when wiring endpoints or upgrading SDK versions:

- **Godot 4 SDK Guide**: [docs.hanzo.ai/docs/sdks/godot](https://docs.hanzo.ai/docs/sdks/godot)
- **C# / .NET SDK Guide (Godot 4 .NET)**: [docs.hanzo.ai/docs/sdks/csharp](https://docs.hanzo.ai/docs/sdks/csharp)
- **Kai Decisions Wire Reference**: [docs.hanzo.ai/docs/decisions](https://docs.hanzo.ai/docs/decisions)
- **Full Docs Index**: [docs.hanzo.ai/llms.txt](https://docs.hanzo.ai/llms.txt)
- **Pure GDScript SDK (`v1.1.0`)**: [github.com/hanzoai/godot-sdk](https://github.com/hanzoai/godot-sdk)
- **C# NuGet SDK (`8.5.1027`)**: [github.com/hanzo-csharp/sdk](https://github.com/hanzo-csharp/sdk)

## Choose GDScript or Godot 4 .NET (C#)

- **Pure GDScript (`hanzoai/godot-sdk` `v1.1.0`)**: Zero external dependencies, works on desktop, mobile, XR/OpenXR headsets, and headless Linux servers. Copy `addons/hanzo_ai/` into your Godot 4 project and enable **Hanzo AI** in **Project -> Project Settings -> Plugins**. Provides `HanzoClient` (synchronous TLS `HTTPClient` + async non-blocking `HTTPRequest` helpers) and `HanzoAgent3D` (drop-in 3D spatial NPC node).
- **Godot 4 .NET (`Hanzo` `8.5.1027`)**: Add `<PackageReference Include="Hanzo" Version="8.5.1027" />` to your game's `.csproj` (`net8.0+`). Gives strongly typed `AiApi`, `KvApi`, `AgentApi`, `OpenapiApi`, and `PostAudioSpeechBytesAsync` for `AudioStreamMP3`.

```bash
# Pure GDScript addon install
git clone https://github.com/hanzoai/godot-sdk.git /tmp/hanzo-godot-sdk
cp -r /tmp/hanzo-godot-sdk/addons/hanzo_ai ./addons/hanzo_ai

# Or Godot 4 .NET C# NuGet package
dotnet add package Hanzo --version 8.5.1027
```

## Canonical Models Only

Always target current Hanzo production model IDs on `https://api.hanzo.ai/v1`:

- **Tactical NPC Brains & AI Director (`/v1/decisions`)**: `kai` (`usage.output_tokens == 0`, calibrated probabilities over named options).
- **Fast Real-Time NPC Bark & Radio Lines (`/v1/chat/completions`)**: `zen6-flash`.
- **Deep Narrative, Lore & Quest Dialogue (`/v1/chat/completions`)**: `zen6` (pass `"reasoning_effort": "low"` and `"max_tokens": 800` for interactive responsiveness).
- **Code / Procedural Generation (`/v1/chat/completions`)**: `zen6-coder` or `zen6-max`.
- **Spatial 3D Speech Synthesis (`/v1/audio/speech`)**: `zen-voice` (default voice `af_heart`, returns binary MP3 bytes).

Never use retired `zen-4` or `zen-5` identifiers.

## Full Serverless Godot Game Backend Pattern

```gdscript
extends Node

const HanzoClientScript := preload("res://addons/hanzo_ai/hanzo_client.gd")

var hanzo: RefCounted
var radio_3d: AudioStreamPlayer3D

func _ready() -> void:
    # Resolves HANZO_API_KEY from env or ~/.hanzo/auth.json
    hanzo = HanzoClientScript.new()
    radio_3d = AudioStreamPlayer3D.new()
    add_child(radio_3d)

func save_checkpoint(bucket: String, slot: String, state: Dictionary) -> Dictionary:
    # Bucket names MUST match ^[A-Za-z0-9_]{1,64}$ (underscores, never hyphens)
    hanzo.create_kv_bucket(bucket, 10)
    return hanzo.put_kv(bucket, slot, JSON.stringify(state))

func load_checkpoint(bucket: String, slot: String) -> Dictionary:
    var res: Dictionary = hanzo.get_kv(bucket, slot)
    if not bool(res.get("ok", false)):
        return {}
    var raw: String = str(res.get("data", {}).get("value", ""))
    var parsed: Variant = JSON.parse_string(raw)
    return parsed if parsed is Dictionary else {}

func decide_npc_tactic(world_state: Dictionary) -> String:
    var res: Dictionary = hanzo.decide(world_state, {
        "tactic": {
            "type": "choice",
            "instructions": "Pick the NPC squad's immediate tactical state.",
            "criteria": {
                "patrol": "No confirmed target or noise.",
                "investigate_noise": "Gunshot or footsteps heard, target unseen.",
                "stalk": "Target spotted beyond 15m; shadow quietly.",
                "attack": "Target visible within 15m and health is solid.",
                "flank": "Taking fire at medium range with cover nearby.",
                "retreat": "Health below 25% under active fire."
            }
        }
    })
    return str(res.get("data", {}).get("answers", {}).get("tactic", {}).get("choice", "patrol"))

func play_radio_transmission(prompt: String) -> void:
    var chat: Dictionary = hanzo.chat_completion(
        [
            {"role": "system", "content": "Give one terse 12-word tactical radio update."},
            {"role": "user", "content": prompt}
        ],
        "zen6-flash",
        {"max_tokens": 120, "temperature": 0.6}
    )
    var line: String = hanzo.extract_message_text(chat)
    if line == "":
        return
    var tts: Dictionary = hanzo.speech_mp3(line, "af_heart", "zen-voice")
    var stream: AudioStreamMP3 = tts.get("stream")
    if stream != null:
        radio_3d.stream = stream
        radio_3d.play()
```

## Non-Blocking Frame Budgets (Desktop & OpenXR / VR)

- **Never stall `_process` or `_physics_process`**: Synchronous `HanzoClient` methods poll `HTTPClient` until TLS + response complete. In live gameplay—especially 90–120 Hz OpenXR/VR—either use `HanzoClient`'s async `Node`-parented `HTTPRequest` methods (`decide_async`, `chat_completion_async`, `speech_mp3_async`) or dispatch synchronous calls onto Godot's `WorkerThreadPool.add_task(...)` and marshal results back with `call_deferred(...)`.
- **SceneTree CLI TLS initialization**: When writing headless test harnesses (`extends SceneTree`, invoked via `godot --headless --path . -s`), never call `TLSOptions.client()` inside `_init()`. Godot loads CA certificates right before `_initialize()`, so run live HTTPS assertions inside `_initialize()`.
- **Cloud KV bucket naming**: `/v1/kv/{bucket}/{key}` accepts `[A-Za-z0-9_]{1,64}` for `{bucket}`. Replace hyphens with underscores (`dead_air_saves`, never `dead-air-saves`), and keep a local `user://hanzo_kv/` JSON mirror so saves work offline and sync when online.
- **Avoid persona conflicts on `zen6`**: Frame NPC system prompts as roleplay context (`"Speak in character as Dispatch Unit 4..."`) rather than `"You are Dispatch Unit 4, not an AI"`, and pass `"reasoning_effort": "low"` when calling flagship `zen6` so reasoning budget goes into dialogue rather than identity resolution.
- **Deterministic offline fallback**: Always gate live calls behind `has_auth()` (or `HANZO_OFFLINE=1`) with deterministic local heuristics so CI tests and offline single-player runs never break without network access.

## Headless Verification

Verify any Godot integration headlessly against `https://api.hanzo.ai/v1`:

```bash
# Offline unit & schema assertions (no key required)
HANZO_OFFLINE=1 godot --headless --path . -s res://tests/sdk_check.gd

# Live end-to-end assertions (models, kai, zen6, zen-voice MP3, KV revisions, agents, MCP)
export HANZO_API_KEY="$(hanzo auth token)"
godot --headless --path . -s res://tests/sdk_check.gd
godot --headless --path . -s res://examples/game_backend.gd
```

## Install

```bash
npx skills add https://hanzoskills.com/godot.md
```
