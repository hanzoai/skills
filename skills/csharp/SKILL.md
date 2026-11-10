---
name: csharp
description: "Build .NET 8+, ASP.NET Core, Unity, and Godot 4 .NET applications on Hanzo Cloud (https://api.hanzo.ai) with the official Hanzo C# SDK (Hanzo 8.5.1027 from hanzo-csharp/sdk). Load when installing or calling the Hanzo NuGet package for Zen 6 chat completions (/v1/chat/completions), Kai zero-output-token decisions (/v1/decisions), Zen Voice binary MP3 synthesis (PostAudioSpeechBytes on /v1/audio/speech), revisioned Cloud KV state and game saves (KvApi on /v1/kv), Cloud Agents (PostAgentByRefRun on /v1/agent), IAM/KMS/Commerce APIs, or Cloud MCP tools (/v1/mcp)."
license: MIT
---

# C# / .NET SDK

`Hanzo` (`8.5.1027`, namespace `Hanzo.AI`) is the official C# client for `https://api.hanzo.ai`. It is generated from the live Hanzo Cloud OpenAPI specification (`38` service APIs across AI, Decisions, Cloud KV, Agents, Base, IAM, KMS, Commerce, Console, Platform, Tasks, O11y, and MCP) and extended with handwritten helpers for `Gateway.Configuration()`, agent run execution (`PostAgentByRefRunAsync`), and binary MP3 speech synthesis (`PostAudioSpeechBytesAsync`).

## Docs & Repositories

- **C# / .NET SDK Guide**: [docs.hanzo.ai/docs/sdks/csharp](https://docs.hanzo.ai/docs/sdks/csharp)
- **Godot 4 & Game Backend Guide**: [docs.hanzo.ai/docs/sdks/godot](https://docs.hanzo.ai/docs/sdks/godot)
- **Kai Decisions Wire Reference**: [docs.hanzo.ai/docs/decisions](https://docs.hanzo.ai/docs/decisions)
- **Full Docs Index**: [docs.hanzo.ai/llms.txt](https://docs.hanzo.ai/llms.txt)
- **Source, Releases & Runnable Slices**: [github.com/hanzo-csharp/sdk](https://github.com/hanzo-csharp/sdk)

## Install & Configure

Target `.NET 8.0+` (`net8.0`). Install from NuGet or GitHub Packages (`https://nuget.pkg.github.com/hanzo-csharp/index.json`):

```bash
dotnet add package Hanzo --version 8.5.1027
```

`Gateway.Configuration()` reads `HANZO_API_KEY` (falling back to `OPENAI_API_KEY`) and `HANZO_BASE_URL` (default `https://api.hanzo.ai`), wiring Bearer auth automatically:

```csharp
using Hanzo.AI;
using Hanzo.AI.Api;
using Hanzo.AI.Client;

Configuration cfg = Gateway.Configuration();
cfg.Timeout = TimeSpan.FromSeconds(90);

var ai = new AiApi(cfg);
var kv = new KvApi(cfg);
var agents = new AgentApi(cfg);
var mcp = new OpenapiApi(cfg);
```

Keep secret `sk-` keys server-side or in local developer environments (`export HANZO_API_KEY="$(hanzo auth token)"`); never ship secret keys inside browser WASM bundles.

## Core Operations

### 1. Zen 6 Chat Completions (`/v1/chat/completions`)

Use `zen6-flash` for low-latency interactive replies, `zen6` for deep reasoning (`reasoning_effort = "low"` keeps interactive calls fast), and `zen6-coder` / `zen6-max` for heavy synthesis. Never use retired `zen-4` or `zen-5` IDs.

```csharp
object reply = await ai.PostChatCompletionsAsync(new
{
    model = "zen6-flash",
    max_tokens = 256,
    messages = new[]
    {
        new { role = "system", content = "Reply concisely." },
        new { role = "user", content = "Summarize three benefits of revisioned KV saves." }
    }
});
```

### 2. Kai Zero-Output-Token Decisions (`/v1/decisions`)

Use `model = "kai"` whenever C# code needs a bounded judgment over known options (routing, triage, moderation gates, NPC state selection, AI Director pacing) instead of parsing LLM prose. `usage.output_tokens` is `0`:

```csharp
object decision = await ai.PostDecisionsAsync(new
{
    model = "kai",
    state = new { health = 0.22, under_fire = true, cover_distance_m = 4.0 },
    questions = new
    {
        action = new
        {
            type = "choice",
            instructions = "Select the squad unit's immediate combat action.",
            criteria = new Dictionary<string, string>
            {
                ["attack"] = "Healthy unit with clear close-range shot.",
                ["flank"] = "Medium range engagement with lateral cover.",
                ["retreat"] = "Health below 25% while under active fire."
            }
        }
    }
});
```

### 3. Zen Voice Binary MP3 Synthesis (`/v1/audio/speech`)

Call the handwritten `PostAudioSpeechBytesAsync` extension (`AiApiSpeech.cs`) to receive raw MP3 `byte[]` from `zen-voice` without JSON string corruption:

```csharp
byte[] mp3Bytes = await ai.PostAudioSpeechBytesAsync(new
{
    model = "zen-voice",
    voice = "af_heart",
    input = "Checkpoint synced. Hostiles flanking from sector north."
});
```

In Godot 4 `.NET`, assign directly to an `AudioStreamPlayer3D`:

```csharp
player3D.Stream = new AudioStreamMP3 { Data = mp3Bytes };
player3D.Play();
```

### 4. Revisioned Cloud KV State & Game Saves (`/v1/kv/{bucket}/{key}`)

Bucket names MUST use 1–64 ASCII letters, digits, or underscores (`^[A-Za-z0-9_]{1,64}$`, never hyphens):

```csharp
string bucket = "game_saves_prod";
await kv.PostKvByBucketAsync(bucket, new { history = 10 });

await kv.PutKvByBucketAndKeyAsync(bucket, "slot_1", new
{
    value = "{\"scene\":\"bunker_02\",\"tension\":0.68,\"ammo\":15}"
});

object current = await kv.GetKvByBucketAndKeyAsync(bucket, "slot_1");
object revisions = await kv.GetKvByBucketAndKeyHistoryAsync(bucket, "slot_1");
await kv.DeleteKvByBucketAsync(bucket);
```

### 5. Cloud Agents & MCP (`/v1/agent`, `/v1/mcp`)

Create persistent agents on `zen6-flash`, trigger runs via `PostAgentByRefRunAsync`, and call the 50+ Cloud MCP tools over JSON-RPC `2.0` (`protocolVersion = "2026-07-28"`):

```csharp
object created = await agents.PostAgentAsync(new
{
    name = "world-sim",
    model = "zen6-flash",
    system = "Return compact JSON world-event updates."
});

object mcpInit = await mcp.PostMcpAsync(new
{
    jsonrpc = "2.0",
    id = 1,
    method = "initialize",
    @params = new
    {
        protocolVersion = "2026-07-28",
        capabilities = new { },
        clientInfo = new { name = "hanzo-csharp", version = "8.5.1027" }
    }
});
```

## Verification

Run both wire tests (offline) and live `https://api.hanzo.ai` integration tests in `hanzo-csharp/sdk`:

```bash
# Offline wire & serialization tests
HANZO_OFFLINE=1 dotnet test

# Live api.hanzo.ai integration suite + full game-backend slice
export HANZO_API_KEY="$(hanzo auth token)"
dotnet test
dotnet run --project examples -- game-backend
```

## Install

```bash
npx skills add https://hanzoskills.com/csharp.md
```
