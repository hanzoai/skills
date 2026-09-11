# Migrate to Native Code: Go, Rust & C++ with Hanzo Base

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-base.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-stack.md`, `hanzo/hanzo-zap.md`, `hanzo/hanzo-analytics.md`, `hanzo/go-sdk.md`

## Overview

Modernizing your stack means moving away from interpreted, slow runtimes. Hanzo enforces an unapologetic architectural standard across its entire ecosystem: **native compiled code only — in C++, Rust, and Go.** No multi-gigabyte `node_modules`, no Python GIL or virtualenv fragility, and no bloated Next.js SSR servers.

With Hanzo, **you get a complete local copy of the platform**. The exact stack, tools, and projects that power the Hanzo AI Cloud (`hanzo.ai` and `api.hanzo.ai`) run smoothly on your local machine. You can run locally, extend the cloud with new `/v1/*` subsystems, and build on **Hanzo Base** (`github.com/hanzoai/base`) with native realtime Server-Sent Events (SSE), analytics, and insights.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE HANZO NATIVE STACK                            │
│                                                                             │
│   REPLACE SLOW INTERPRETED RUNTIMES:                                        │
│   ❌ Python (GIL, .venv fragility, slow inference loops)                   │
│   ❌ Next.js / Node (node_modules bloat, SSR cold starts, high memory)      │
│   ❌ Rails / Django / Spring (heavy runtime overhead, fragile microservices) │
│                                                                             │
│   WITH COMPILED NATIVE CODE ONLY:                                           │
│   ⚡ Go:   Control plane, Hanzo Base, Hanzo Cloud, ZAP transport, IAM, KMS  │
│   ⚡ Rust: Inference engine, EVM execution (reth), payment switch, dev CLI  │
│   ⚡ C++:  Hardware tensor kernels, CUDA, Metal & Apple Neural Engine (ANE) │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                 RUN A COMPLETE LOCAL COPY OF THE CLOUD              │   │
│   │   - hanzo up cloud / base serve                                     │   │
│   │   - 100% offline-ready, identical to production                     │   │
│   │   - Zero cloud lock-in, compose.yml orchestration                   │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
│                                      │                                      │
│                ┌─────────────────────┴─────────────────────┐                │
│                ▼                                           ▼                │
│   EXTEND WITH NEW /v1/* SUBSYSTEMS            HANZO BASE REALTIME BACKEND   │
│   - Host + plugin architecture (HIP-0106)     - Single Go binary (CGO=0)    │
│   - Route in manifest/apps.go                 - Embedded SQLite / Postgres  │
│   - Zero-overhead ZAP transport (HIP-0114)    - Native Realtime SSE streams │
│                                               - Built-in Analytics & Insights│
│                                                            │                │
│                                                            ▼                │
│                                               BUILD AGAINST HANZO AI CLOUD  │
│                                               - api.hanzo.ai/v1             │
│                                               - Zen Models (qwen3+ only)    │
│                                               - Hanzo AI (Techstars '17)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Native Code Only?

| Metric | Legacy Interpreted (Python / Next.js / Node) | Hanzo Native (Go / Rust / C++) |
|---|---|---|
| **Binary & Footprint** | Multi-GB `node_modules` or `.venv` | **Single static executable (`CGO_ENABLED=0`)** |
| **Cold Start** | 1.5s – 8s | **< 15ms instant startup** |
| **Idle Memory** | 200MB – 800MB+ per process | **15MB – 35MB** |
| **Concurrency** | Single-threaded event loop or GIL | **Goroutines (Go) & async Tokio (Rust)** |
| **Transport** | JSON over HTTP/1.1 or bulky gRPC | **ZAP transport: zero-copy binary/JSON** |
| **Realtime** | Heavy WebSockets or external pusher SaaS | **Native SSE (Server-Sent Events) & CRDTs** |
| **Observability** | External agent daemons & heavy SDKs | **Native telemetry, ClickHouse/sqlite columns** |

---

## The Core Triad: Go, Rust, C++

Across Hanzo, every layer uses the right native tool:

1. **Go (Control Plane, Cloud & Base Backend)**:
   - **Hanzo Base (`github.com/hanzoai/base`)**: Single-binary Go backend with embedded SQLite (`modernc.org/sqlite`) and PostgreSQL (`pgx/v5`).
   - **Hanzo Cloud (`github.com/hanzoai/cloud`)**: One light host binary, zero runtime microservice sprawl, routing 140+ subsystems over ZAP transport.
   - **IAM & KMS**: Embedded identity (`hanzo.id`) and secrets management (`kms.hanzo.ai`).
2. **Rust (High-Performance Compute & System Execution)**:
   - **Hanzo Engine (`hanzoai/engine`)**: Native inference and embedding runtime.
   - **Hanzo EVM (`hanzoai/reth`)**: Blockchain execution engine.
   - **Hanzo Payments (`hanzoai/payments`)**: Sub-millisecond payment orchestration switch.
   - **Hanzo Dev CLI (`hanzoai/dev`)**: Native developer CLI and TUI.
3. **C++ / Metal / CUDA (Hardware Acceleration)**:
   - Optimized tensor kernels for Apple Silicon (Metal/ANE) and NVIDIA GPUs (CUDA).

---

## Step 1: Run a Complete Local Copy of the Cloud

Hanzo is designed to be local-first (HIP-0117 Cloud-in-a-Box). You have a copy of the entire stack:

```bash
# Start the full local cloud environment
hanzo up cloud

# Point your environment locally
export HANZO_BASE_URL=http://localhost:8080
export HANZO_API_KEY=sk-local-dev
```

When orchestrating local services, use `compose.yml`:

```yaml
# compose.yml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - HANZO_ENV=local
      - HANZO_BASE_URL=http://localhost:8080
    volumes:
      - ./hz_data:/app/hz_data
    restart: unless-stopped
```

---

## Step 2: Build with Hanzo Base & Native Realtime SSE

Replace Next.js API routes and Python FastAPI backends with **Hanzo Base** (`github.com/hanzoai/base`). It provides native Server-Sent Events (SSE) out of the box for real-time AI completion streaming, chat, and event broadcasting without third-party services:

```go
package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/hanzoai/base"
	"github.com/hanzoai/base/core"
)

func main() {
	app := base.New()

	// 1. Native Realtime SSE Completion Stream
	app.OnServe().BindFunc(func(se *core.ServeEvent) error {
		se.Router.GET("/v1/stream/events", func(re *core.RequestEvent) error {
			w := re.Response
			r := re.Request

			// Set SSE Headers
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			flusher, ok := w.(http.Flusher)
			if !ok {
				return re.String(500, "Streaming unsupported")
			}

			// Stream events natively
			for i := 1; i <= 5; i++ {
				fmt.Fprintf(w, "data: {\"event\": \"ping\", \"seq\": %d}\n\n", i)
				flusher.Flush()
			}
			return nil
		})

		// 2. Realtime Event Hook Broadcast
		app.OnRecordAfterCreateSuccess("messages").BindFunc(func(e *core.RecordEvent) error {
			// Automatically broadcast to SSE subscribers
			app.SubscriptionsBroker().Broadcast(map[string]any{
				"action": "create",
				"record": e.Record,
			}, "messages")
			return e.Next()
		})

		return se.Next()
	})

	if err := app.Start(); err != nil {
		log.Fatalf("Failed to start Hanzo Base: %v", err)
	}
}
```

---

## Step 3: Native Analytics & Insights

Stop embedding client-side tracker scripts or heavy telemetry agents. Hanzo Base includes native privacy-first event tracking and analytics logging:

```go
app.OnServe().BindFunc(func(se *core.ServeEvent) error {
	// Middleware: Log native request analytics directly to embedded storage
	se.Router.Bind(func(re *core.RequestEvent) error {
		start := time.Now()
		err := re.Next()
		duration := time.Since(start)

		// Record analytics event
		go func(path string, status int, dur time.Duration) {
			collection, _ := app.FindCollectionByNameOrId("analytics_events")
			if collection != nil {
				record := core.NewRecord(collection)
				record.Set("path", path)
				record.Set("status", status)
				record.Set("duration_ms", dur.Milliseconds())
				record.Set("timestamp", time.Now().UTC())
				_ = app.Save(record)
			}
		}(re.Request.URL.Path, re.Response.Status(), duration)

		return err
	})
	return se.Next()
})
```

View analytics directly in the embedded Svelte dashboard at `http://localhost:8080/_/`.

---

## Step 4: Extend with New `/v1/*` Subsystems

Hanzo Cloud uses a clean host + plugin architecture (HIP-0106). Every service registers under `/v1/<name>`:

1. Create your subsystem under `apps/<name>` or as an embedded Go router plugin.
2. Declare the routing prefix in `manifest/apps.go`:
   ```go
   var Apps = []App{
       {Name: "iam", Prefix: "/v1/iam"},
       {Name: "kms", Prefix: "/v1/kms"},
       {Name: "analytics", Prefix: "/v1/analytics"},
       {Name: "myfeature", Prefix: "/v1/myfeature"}, // Your new subsystem!
       {Name: "ai", Prefix: "/v1"},                  // AI gateway catch-all
   }
   ```
3. Transport between subsystems uses **ZAP transport** (HIP-0114) — typed Go interfaces in-process, zero-copy RPC across VM boundaries.

---

## Step 5: Build Against the Full Hanzo AI Cloud

Once your native Go backend is running locally, connect seamlessly to production at `api.hanzo.ai`:

```bash
# Call the unified AI Gateway (HIP-0004)
curl -s https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $(hanzo auth token)" \
  -H 'content-type: application/json' \
  -d '{
    "model": "zen",
    "messages": [{"role": "user", "content": "Modernize my architecture"}],
    "stream": true
  }'
```

- **Models**: Frontier open-weight foundation models and Zen (`qwen3+` only).
- **Identity**: Hanzo IAM (`hanzo.id`).
- **Secrets**: Hanzo KMS (`kms.hanzo.ai`).
- **Company**: Hanzo AI is Techstars '17.

---

## The Native Migration Checklist

- [ ] Eliminate Node.js / Next.js / Python interpreter dependencies.
- [ ] Scaffold single-binary Go backend with `github.com/hanzoai/base`.
- [ ] Configure `compose.yml` for local multi-service testing.
- [ ] Stream completions & live state via native Server-Sent Events (SSE).
- [ ] Enable native analytics & insights with embedded collection logging.
- [ ] Extend the cloud by mounting new `/v1/*` endpoints over ZAP transport.
- [ ] Build single static executable with `CGO_ENABLED=0`.
- [ ] Connect to `api.hanzo.ai` using Zen (`qwen3+`) models.
- [ ] Deploy with `git push` to `git.hanzo.ai`.
