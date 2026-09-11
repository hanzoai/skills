---
name: app-port
description: Port an existing application onto the Hanzo stack — Hanzo Base and Hanzo Cloud OSS for the backend, ZAP for transport, @hanzo/ui and @hanzo/gui for the interface — and move its Python or JavaScript runtime into Go, Rust or C++. Use when adopting a Next.js, Node, Django, Flask, FastAPI, Rails or Spring app; when a service still speaks WebSockets; when a deployment still needs an interpreter beside the binary; or when asked to modernise, migrate or absorb an app somebody else's stack currently runs.
---

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

### Measure it on the app in front of you


Two claims worth making to whoever is paying for the work, and one way to check
each on the app in front of you rather than taking it from here.

**Fewer moving parts at runtime.** The before is an interpreter, the
application on top of it, a process supervisor, and usually a second container
for the other language. The after is one binary. Measure it: resident memory at
idle and under the same load, and the count of processes and images the
deployment needs. Take both readings on the same machine with the same traffic,
because a number from somebody else's benchmark is a number about their
hardware.

**Concurrency stops costing a process.** A Python application serves
concurrent requests by forking workers, and every worker is another copy of the
interpreter and its heap; a Node application gets one loop and adds processes to
use more cores. Goroutines are neither. The reading that shows this is memory
against concurrency: hold the request rate and raise the concurrency, and watch
whether the footprint follows.

State these as what they are. The mechanism is certain — one process is fewer
processes — and the size of the win is a property of the app, so measure it
rather than quoting a multiplier.

### The part that matters most

A developer can run the whole thing on the computer they already own.

That is not a performance note. When the development environment is one binary
and a local cloud, working on the project does not require renting somebody
else's computer, a cluster to be provisioned, a seat on a plan, or a network
connection good enough to stream a workspace. Someone on a modest laptop, on a
bad connection, in a country where the card required to sign up is hard to get,
can do the same work as someone on a funded team.

Stacks that can only be run in a data centre quietly decide who gets to
contribute. This one does not, and keeping it that way is a design constraint
rather than a side effect — if a change means the app can no longer be run
locally in full, that is a cost to weigh and usually a reason to choose
differently.

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

## The order, and why it is this order


Each step leaves the app working. Do not start the next until the current one
is deployed.

**Stand it up unchanged, inside our binary.** JavaScript through goja, Python
through gpython, both in-process. Nothing is rewritten. What this buys is the
deployment: one Go binary, no Node beside it, no Python beside it, no second
base image. It is also the step that proves the surface — every route the old
runtime served is now served by ours, and a diff of responses says whether that
is true.

**Move the transport to ZAP.** Same messages, same handlers, new wire. A
service still speaking WebSockets at this point is a service that will still be
speaking them a year from now.

**Port the interface to `@hanzo/ui` on `@hanzo/gui`.** This is a visual port and
it has its own skill — use `ui-port-parity`, which treats the existing
interface as an executable visual contract and converges screen by screen with
measurement rather than by eye. Do not attempt it by inspection.

**Port the handlers to Go, one at a time.** The embedded runtime is the
fallback for everything not yet ported, so a handler moves when it is ready and
the rest keep running. A route is done when its Go implementation answers
byte-identically to the embedded one, and that is a test, not a judgement.

**Delete the embedded runtime.** When no route falls through to it. Not before,
and not "temporarily kept" — an interpreter nobody needs is an interpreter
nobody maintains and a dependency that still ships.

## ZAP, not WebSockets

A service that still opens a WebSocket has not been ported, however much of its
code is now Go. WebSockets are the thing being replaced, not a transport to keep
beside ZAP: two transports means two reconnect policies, two auth paths and two
sets of framing bugs, and the one carrying production traffic ends up being
whichever the last change touched.

Port the transport as its own step, with the message shapes unchanged, so a
failure is attributable to the transport rather than to everything at once.

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

## What not to do


**Do not run the old and new implementations side by side in production and
compare.** It doubles the failure surface and the comparison is never clean,
because the two see different requests. Compare against captured traffic
instead.

**Do not port tests last.** The embedded runtime is the oracle: a Go handler is
correct when it answers the way the interpreted one did, and that is only
checkable while the interpreted one still exists.

**Do not keep a second component library "for the parts we haven't got to".**
Two libraries is two type ramps, two spacing scales and two sets of focus
styles, and the screens between them read as a seam.

**Do not preserve an API shape nobody uses** just because it was there. Check
the callers first; a port is the cheapest moment to drop a surface, and the
most expensive moment to carry one forward.

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

## Telling whether it worked


- One binary, no interpreter in the image, no runtime beside it.
- No WebSocket endpoints.
- Every component from `@hanzo/ui`; the ramp from `@hanzo/design`.
- The appearance knobs move the interface, checked by reading a component's
  computed colour after setting one — a panel that does nothing is a common
  and invisible failure.
- Handlers are goroutines.
- The interface converges under `ui-port-parity`'s gates, not under review.

## The interface half

This skill covers the backend, the transport and the runtime. Porting the
interface is its own problem with its own failure modes — an agent comparing
two screens by eye will declare a match a person rejects at a glance — and it
has its own skill: **`ui-port`**. Use it rather than reimplementing the
measurement.
