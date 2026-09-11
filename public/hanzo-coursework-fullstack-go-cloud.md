# Coursework: Fullstack Cloud Engineering with Go & Hanzo Base

**Category**: Hanzo Ecosystem & Engineering Curriculum  
**Level**: Intermediate to Advanced  
**Prerequisites**: Familiarity with web services, basic understanding of concurrency.  
**Related Skills**: `hanzo/hanzo-migrate-backend-to-go.md`, `hanzo/hanzo-base.md`, `hanzo/hanzo-cloud.md`, `hanzo/hanzo-tutorial-realtime-sse-backend.md`, `hanzo/hanzo-examples-native-cloud.md`

---

## Curriculum Overview

This coursework trains engineering teams migrating from slow, interpreted, memory-heavy environments (Python, Next.js, Node.js, Ruby on Rails) to Hanzo's high-performance native stack (**Go, Rust, C++**).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               CURRICULUM: FULLSTACK CLOUD ENGINEERING WITH GO               │
│                                                                             │
│   [MODULE 1] Architectural Foundations & Runtime Shift                     │
│              - Why native compiled code wins: Go, Rust & C++                │
│              - Benchmarking: 15ms cold starts vs 6s Next.js SSR             │
│                                                                             │
│   [MODULE 2] Running Your Own Cloud Locally                                 │
│              - Complete local copy of the Hanzo AI Cloud                    │
│              - `hanzo up cloud` / `base serve` with compose.yml             │
│                                                                             │
│   [MODULE 3] Realtime SSE Streaming & State Synchronization                 │
│              - Native Server-Sent Events (SSE) in Go                        │
│              - Zero-overhead streaming without Pusher / external SaaS       │
│                                                                             │
│   [MODULE 4] Extending Hanzo Cloud with `/v1/*` Subsystems                  │
│              - Plugin host architecture & ZAP binary transport (HIP-0114)   │
│              - Wiring routes in `manifest/apps.go`                          │
│                                                                             │
│   [MODULE 5] Production Observability, Storage & Deployment                 │
│              - Embedded SQLite + ClickHouse datastore hybrid                │
│              - Zero-dependency static binary deployments (`CGO_ENABLED=0`)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: The Runtime Shift (Python/Next.js to Native Go)

### 1.1 The Cost of Interpreted Runtimes
Interpreted runtimes impose hidden taxes in enterprise environments:
- **Python**: The Global Interpreter Lock (GIL) serializes threads; async runtimes (`asyncio`) suffer from event-loop starvation and heavy memory fragmentation.
- **Node.js / Next.js**: Gigabytes of `node_modules`, cold start latencies (1.5s - 8s), high idle footprint (250MB+ per SSR container), and fragile build tooling.
- **Java / Spring**: High JVM memory overhead and slow warmup periods.

### 1.2 The Hanzo Triad
1. **Go (`github.com/hanzoai/base`, `github.com/hanzoai/cloud`)**:
   - Compiles to a single static binary (`CGO_ENABLED=0`).
   - Memory footprint: ~15MB idle.
   - Concurrency: Goroutines scheduled on `M:N` runtime with non-blocking network pollers.
2. **Rust (`hanzoai/engine`, `hanzoai/reth`)**:
   - Zero-cost abstractions, deterministic memory safety without GC pauses.
   - Inference execution and cryptographic settlement.
3. **C++ (Metal, CUDA, ANE)**:
   - Bare-metal hardware kernels for Apple Silicon and NVIDIA tensor cores.

### 1.3 Lab 1: Profiling Memory & Cold Starts
Compare a basic Next.js / FastAPI endpoint against a Go Hanzo Base endpoint:

```bash
# Build Go binary statically
CGO_ENABLED=0 go build -ldflags="-s -w" -o ./bin/server ./main.go

# Inspect size and launch time
ls -lh ./bin/server
time ./bin/server --healthcheck
```
*Expected Result*: Binary size < 25MB, startup time < 15 milliseconds.

---

## Module 2: Local-First Cloud (Run a Complete Copy)

Hanzo Cloud is engineered so every developer runs a complete copy locally.

### 2.1 The Philosophy
Never mock third-party cloud APIs. Run the actual Hanzo services in-process or via lightweight local containers.

### 2.2 Local Orchestration with `compose.yml`
Always use `compose.yml` (never `docker-compose.yml`):

```yaml
# compose.yml - Complete Hanzo Local Cloud
services:
  cloud:
    image: ghcr.io/hanzoai/cloud:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - HANZO_ENV=local
      - HANZO_PORT=8080
      - HANZO_STORAGE=sqlite
      - HANZO_SQLITE_PATH=/data/hanzo.db
      - HANZO_ZAP_PORT=9090
    volumes:
      - hanzo-data:/data
    restart: unless-stopped

  base:
    image: ghcr.io/hanzoai/base:latest
    ports:
      - "8090:8090"
    environment:
      - BASE_PORT=8090
      - BASE_DB=/data/base.db
      - BASE_REALTIME_SSE=true
      - BASE_CLOUD_URL=http://cloud:8080
    volumes:
      - base-data:/data

volumes:
  hanzo-data:
  base-data:
```

### 2.3 Lab 2: Launch & Healthcheck
```bash
# Start your local cloud
hanzo up cloud

# Verify control plane and base services
curl -i http://localhost:8080/v1/health
curl -i http://localhost:8090/v1/health
```

---

## Module 3: Realtime Native SSE (Server-Sent Events)

### 3.1 Why Native SSE Over WebSockets?
- Built directly on HTTP/1.1 and HTTP/2 (RFC 8895).
- Automatic browser reconnection via `Last-Event-ID`.
- Zero firewall or proxy websocket-upgrade friction.
- Efficient unidirectional event delivery for LLM tokens, notifications, and analytics.

### 3.2 Implementing the Hub in Go
A lock-free, concurrent SSE multiplexer:

```go
package realtime

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

type Event struct {
	ID    string
	Event string
	Data  []byte
}

type Hub struct {
	mu      sync.RWMutex
	clients map[chan Event]struct{}
}

func NewHub() *Hub {
	return &Hub{
		clients: make(map[chan Event]struct{}),
	}
}

func (h *Hub) Subscribe() chan Event {
	ch := make(chan Event, 64)
	h.mu.Lock()
	h.clients[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *Hub) Unsubscribe(ch chan Event) {
	h.mu.Lock()
	delete(h.clients, ch)
	close(ch)
	h.mu.Unlock()
}

func (h *Hub) Broadcast(e Event) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for ch := range h.clients {
		select {
		case ch <- e:
		default:
			// Client channel full; drop or handle slow consumer
		}
	}
}

func (h *Hub) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		ch := h.Subscribe()
		defer h.Unsubscribe(ch)

		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-r.Context().Done():
				return
			case <-ticker.C:
				fmt.Fprintf(w, ": heartbeat\n\n")
				flusher.Flush()
			case e := <-ch:
				if e.ID != "" {
					fmt.Fprintf(w, "id: %s\n", e.ID)
				}
				if e.Event != "" {
					fmt.Fprintf(w, "event: %s\n", e.Event)
				}
				fmt.Fprintf(w, "data: %s\n\n", e.Data)
				flusher.Flush()
			}
		}
	}
}
```

### 3.3 Lab 3: Verifying SSE Streams with cURL
```bash
# Listen to live stream
curl -N http://localhost:8090/v1/events/stream
```

---

## Module 4: Extending Hanzo Cloud with `/v1/*` Subsystems

Hanzo Cloud (`hanzoai/cloud`) uses a host + plugin registry architecture connected via **ZAP Transport** (HIP-0114).

### 4.1 Subsystem Interface
Every subsystem implements `cloud.Plugin`:

```go
package custom

import (
	"net/http"
	"github.com/hanzoai/cloud"
)

func init() {
	cloud.Register("analytics-custom", NewPlugin)
}

type Plugin struct{}

func NewPlugin(cfg *cloud.Config) (cloud.Plugin, error) {
	return &Plugin{}, nil
}

func (p *Plugin) Mount(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/custom/ingest", p.HandleIngest)
	mux.HandleFunc("GET /v1/custom/stats", p.HandleStats)
}

func (p *Plugin) HandleIngest(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusAccepted)
	w.Write([]byte(`{"status":"queued"}`))
}

func (p *Plugin) HandleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"active_workers":16,"throughput_per_sec":124000}`))
}
```

---

## Module 5: Storage & Observability

### 5.1 Hybrid Storage
- **Transactional State**: Embedded SQLite (`modernc.org/sqlite`) for single-binary zero-dependency nodes or PostgreSQL (`pgx/v5`).
- **Telemetry & Logs**: Columnar ClickHouse datastore.

### 5.2 Test-Driven Development (TDD)
Every module must ship with automated tests proving correctness:

```go
// hub_test.go
package realtime_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/hanzoai/base/realtime"
)

func TestSSEHub(t *testing.T) {
	hub := realtime.NewHub()
	server := httptest.NewServer(hub.Handler())
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", server.URL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Failed to connect to SSE stream: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Expected Content-Type text/event-stream, got %s", ct)
	}
}
```

---

## Conclusion & Certification

By completing this coursework, your engineering team possesses the patterns, tooling, and architectural foundation to:
1. Eliminate legacy interpreted runtimes in favor of compiled Go and Rust.
2. Build and run offline-ready local cloud environments.
3. Stream realtime data natively via SSE.
4. Scale seamlessly against Hanzo AI Cloud (`hanzo.ai` and `api.hanzo.ai`).
