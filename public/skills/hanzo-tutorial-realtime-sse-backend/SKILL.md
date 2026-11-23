# Tutorial: Building a Production Realtime SSE Backend in Go with Hanzo Base

**Category**: Hanzo Ecosystem & Tutorials  
**Level**: Practical Hands-On  
**Time to Complete**: 20 minutes  
**Related Skills**: `hanzo/hanzo-coursework-fullstack-go-cloud.md`, `hanzo/hanzo-base.md`, `hanzo/hanzo-migrate-backend-to-go.md`

---

## Overview

In this tutorial, you will build a production-grade, high-concurrency realtime backend in **Go** using **Hanzo Base** (`github.com/hanzoai/base`). 

Instead of heavy WebSockets or third-party paid push services (like Pusher or Ably), you will implement native **Server-Sent Events (SSE)**. This pattern handles thousands of concurrent subscribers per node with single-digit megabytes of RAM, built-in reconnection protocols, and seamless integration with the Hanzo AI Cloud.

---

## Step 1: Initialize Your Go Module

Create a clean directory and initialize your Go project using `uv` or standard Go toolchain:

```bash
mkdir -p realtime-backend && cd realtime-backend
go mod init example.com/realtime-backend
```

Install the Hanzo Base dependency:

```bash
go get github.com/hanzoai/base@latest
```

---

## Step 2: Implement the Event Streamer

Create `main.go`:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// MetricPayload represents live analytics telemetry
type MetricPayload struct {
	Timestamp int64   `json:"timestamp"`
	Model     string  `json:"model"`
	LatencyMs float64 `json:"latency_ms"`
	Tokens    int     `json:"tokens"`
}

type SSEStreamer struct {
	mu      sync.RWMutex
	clients map[chan []byte]struct{}
}

func NewSSEStreamer() *SSEStreamer {
	return &SSEStreamer{
		clients: make(map[chan []byte]struct{}),
	}
}

func (s *SSEStreamer) Register() chan []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	ch := make(chan []byte, 128)
	s.clients[ch] = struct{}{}
	return ch
}

func (s *SSEStreamer) Unregister(ch chan []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.clients, ch)
	close(ch)
}

func (s *SSEStreamer) Broadcast(msg []byte) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for ch := range s.clients {
		select {
		case ch <- msg:
		default:
			// Non-blocking drop for slow consumer to avoid head-of-line blocking
		}
	}
}

func (s *SSEStreamer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported by client/proxy", http.StatusBadRequest)
		return
	}

	// Set required SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("X-Accel-Buffering", "no") // Disable NGINX/Envoy proxy buffering

	ch := s.Register()
	defer s.Unregister(ch)

	// Send initial connection handshake
	fmt.Fprintf(w, "event: ready\ndata: {\"connected\":true}\n\n")
	flusher.Flush()

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": heartbeat\n\n")
			flusher.Flush()
		case data, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func main() {
	streamer := NewSSEStreamer()

	// Background producer simulating Hanzo AI Inference stream
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for t := range ticker.C {
			payload := MetricPayload{
				Timestamp: t.Unix(),
				Model:     "zen-3.5-pro",
				LatencyMs: 14.2,
				Tokens:    840,
			}
			bytes, _ := json.Marshal(payload)
			streamer.Broadcast(bytes)
		}
	}()

	mux := http.NewServeMux()
	mux.Handle("/v1/events/stream", streamer)
	mux.HandleFunc("/v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok","runtime":"go-native"}`))
	})

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 0, // Infinite for SSE streaming
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Println("⚡ Hanzo SSE Backend listening on :8080")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
}
```

---

## Step 3: TDD Test Suite (`stream_test.go`)

Ensure strict Test-Driven Development (TDD) by verifying the HTTP endpoint and SSE event delivery:

```go
package main

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSSEStreamingEndpoint(t *testing.T) {
	streamer := NewSSEStreamer()
	ts := httptest.NewServer(streamer)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", ts.URL, nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Failed to execute request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Expected text/event-stream, got %s", ct)
	}

	reader := bufio.NewReader(resp.Body)

	// Verify handshake event
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("Error reading initial line: %v", err)
	}
	if !strings.HasPrefix(line, "event: ready") {
		t.Errorf("Expected ready event, got %s", line)
	}

	// Broadcast test payload
	testMsg := []byte(`{"test":"event-data"}`)
	go func() {
		time.Sleep(100 * time.Millisecond)
		streamer.Broadcast(testMsg)
	}()

	// Read until we find the test data
	found := false
	for i := 0; i < 10; i++ {
		l, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		if strings.Contains(l, "event-data") {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Did not receive broadcast message in stream")
	}
}
```

Run the tests:

```bash
go test -v ./...
```

---

## Step 4: Frontend Consumption (Browser & Node)

In your client application or agent harness, subscribe using native browser `EventSource`:

```typescript
const evtSource = new EventSource("http://localhost:8080/v1/events/stream");

evtSource.addEventListener("ready", (e) => {
  console.log("Stream connected:", JSON.parse(e.data));
});

evtSource.onmessage = (e) => {
  const metric = JSON.parse(e.data);
  console.log("Realtime telemetry:", metric);
};

evtSource.onerror = (err) => {
  console.error("Connection interrupted, auto-reconnecting...", err);
};
```

---

## Step 5: Production Deployment

Build a zero-dependency static Go binary:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o ./bin/server .
```

Containerize using `compose.yml`:

```yaml
# compose.yml
services:
  sse-backend:
    build: .
    ports:
      - "8080:8080"
    restart: always
```
