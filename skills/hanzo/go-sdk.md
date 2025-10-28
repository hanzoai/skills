# Hanzo Go SDK

**Category**: Hanzo Ecosystem
**Skill Level**: Intermediate to Advanced
**Prerequisites**: Go 1.18+, basic understanding of LLM APIs
**Related Skills**: python-sdk.md, hanzo-engine.md, hanzo-node.md

## Overview

The Hanzo Go SDK provides a type-safe, idiomatic Go interface to Hanzo AI infrastructure with automatic routing, privacy-first local inference, and seamless cloud fallback. Perfect for backend services, concurrent workloads, and blockchain node applications.

**Core Philosophy**: High-performance AI integration with Go's simplicity and concurrency primitives.

## Key Features

### 🚀 Go-Idiomatic Design
- **Type-safe API**: Leverages Go's type system for compile-time safety
- **Context-aware**: First-class `context.Context` support for cancellation and timeouts
- **Goroutine-safe**: Concurrent request handling with minimal overhead
- **Error handling**: Explicit error returns following Go conventions

### 🔒 Privacy-First Architecture
- **Local-first inference**: Automatically uses Hanzo Node when available
- **Selective cloud routing**: Only routes to cloud when necessary
- **Data sovereignty**: Keep sensitive data on your infrastructure

### ⚡ High Performance
- **Connection pooling**: Reuses HTTP connections efficiently
- **Streaming support**: Real-time response streaming
- **Batch operations**: Process multiple requests concurrently
- **Low latency**: Optimized for sub-millisecond overhead

### 🎯 Automatic Model Routing
- **Complexity analysis**: Routes based on query complexity
- **Cost optimization**: Balances quality vs cost automatically
- **Latency optimization**: Uses fastest available model

## Installation

```bash
# Latest version
go get github.com/hanzoai/go-sdk

# Specific version
go get github.com/hanzoai/go-sdk@v0.1.0-alpha.3
```

**Requirements**: Go 1.18+

## API Key & Free Credits

### Sign Up for Hanzo API

Get started with Hanzo API and receive **free credits** or **subscription discounts**:

1. **Create Account**: Visit [https://hanzo.ai/signup](https://hanzo.ai/signup)
2. **Get API Key**: Navigate to dashboard → API Keys → Create New Key
3. **Free Credits**: New signups receive $10 in free credits (no credit card required)
4. **Subscription Discounts**: Annual plans get 20% off, students/educators get 50% off

### API Key Setup

```go
import (
    "os"
    "github.com/hanzoai/go-sdk"
    "github.com/hanzoai/go-sdk/option"
)

// From environment variable (recommended)
client := hanzoai.NewClient(
    option.WithAPIKey(os.Getenv("HANZO_API_KEY")),
)

// Or directly
client := hanzoai.NewClient(
    option.WithAPIKey("hanz_..."),
)
```

**Note**: For local-only inference (Hanzo Node), no API key needed. API key is only required for cloud routing or when using cloud models.

## SDK Equivalency

The Hanzo Go SDK is **feature-equivalent** with:
- **Python SDK** (`hanzo`) - For ML/AI workflows
- **Rust SDK** (`hanzo-rs`) - For maximum performance
- **JavaScript/TypeScript SDK** (`@hanzo/sdk`) - For web applications

All SDKs share the same API surface. Choose Go for:
- Backend services requiring high concurrency
- Blockchain node applications
- Microservices architectures
- Systems requiring goroutine-based parallelism

**Coming Soon**: Golang Node implementation (blockchain node rewritten in Go from Rust).

## Quick Start

### Basic Usage (Local Inference)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/hanzoai/go-sdk"
    "github.com/hanzoai/go-sdk/option"
)

func main() {
    client := hanzoai.NewClient(
        option.WithInferenceMode("local"),
        option.WithNodeURL("http://localhost:8080"),
    )

    response, err := client.Chat.Completions.Create(
        context.Background(),
        hanzoai.ChatCompletionCreateParams{
            Model: hanzoai.F("llama-3-8b"),
            Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                hanzoai.SystemMessage("You are a helpful assistant."),
                hanzoai.UserMessage("Explain quantum computing in simple terms."),
            }),
            Temperature: hanzoai.F(0.7),
            MaxTokens:   hanzoai.Int(500),
        },
    )

    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

### Hybrid Mode (Local + Cloud)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/hanzoai/go-sdk"
    "github.com/hanzoai/go-sdk/option"
)

func main() {
    client := hanzoai.NewClient(
        option.WithInferenceMode("hybrid"),
        option.WithAutoRoute(true),
        option.WithCostOptimize(true),
        option.WithAPIKey("hanz_..."), // For cloud fallback
    )

    // SDK automatically routes based on complexity
    response, err := client.Chat.Completions.Create(
        context.Background(),
        hanzoai.ChatCompletionCreateParams{
            Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                hanzoai.UserMessage("Write a sorting algorithm in Go"),
            }),
        },
    )

    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

## Advanced Features

### Streaming Responses

```go
stream := client.Chat.Completions.CreateStreaming(
    context.Background(),
    hanzoai.ChatCompletionCreateParams{
        Model: hanzoai.F("llama-3-8b"),
        Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
            hanzoai.UserMessage("Write a story"),
        }),
        Stream: hanzoai.Bool(true),
    },
)

for stream.Next() {
    chunk := stream.Current()
    fmt.Print(chunk.Choices[0].Delta.Content)
}

if err := stream.Err(); err != nil {
    log.Fatal(err)
}
```

### Concurrent Requests with Goroutines

```go
func processBatch(client *hanzoai.Client, prompts []string) {
    ctx := context.Background()
    results := make(chan string, len(prompts))
    errors := make(chan error, len(prompts))

    // Process all prompts concurrently
    for _, prompt := range prompts {
        go func(p string) {
            response, err := client.Chat.Completions.Create(ctx,
                hanzoai.ChatCompletionCreateParams{
                    Model: hanzoai.F("llama-3-8b"),
                    Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                        hanzoai.UserMessage(p),
                    }),
                },
            )

            if err != nil {
                errors <- err
                return
            }

            results <- response.Choices[0].Message.Content
        }(prompt)
    }

    // Collect results
    for i := 0; i < len(prompts); i++ {
        select {
        case result := <-results:
            fmt.Println("Result:", result)
        case err := <-errors:
            log.Printf("Error: %v", err)
        }
    }
}
```

### Context and Timeouts

```go
// With timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := client.Chat.Completions.Create(ctx,
    hanzoai.ChatCompletionCreateParams{
        Model: hanzoai.F("llama-3-8b"),
        Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
            hanzoai.UserMessage("Quick query"),
        }),
    },
)

if err != nil {
    if ctx.Err() == context.DeadlineExceeded {
        log.Println("Request timed out")
    } else {
        log.Fatal(err)
    }
}
```

### Error Handling

```go
response, err := client.Chat.Completions.Create(ctx, params)
if err != nil {
    var apiErr *hanzoai.Error
    if errors.As(err, &apiErr) {
        switch apiErr.StatusCode {
        case 401:
            log.Println("Authentication failed - check API key")
        case 429:
            log.Println("Rate limited - retry after backoff")
        case 500:
            log.Println("Server error - fallback to local model")
        default:
            log.Printf("API error: %s", apiErr.Message)
        }
    }
    return err
}
```

## Integration Patterns

### HTTP Handler Middleware

```go
type AIMiddleware struct {
    client *hanzoai.Client
}

func (m *AIMiddleware) Handle(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract query from request
        query := r.URL.Query().Get("q")

        // Process with Hanzo
        response, err := m.client.Chat.Completions.Create(
            r.Context(),
            hanzoai.ChatCompletionCreateParams{
                Model: hanzoai.F("llama-3-8b"),
                Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                    hanzoai.UserMessage(query),
                }),
            },
        )

        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        // Add AI response to context
        ctx := context.WithValue(r.Context(), "ai_response", response)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

### gRPC Server Integration

```go
type AIService struct {
    client *hanzoai.Client
    pb.UnimplementedAIServiceServer
}

func (s *AIService) ProcessQuery(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
    response, err := s.client.Chat.Completions.Create(ctx,
        hanzoai.ChatCompletionCreateParams{
            Model: hanzoai.F(req.Model),
            Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                hanzoai.UserMessage(req.Query),
            }),
        },
    )

    if err != nil {
        return nil, status.Errorf(codes.Internal, "AI processing failed: %v", err)
    }

    return &pb.QueryResponse{
        Answer: response.Choices[0].Message.Content,
        Model:  response.Model,
        Tokens: int32(response.Usage.TotalTokens),
    }, nil
}
```

### Blockchain Node Integration

```go
// For blockchain nodes (current Rust implementation, Go version coming soon)
type BlockchainAINode struct {
    client   *hanzoai.Client
    nodeAddr string
}

func (n *BlockchainAINode) ProcessAITransaction(ctx context.Context, tx *Transaction) error {
    // Extract AI query from transaction
    query := string(tx.Data)

    // Process with local Hanzo Node
    response, err := n.client.Chat.Completions.Create(ctx,
        hanzoai.ChatCompletionCreateParams{
            Model: hanzoai.F("llama-3-8b"),
            Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                hanzoai.UserMessage(query),
            }),
        },
    )

    if err != nil {
        return fmt.Errorf("AI processing failed: %w", err)
    }

    // Store result in blockchain state
    return n.StoreAIResult(tx.ID, response.Choices[0].Message.Content)
}
```

## Performance Optimization

### Connection Pooling

```go
// Reuse client for connection pooling
var client = hanzoai.NewClient(
    option.WithHTTPClient(&http.Client{
        Transport: &http.Transport{
            MaxIdleConns:        100,
            MaxIdleConnsPerHost: 100,
            IdleConnTimeout:     90 * time.Second,
        },
    }),
)

func handler(w http.ResponseWriter, r *http.Request) {
    // Reuses connections automatically
    response, err := client.Chat.Completions.Create(r.Context(), params)
    // ...
}
```

### Batch Processing with Worker Pools

```go
type Worker struct {
    client *hanzoai.Client
    jobs   <-chan string
    results chan<- string
}

func (w *Worker) Start() {
    for prompt := range w.jobs {
        response, err := w.client.Chat.Completions.Create(
            context.Background(),
            hanzoai.ChatCompletionCreateParams{
                Model: hanzoai.F("llama-3-8b"),
                Messages: hanzoai.F([]hanzoai.ChatCompletionMessageParam{
                    hanzoai.UserMessage(prompt),
                }),
            },
        )

        if err != nil {
            w.results <- fmt.Sprintf("Error: %v", err)
            continue
        }

        w.results <- response.Choices[0].Message.Content
    }
}

func ProcessWithWorkers(prompts []string, numWorkers int) []string {
    jobs := make(chan string, len(prompts))
    results := make(chan string, len(prompts))

    client := hanzoai.NewClient(option.WithInferenceMode("local"))

    // Start workers
    for i := 0; i < numWorkers; i++ {
        worker := &Worker{client: client, jobs: jobs, results: results}
        go worker.Start()
    }

    // Send jobs
    for _, prompt := range prompts {
        jobs <- prompt
    }
    close(jobs)

    // Collect results
    var output []string
    for i := 0; i < len(prompts); i++ {
        output = append(output, <-results)
    }

    return output
}
```

## Cloud Infrastructure

**Note**: Hanzo's cloud infrastructure is fully managed and optimized for performance and cost. When using cloud routing, the SDK automatically handles:
- Load balancing across regions
- Model availability and failover
- Rate limiting and quotas
- SSL/TLS encryption
- Request queuing and prioritization

You don't need to worry about cloud infrastructure details - just use the SDK and it handles all routing, failover, and optimization automatically.

## Best Practices

### 1. Use Context Everywhere

```go
// Good - respects cancellation and timeouts
func processWithTimeout(client *hanzoai.Client, query string) error {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    _, err := client.Chat.Completions.Create(ctx, params)
    return err
}

// Bad - no context control
func processNoContext(client *hanzoai.Client, query string) error {
    _, err := client.Chat.Completions.Create(context.Background(), params)
    return err
}
```

### 2. Reuse Clients

```go
// Good - one client per application
var globalClient = hanzoai.NewClient(options...)

func handler1(w http.ResponseWriter, r *http.Request) {
    globalClient.Chat.Completions.Create(r.Context(), params)
}

// Bad - creating client per request
func handler2(w http.ResponseWriter, r *http.Request) {
    client := hanzoai.NewClient(options...) // Wastes connections!
    client.Chat.Completions.Create(r.Context(), params)
}
```

### 3. Handle Errors Explicitly

```go
// Good - handles specific error types
response, err := client.Chat.Completions.Create(ctx, params)
if err != nil {
    var apiErr *hanzoai.Error
    if errors.As(err, &apiErr) {
        // Handle API-specific errors
        return handleAPIError(apiErr)
    }
    // Handle other errors
    return err
}

// Bad - ignores error details
response, _ := client.Chat.Completions.Create(ctx, params)
```

## Troubleshooting

### Connection Refused

```go
// Check Hanzo Node is running
// hanzo-node start --gpu

client := hanzoai.NewClient(
    option.WithInferenceMode("local"),
    option.WithNodeURL("http://localhost:8080"),
)
```

### Rate Limiting

```go
// Implement exponential backoff
func retryWithBackoff(ctx context.Context, fn func() error) error {
    backoff := time.Second
    for i := 0; i < 5; i++ {
        err := fn()
        if err == nil {
            return nil
        }

        var apiErr *hanzoai.Error
        if errors.As(err, &apiErr) && apiErr.StatusCode == 429 {
            time.Sleep(backoff)
            backoff *= 2
            continue
        }

        return err
    }
    return fmt.Errorf("max retries exceeded")
}
```

## Related Skills

- **python-sdk.md** - Python SDK for ML/AI workflows
- **hanzo-engine.md** - Native Rust inference & embedding engine
- **hanzo-node.md** - Local AI inference infrastructure
- **hanzo-dev.md** - Terminal AI coding agent (available in Go)

## Additional Resources

- **GitHub**: https://github.com/hanzoai/go-sdk
- **API Documentation**: [api.md](https://github.com/hanzoai/go-sdk/blob/main/api.md)
- **Hanzo AI**: https://hanzo.ai
- **Go Package**: https://pkg.go.dev/github.com/hanzoai/go-sdk

---

**Remember**: Use the Hanzo Go SDK for type-safe, high-performance AI integration with idiomatic Go patterns - perfect for concurrent workloads and backend services.
