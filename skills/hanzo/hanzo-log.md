# Hanzo Log - Structured Logging for Go

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-cli.md`, `hanzo/hanzo-orm.md`

## Overview

Hanzo Log is a **high-performance, zero-allocation structured logging library** for Go. Based on zerolog with a dual API: a zero-allocation chaining style and a geth-compatible variadic style. Both share the same underlying Event system backed by sync.Pool. The standard logging library for all Hanzo Go services.

### Why Hanzo Log?

- **Zero allocation**: ~8 ns/op for empty log, 0 B/op via sync.Pool event recycling
- **Dual API**: Method chaining (zerolog-style) and variadic key-value (geth-style)
- **Logger interface**: All logging goes through a `Logger` interface, enabling Noop/test/custom loggers
- **slog integration**: Maps to `log/slog` levels for stdlib interop
- **Colorized console**: Pretty `ConsoleWriter` with ANSI colors for development
- **Sampling**: `BasicSampler` to throttle high-volume logs in production
- **Hooks**: Attach arbitrary processing to log events
- **Lumberjack rotation**: Built-in log file rotation via `lumberjack.v2`

### Tech Stack

- **Language**: Go 1.26+
- **Module**: `github.com/hanzoai/log`
- **Dependencies**: `go-colorable`, `go-isatty`, `lumberjack.v2` (all minimal)
- **OSS Base**: zerolog architecture with geth API layer added
- **CI**: GitHub Actions (ci.yml, release.yml, docs.yml)

Repo: `github.com/hanzoai/log`

## When to use

- Any Hanzo Go service that needs structured logging
- Replacing `go-ethereum/log` in Lux/EVM codebases (geth API is compatible)
- High-throughput services where allocation matters
- When you need both JSON and human-readable log output

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/log` |
| Repo | `github.com/hanzoai/log` |
| Go version | 1.26+ |
| Install | `go get github.com/hanzoai/log` |
| License | See LICENSE |
| Sub-packages | `github.com/hanzoai/log/level` |

## Installation

```bash
go get github.com/hanzoai/log
```

## Usage

### Chaining API (zero-allocation)

```go
import "github.com/hanzoai/log"

// Simple
log.Info().Msg("hello world")

// With fields
log.Info().
    Str("user", "alice").
    Int("attempt", 3).
    Msg("login successful")

// With timestamp
l := log.NewWriter(os.Stdout).With().Timestamp().Logger()
l.Info().Str("service", "api").Msg("started")
```

### Variadic API (geth-compatible)

```go
import "github.com/hanzoai/log"

// Simple
logger := log.New("component", "myapp")
logger.Info("hello world")

// With fields
logger.Info("login successful",
    log.String("user", "alice"),
    log.Int("attempt", 3),
)

// Error with key
logger.Error("operation failed", log.Err(err))
```

### Logger Interface

All logging goes through the `Logger` interface:

```go
type Logger interface {
    Trace(msg string, ctx ...interface{})
    Debug(msg string, ctx ...interface{})
    Info(msg string, ctx ...interface{})
    Warn(msg string, ctx ...interface{})
    Error(msg string, ctx ...interface{})
    Fatal(msg string, ctx ...interface{})
    Panic(msg string, ctx ...interface{})
    With() Context
    New(ctx ...interface{}) Logger
    Level(lvl Level) Logger
    // ... plus chaining methods (TraceEvent, DebugEvent, etc.)
}

// Disabled logger for optional params
logger := log.Noop()
```

### Console Output

```go
output := log.ConsoleWriter{Out: os.Stdout}
l := log.NewWriter(output)
l.Info().Str("foo", "bar").Msg("Hello World")
// Output: 3:04pm INF Hello World foo=bar
```

### Log Levels

```go
log.SetGlobalLevel(log.WarnLevel)

// Levels: TraceLevel(-1), DebugLevel(0), InfoLevel(1),
//         WarnLevel(2), ErrorLevel(3), FatalLevel(4), PanicLevel(5)
```

### EVM/VM Initialization

```go
logger, err := log.InitLogger("C-chain", "info", false, writer)
```

## Key Files

| File | Purpose |
|------|---------|
| `log.go` | Logger interface, constructors, core implementation |
| `event.go` | Event type with sync.Pool recycling |
| `geth.go` | Geth-compatible Field constructors (String, Int, Err, etc.) |
| `fields.go` | Internal field encoding for variadic API |
| `globals.go` | Global config (field names, time formats, colors) |
| `console.go` | ConsoleWriter for human-readable output |
| `context.go` | Context builder for pre-set fields |
| `slog.go` | log/slog handler integration |
| `sampler.go` | Log sampling (BasicSampler) |
| `encoder_json.go` | JSON encoder |
| `hook.go` | Hook interface |

## Performance

```
BenchmarkLogEmpty-10       161M ops     8.4 ns/op    0 B/op   0 allocs/op
BenchmarkLogFields-10       32M ops    40.2 ns/op    0 B/op   0 allocs/op
BenchmarkLogFieldType/Str   95M ops    11.7 ns/op    0 B/op   0 allocs/op
```

Compared to: zerolog (19 ns), zap (236 ns), logrus (1244 ns, 27 allocs).

## Field Constructors (geth-style)

```go
log.String(key, val)    log.Int(key, val)       log.Bool(key, val)
log.Float64(key, val)   log.Duration(key, val)  log.Time(key, val)
log.Err(err)            log.Any(key, val)       log.Binary(key, val)
log.Strings(key, vals)  log.Ints(key, vals)     log.Uint64(key, val)
```

## Related Skills

- `hanzo/hanzo-orm.md` - ORM that uses hanzo/log
- `hanzo/hanzo-cli.md` - CLI tools
- `hanzo/go-sdk.md` - Go SDK

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: logging, go, structured-logging, zerolog, geth, performance
**Prerequisites**: Go 1.26+
