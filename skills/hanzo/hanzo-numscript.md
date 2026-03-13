# Hanzo Numscript - DSL for Financial Transactions

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-reconciliation.md`, `hanzo/hanzo-billing.md`

## Overview

Numscript is a **domain-specific language for modeling complex financial transactions**. It provides declarative syntax for multi-party fund transfers with percentage splits, ordered funding sources, overdraft controls, and parameterized scripts. Used by Hanzo Ledger for programmable money movement. Includes a CLI, Go library, LSP server, and MCP server.

### Why Numscript?

- **Declarative transfers** -- Express multi-party fund routing in readable syntax
- **Percentage splits** -- Route funds by percentage or fixed amount to multiple destinations
- **Source ordering** -- Define fallback funding sources with automatic overdraft prevention
- **Type safety** -- Compile-time validation of account references and asset types
- **ANTLR grammar** -- Formal grammar (Numscript.g4 + Lexer.g4) for parsing
- **LSP support** -- Language Server Protocol for editor integration
- **MCP server** -- Model Context Protocol integration for AI tooling

### Tech Stack

- **Language**: Go 1.24
- **Parser**: ANTLR4 (Go target)
- **CLI**: cobra (goreleaser for cross-platform builds)
- **Testing**: testify + go-snaps (snapshot testing)
- **Error tracking**: Sentry
- **LSP**: go.lsp.dev/protocol
- **MCP**: mark3labs/mcp-go
- **License**: MIT

### OSS Base

Repo: `hanzoai/numscript` (Formance numscript fork). Go module path is `github.com/formancehq/numscript` (upstream module name retained).

## When to use

- Modeling multi-party financial transfers (e.g., marketplace payouts)
- Splitting payments by percentage across multiple destinations
- Defining ordered funding sources with overdraft limits
- Parameterizing transaction scripts for reuse
- Validating transaction scripts before execution (parse + analysis)
- Integrating financial transaction DSL into a ledger system

## Hard requirements

1. **Go 1.24+** for building from source
2. **ANTLR4** for regenerating the parser from grammar files
3. **Hanzo Ledger** for executing scripts against real accounts (optional for parsing)

## Quick reference

| Item | Value |
|------|-------|
| Go module | `github.com/formancehq/numscript` |
| CLI binary | `numscript` |
| Go version | 1.24 |
| Grammar | `Numscript.g4` + `Lexer.g4` (ANTLR4) |
| Default branch | `main` |
| License | MIT |
| Repo | `github.com/hanzoai/numscript` |

## One-file quickstart

### Install CLI

```bash
go install github.com/hanzoai/numscript/cmd/numscript@latest
```

### Parse and validate

```bash
numscript check script.num
numscript fmt script.num
```

### Go library

```go
package main

import (
    "context"
    "fmt"
    "math/big"

    "github.com/formancehq/numscript"
)

func main() {
    program := numscript.Parse(`
        send [USD/2 5000] (
            source = @users:alice
            destination = {
                85% to @merchants:shop
                10% to @platform:fees
                5%  to @platform:reserve
            }
        )
    `)

    if errs := program.GetParsingErrors(); len(errs) > 0 {
        fmt.Println("Parse errors:", numscript.ParseErrorsToString(errs))
        return
    }

    // Get needed variables (for parameterized scripts)
    vars := program.GetNeededVariables()
    fmt.Println("Variables:", vars)

    // Execute against a store
    result, err := program.Run(context.Background(),
        numscript.VariablesMap{},
        &numscript.StaticStore{
            Balances: numscript.Balances{
                "users:alice": {"USD/2": big.NewInt(10000)},
            },
        },
    )
    if err != nil {
        fmt.Println("Error:", err)
        return
    }

    for _, posting := range result.Postings {
        fmt.Printf("%s -> %s: %s %s\n",
            posting.Source, posting.Destination,
            posting.Amount, posting.Asset)
    }
}
```

## Core Concepts

### Language Syntax

```numscript
// Variables with types
vars {
    monetary $amount
    account  $sender
    account  $receiver
}

// Simple transfer
send $amount (
    source = $sender
    destination = $receiver
)

// Multi-source with fallback (ordered, first with funds wins)
send [USD/2 10000] (
    source = {
        @users:alice
        @users:alice:savings
        @world                    // infinite source (minting)
    }
    destination = @merchants:shop
)

// Percentage split destination
send [USD/2 10000] (
    source = @users:alice
    destination = {
        85%       to @merchants:shop
        10%       to @platform:fees
        remaining to @platform:reserve
    }
)

// Overdraft controls
send [USD/2 100] (
    source = {
        @a allowing overdraft up to [USD/2 10]
        @b allowing unbounded overdraft
    }
    destination = @dest
)

// Save (earmark funds)
save [USD/2 10] from @alice

// Metadata-driven accounts
vars {
    account $dest = meta(@config, "payout_account")
}

// Function calls
set_tx_meta("key", "value")
```

### Architecture

```
                Parse                  Run
numscript.go ──────────► ParseResult ──────► ExecutionResult
                  |                      |
                  |   ANTLR4 parser      |   interpreter
                  |   (Lexer.g4 +        |   (balance queries,
                  |    Numscript.g4)      |    metadata queries,
                  |                      |    posting generation)
                  |                      |
                  v                      v
             ParserError[]          Posting[]
                                    Metadata{}
                                    AccountsMetadata{}
```

### Public API

```go
// Parse source code
numscript.Parse(code string) ParseResult

// Inspect parsed result
ParseResult.GetParsingErrors() []ParserError
ParseResult.GetNeededVariables() map[string]string
ParseResult.GetSource() string

// Execute
ParseResult.Run(ctx, vars, store) (ExecutionResult, InterpreterError)
ParseResult.RunWithFeatureFlags(ctx, vars, store, flags) (ExecutionResult, InterpreterError)

// Types
type Posting struct {
    Source, Destination string
    Amount              *big.Int
    Asset               string
}

type ExecutionResult struct {
    Postings         []Posting
    Metadata         Metadata           // set_tx_meta() results
    AccountsMetadata AccountsMetadata   // set_account_meta() results
}

// Store interface (implement for your ledger)
type Store interface {
    GetBalances(ctx, BalanceQuery) (Balances, error)
    GetAccountsMetadata(ctx, MetadataQuery) (AccountsMetadata, error)
}
```

### Feature Flags

```go
flags.ExperimentalOneofFeatureFlag       // oneof {} source selection
flags.ExperimentalMidScriptFunctionCall  // balance() calls mid-script
```

## Directory structure

```
github.com/hanzoai/numscript/
    numscript.go                    # Public API: Parse, Run, types
    numscript_test.go               # Integration tests (20+ test cases)
    Numscript.g4                    # ANTLR4 grammar (parser rules)
    Lexer.g4                        # ANTLR4 grammar (lexer rules)
    Justfile                        # Build commands (generate, test, lint, release)
    .goreleaser.yaml                # Cross-platform binary releases
    inputs.schema.json              # JSON schema for script inputs
    specs.schema.json               # JSON schema for specs
    cmd/numscript/
        main.go                     # CLI entrypoint (check, fmt subcommands)
    internal/
        parser/                     # ANTLR4-generated parser + parse tree
        interpreter/                # Script execution engine
        analysis/                   # Static analysis and validation
        lsp/                        # Language Server Protocol implementation
        mcp_impl/                   # Model Context Protocol server
        jsonrpc2/                   # JSON-RPC 2.0 transport
        cmd/                        # CLI command implementations
        flags/                      # Feature flag definitions
        ansi/                       # Terminal color output
        specs_format/               # Spec formatting utilities
        utils/                      # Shared utilities
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Parse error on `@world` | Missing source block | `@world` must be inside a `source = {}` block |
| MissingFundsErr | Source accounts have insufficient balance | Add `@world` as final fallback or use `allowing overdraft` |
| Variables not resolved | Missing from VariablesMap | Check `GetNeededVariables()` and provide all required vars |
| ANTLR regeneration fails | ANTLR4 not installed | Install via `brew install antlr` or use Nix flake |
| Module path mismatch | Upstream module name | Import as `github.com/formancehq/numscript` (retained from fork) |

## Related Skills

- `hanzo/hanzo-reconciliation.md` - Transaction reconciliation engine
- `hanzo/hanzo-billing.md` - Billing and subscription management

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: dsl, fintech, ledger, transactions, payments, go
**Prerequisites**: Go 1.24+, ANTLR4 (for grammar regeneration)
