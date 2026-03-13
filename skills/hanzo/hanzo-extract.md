# Hanzo Extract - Content Extraction & Sanitization

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-aci.md`, `hanzo/hanzo-guard.md`

## Overview

Hanzo Extract is a **Rust crate for extracting clean text** from web pages, PDFs, and Claude Code conversation logs. Designed to produce LLM-ready content with optional PII sanitization via `hanzo-guard`. Ships two CLI binaries (`extract-web`, `extract-conversations`) and a library API with async trait-based extractors.

### Why Hanzo Extract?

- **Web extraction**: Fetches pages, strips scripts/nav/footer, extracts main content area
- **PDF extraction**: Text extraction from PDF documents via `lopdf`
- **Conversation export**: Turns Claude Code JSONL session logs into training datasets with quality scoring and train/val/test splits
- **Sanitization**: PII redaction pipeline via hanzo-guard (feature-gated)
- **Feature flags**: Compile only what you need (`web`, `pdf`, `conversations`, `sanitize`)

### Tech Stack

- **Language**: Rust (edition 2021)
- **Crate**: `hanzo-extract` v0.1.0
- **Async runtime**: Tokio
- **Web**: reqwest + scraper (HTML parsing/CSS selectors)
- **PDF**: lopdf
- **Conversations**: walkdir, sha2, chrono, rand (anonymization + splits)
- **Error handling**: thiserror
- **Serialization**: serde + serde_json
- **CI**: GitHub Actions (ci.yml, pages.yml for docs)

Repo: `github.com/hanzoai/extract`

## When to use

- Extracting clean text from web pages for LLM context/RAG
- Extracting text from PDF documents
- Exporting Claude Code conversations for fine-tuning datasets
- Building content pipelines that need PII redaction

## Quick reference

| Item | Value |
|------|-------|
| Crate | `hanzo-extract` |
| Repo | `github.com/hanzoai/extract` |
| Version | 0.1.0 |
| License | MIT OR Apache-2.0 |
| Docs | `docs.rs/hanzo-extract` |
| Default features | `web`, `pdf` |

## Installation

```bash
cargo add hanzo-extract
```

```toml
# Cargo.toml - pick what you need
[dependencies]
hanzo-extract = "0.1"                                         # web + pdf (default)
hanzo-extract = { version = "0.1", features = ["full"] }      # everything
hanzo-extract = { version = "0.1", features = ["conversations"] }  # dataset export
```

## Usage

### Web Extraction

```rust
use hanzo_extract::{WebExtractor, ExtractorConfig, Extractor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extractor = WebExtractor::new(ExtractorConfig::default());
    let result = extractor.extract("https://example.com").await?;

    println!("Title: {:?}", result.title);
    println!("Text: {}", result.text);
    println!("Words: {}", result.word_count);
    Ok(())
}
```

### PDF Extraction

```rust
use hanzo_extract::{PdfExtractor, Extractor};

let extractor = PdfExtractor::default();
let result = extractor.extract("document.pdf").await?;
println!("{}", result.text);
```

### Conversation Export (training datasets)

```rust
use hanzo_extract::conversations::{ConversationExporter, ExporterConfig};
use std::path::Path;

let mut exporter = ConversationExporter::new();
exporter.export(
    Path::new("~/.claude/projects"),
    Path::new("./training-data"),
)?;
```

Output structure:
```
./training-data/
  conversations_20260313.jsonl   # Full conversation data
  training_20260313.jsonl        # Instruction/response pairs
  splits/
    train_20260313.jsonl         # 80%
    val_20260313.jsonl           # 10%
    test_20260313.jsonl          # 10%
```

### CLI Binaries

```bash
# Web extraction
cargo install hanzo-extract --features web
extract-web https://example.com
extract-web https://example.com --json

# Conversation export
cargo install hanzo-extract --features conversations
extract-conversations --source ~/.claude/projects --output ./conversations
```

## Configuration

```rust
use hanzo_extract::ExtractorConfig;

let config = ExtractorConfig::default()
    .with_max_length(200_000)       // Max chars (default: 100,000)
    .with_timeout(60)               // Request timeout secs (default: 30)
    .with_clean_text(true);         // Strip HTML/scripts (default: true)

// With sanitization (requires `sanitize` feature)
let config = config
    .with_sanitize(true)
    .with_redact_pii(true)
    .with_detect_injection(true);
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `web` | Yes | Web page extraction (reqwest + scraper) |
| `pdf` | Yes | PDF text extraction (lopdf) |
| `conversations` | No | Claude Code session export |
| `sanitize` | No | PII redaction via hanzo-guard |
| `full` | No | All features |

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root, Extractor trait, re-exports |
| `src/web.rs` | WebExtractor with HTML parsing and content area detection |
| `src/pdf.rs` | PdfExtractor for PDF text extraction |
| `src/conversations.rs` | ConversationExporter with quality scoring and splits |
| `src/config.rs` | ExtractorConfig with builder pattern |
| `src/sanitize.rs` | Sanitization pipeline (hanzo-guard integration) |
| `src/error.rs` | ExtractError enum (thiserror) |
| `src/result.rs` | ExtractResult struct |

## Quality Scoring (Conversations)

Conversations are scored 0.0-1.0 based on:
- Thinking/reasoning presence (+0.2)
- Tool usage (+0.15)
- Agentic tools like Task/dispatch (+0.1)
- Opus/Sonnet model (+0.1/+0.05)
- Response length (+0.1)

## Related Skills

- `hanzo/hanzo-aci.md` - Agent Computer Interface (uses extraction for document conversion)
- `hanzo/hanzo-guard.md` - LLM I/O sanitization (powers the `sanitize` feature)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: extraction, pdf, web, sanitization, llm, training-data, rust
**Prerequisites**: Rust toolchain, Tokio runtime
