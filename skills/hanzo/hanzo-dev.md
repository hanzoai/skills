# Hanzo Dev - Terminal AI Coding Agent

**Category**: Hanzo Ecosystem
**Skill Level**: Beginner to Advanced
**Prerequisites**: Command line familiarity, Git basics

## Overview

Hanzo Dev is an AI-powered terminal coding agent that brings the power of local AI directly to your development workflow. Unlike cloud-based coding assistants that require internet connectivity and send your code to external servers, Hanzo Dev runs entirely on your machine using Hanzo Node, ensuring your code never leaves your infrastructure.

**Available in**: Rust (native CLI), JavaScript/TypeScript (@hanzo/dev), and Python (hanzo-dev)

**Core Philosophy**: AI-assisted development with privacy, speed, and terminal-native workflows across all major programming languages.

## Key Features

### 🔒 Privacy-First Development
- **Local-only inference**: All code analysis and generation happens on your machine
- **No cloud dependencies**: Works offline after initial setup
- **Code stays local**: Never sent to external servers
- **Enterprise-safe**: Meets strict security requirements

### ⚡ Terminal-Native Experience
- **Fast autocomplete**: Sub-100ms suggestions via Hanzo Node
- **Streaming responses**: Real-time code generation
- **Shell integration**: Works with bash, zsh, fish
- **Git-aware**: Understands repository context

### 🎯 Intelligent Coding Assistance
- **Context-aware completions**: Understands your entire codebase
- **Multi-file edits**: Coordinates changes across files
- **Test generation**: Automatic test creation
- **Documentation**: Inline docs and README updates
- **Refactoring**: Safe, intelligent code transformations

### 🛠 Development Workflow Integration
- **Git operations**: Commit message generation, PR descriptions
- **Terminal commands**: Natural language → shell commands
- **Documentation**: Markdown, man pages, comments
- **CI/CD**: GitHub Actions, GitLab CI assistance

## Installation

### Prerequisites

```bash
# 1. Install Hanzo Node (for local inference)
cargo install hanzo-node
hanzo-node init

# 2. Download a code model
hanzo-node models pull codellama-7b
# or
hanzo-node models pull qwen-2-7b-instruct

# 3. Start Hanzo Node
hanzo-node start --gpu
```

### Install Hanzo Dev

**Language Support**: Hanzo Dev is available for **Rust**, **JavaScript/TypeScript**, and **Python** - ensuring high performance and easy integration regardless of your preferred language.

```bash
# Rust (Native CLI - highest performance)
cargo install hanzo-dev

# JavaScript/TypeScript (Node.js)
npm install -g @hanzo/dev

# Python
pip install hanzo-dev

# Or from binary (recommended for CLI usage)
curl -fsSL https://install.hanzo.ai/dev | sh

# From source (Rust)
git clone https://github.com/hanzoai/dev
cd dev
cargo install --path .

# Verify installation
hanzo-dev --version
```

**Language Equivalency**: All three language implementations (Rust, JS/TS, Python) provide the same CLI interface and functionality, making it easy to integrate Hanzo Dev into projects regardless of language choice. The Rust version offers maximum performance, while JS/TS and Python versions integrate seamlessly with their respective ecosystems.

### Shell Integration

```bash
# Bash
echo 'eval "$(hanzo-dev init bash)"' >> ~/.bashrc
source ~/.bashrc

# Zsh
echo 'eval "$(hanzo-dev init zsh)"' >> ~/.zshrc
source ~/.zshrc

# Fish
hanzo-dev init fish | source
echo 'hanzo-dev init fish | source' >> ~/.config/fish/config.fish
```

## Quick Start

### Basic Usage

```bash
# Start interactive session
hanzo-dev

# In the prompt:
> Write a Rust function to parse JSON
> Add error handling to parse_json
> Generate tests for parse_json
> Refactor to use serde_json
```

### Inline Mode (No Session)

```bash
# Generate code inline
hanzo-dev "Write a Python function to calculate fibonacci"

# Explain code
hanzo-dev "Explain this regex: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

# Fix errors
cargo build 2>&1 | hanzo-dev "Fix these compile errors"

# Generate commit message
git diff --staged | hanzo-dev "Write a commit message"
```

### Shell Integration Commands

```bash
# Natural language → shell command (Ctrl+G or Ctrl+Space)
# Type: "list all python files modified today"
# → Suggests: find . -name "*.py" -mtime -1

# AI-powered completion (Tab)
# Type: "git com"
# → Completes with: git commit -m "feat: add user authentication"

# Ask questions in comments (automatically triggered)
# Type: "# How do I read a CSV file in Python?"
# → Inserts: import csv; with open('file.csv') as f: reader = csv.reader(f)
```

## Core Features

### 1. Code Generation

```bash
# Function generation
> Write a Go function to connect to Redis with connection pooling

# Class generation  
> Create a Python class UserRepository with CRUD methods

# Module generation
> Implement a REST API router in Express.js with authentication

# Complete files
> Write a Dockerfile for a Node.js app with multi-stage builds
```

### 2. Code Explanation

```bash
# Explain selected code
> Explain this code: $(cat complex_algorithm.rs)

# Understand errors
> Explain this error: 
error[E0308]: mismatched types
  expected `Result<(), Error>`, found `()`

# Documentation lookup
> What does std::sync::Arc do in Rust?

# API documentation
> Show me examples of using asyncio.gather in Python
```

### 3. Code Modification

```bash
# Refactoring
> Refactor parse_user to use the builder pattern
> Extract this logic into a separate function
> Convert this callback to async/await

# Optimization
> Optimize this loop for performance
> Reduce memory allocations in this function

# Style improvements
> Make this code more idiomatic
> Apply consistent error handling
```

### 4. Test Generation

```bash
# Unit tests
> Generate unit tests for calculate_discount function

# Integration tests
> Write integration tests for the UserController

# Property-based tests
> Add property-based tests using quickcheck

# Test fixtures
> Create test fixtures for user data
```

### 5. Documentation

```bash
# Inline documentation
> Add documentation comments to all public functions

# README generation
> Generate a README for this project

# API documentation
> Create OpenAPI documentation for these endpoints

# Man pages
> Write a man page for this CLI tool
```

## Advanced Usage

### Multi-File Operations

```bash
# Coordinate changes across multiple files
> Refactor User model and update all controllers and tests

# Project-wide changes
> Rename the User class to Account throughout the project

# Dependency updates
> Update all imports after moving User to models/account

# Consistency checks
> Ensure all error handling follows the same pattern
```

### Repository-Aware Context

```bash
# Hanzo Dev automatically reads:
# - .git/ (repository structure)
# - README.md (project overview)
# - package.json / Cargo.toml / pyproject.toml (dependencies)
# - .hanzorc (custom context)

# Use project context
> Following the patterns in auth.rs, implement authorization

# Use codebase knowledge
> What's the pattern we use for database connections?

# Find examples
> Show me existing examples of using the Cache trait
```

### Custom Instructions

```bash
# Create .hanzorc in project root
cat > .hanzorc <<EOF
{
  "model": "codellama-13b",
  "temperature": 0.2,
  "max_tokens": 2000,
  
  "context": {
    "language": "Rust",
    "style": "functional",
    "error_handling": "Result types",
    "testing": "property-based with quickcheck"
  },
  
  "instructions": [
    "Always include error handling",
    "Prefer composition over inheritance",
    "Write property-based tests",
    "Use type-driven development"
  ],
  
  "ignore": [
    "target/",
    "node_modules/",
    "*.generated.*"
  ]
}
EOF

# Hanzo Dev now follows these guidelines automatically
> Write a function to parse config files
# → Generated with Result types, property tests, functional style
```

### Workflow Automation

```bash
# Create custom workflows
cat > .hanzo/workflows/feature.sh <<'EOF'
#!/bin/bash
# Feature development workflow

echo "Creating feature branch..."
git checkout -b "$1"

echo "Generating boilerplate..."
hanzo-dev "Generate REST API boilerplate for $1"

echo "Creating tests..."
hanzo-dev "Generate tests for $1 API"

echo "Updating docs..."
hanzo-dev "Update API documentation for $1"

echo "Feature $1 setup complete!"
EOF

chmod +x .hanzo/workflows/feature.sh

# Run workflow
hanzo-dev workflow feature user-profiles
```

## Integration with Hanzo Ecosystem

### With Hanzo Node (Local Inference)

```bash
# Hanzo Dev automatically uses Hanzo Node for inference
# Configuration in ~/.hanzo/dev/config.toml

[inference]
provider = "local"
node_url = "http://localhost:8080"
model = "codellama-13b"

[performance]
streaming = true
max_concurrent_requests = 4
cache_completions = true

[privacy]
telemetry = false
crash_reports = false
```

### With Hanzo MCP (Tool Access)

```bash
# Hanzo Dev can use MCP tools via Hanzo Node

# Configure MCP integration in .hanzorc
{
  "mcp": {
    "enabled": true,
    "servers": [
      "http://localhost:8081"  // Hanzo MCP server
    ],
    "auto_discover_tools": true
  }
}

# Now Hanzo Dev can use MCP tools
> Search the codebase for authentication patterns
# → Uses MCP grep/search tools automatically

> What's the database schema?
# → Uses MCP database inspection tools

> Run the test suite and explain failures
# → Uses MCP command execution tools
```

### With @hanzo/ui (Frontend Development)

```bash
# Generate UI components compatible with @hanzo/ui

> Create a user profile form using @hanzo/ui

# Generated:
import { Form, Input, Button } from '@hanzo/ui'
import { useForm } from '@hanzo/ui/hooks'

export function UserProfileForm() {
  const form = useForm({
    schema: userProfileSchema,
    onSubmit: handleSubmit
  })
  
  return (
    <Form form={form}>
      <Input name="name" label="Full Name" />
      <Input name="email" label="Email" type="email" />
      <Button type="submit">Save Profile</Button>
    </Form>
  )
}

> Add form validation with Zod
# → Adds Zod schema integrated with @hanzo/ui Form
```

### With Hanzo Python SDK (Backend Development)

```bash
# Generate backend code using Hanzo SDK patterns

> Create a FastAPI endpoint that uses Hanzo SDK for AI

# Generated:
from fastapi import FastAPI, WebSocket
from hanzo import Hanzo

app = FastAPI()
hanzo = Hanzo(inference_mode='local')

@app.post("/api/complete")
async def complete(prompt: str):
    response = hanzo.chat.completions.create(
        model='llama-3-8b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return {'completion': response.choices[0].message.content}

@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    # Streaming implementation...
```

## Git Integration

### Commit Message Generation

```bash
# Automatic commit message generation
git add .
hanzo-dev commit

# Or with git hook
cat > .git/hooks/prepare-commit-msg <<'EOF'
#!/bin/bash
if [ -z "$2" ]; then
  hanzo-dev "Generate commit message" > "$1"
fi
EOF
chmod +x .git/hooks/prepare-commit-msg

# Now commits automatically get AI-generated messages
git commit
# → Opens editor with: "feat: add user authentication with JWT tokens"
```

### Pull Request Descriptions

```bash
# Generate PR description
git diff main...HEAD | hanzo-dev "Write a pull request description"

# Or integrated with gh CLI
alias ghpr='gh pr create --title "$(hanzo-dev "Generate PR title")" --body "$(hanzo-dev "Generate PR description")"'

ghpr
```

### Code Review

```bash
# Review staged changes before commit
git diff --staged | hanzo-dev "Review this code for issues"

# Review entire PR
git diff main...HEAD | hanzo-dev "Review this PR"

# Check for common issues
git diff --staged | hanzo-dev "Check for security issues, performance problems, and style violations"
```

## Terminal Productivity

### Command Suggestions

```bash
# Natural language → shell commands (Ctrl+G)

# Type and press Ctrl+G:
"find all node_modules and delete them"
# → Suggests: find . -name "node_modules" -type d -prune -exec rm -rf {} +

"show git log for last week"
# → Suggests: git log --since="1 week ago"

"check disk usage of large files"
# → Suggests: du -h | sort -hr | head -n 20
```

### Error Explanation

```bash
# Pipe errors to Hanzo Dev for explanation

# Compilation errors
cargo build 2>&1 | hanzo-dev "Explain and fix"

# Runtime errors
node app.js 2>&1 | hanzo-dev "Debug this error"

# Test failures
npm test 2>&1 | hanzo-dev "Why are these tests failing?"
```

### Documentation Lookup

```bash
# Quick documentation access
hanzo-dev "How do I use tokio::select?"
hanzo-dev "Show example of React useEffect"
hanzo-dev "What's the difference between & and &mut in Rust?"

# API reference
hanzo-dev "Show all methods of the Array class in JavaScript"

# Best practices
hanzo-dev "What's the idiomatic way to handle errors in Go?"
```

## Configuration

### Global Configuration

```toml
# ~/.hanzo/dev/config.toml

[inference]
provider = "local"
node_url = "http://localhost:8080"
model = "codellama-13b"
fallback_model = "qwen-2-7b"

[performance]
streaming = true
max_concurrent_requests = 4
cache_completions = true
cache_ttl = 3600

[interface]
theme = "dracula"
syntax_highlighting = true
line_numbers = true
diff_view = "unified"

[keybindings]
complete = "Tab"
suggest_command = "Ctrl+G"
explain = "Ctrl+E"
refactor = "Ctrl+R"

[privacy]
telemetry = false
crash_reports = false
update_checks = true

[editor]
default = "vim"
diff_tool = "vimdiff"
```

### Project Configuration

```json
// .hanzorc (per-project overrides)
{
  "model": "codellama-13b",
  "temperature": 0.2,
  
  "context": {
    "language": "TypeScript",
    "framework": "Next.js",
    "style_guide": "Airbnb",
    "test_framework": "Jest"
  },
  
  "instructions": [
    "Use functional components only",
    "Prefer hooks over class components",
    "Include PropTypes or TypeScript types",
    "Write tests with React Testing Library"
  ],
  
  "patterns": {
    "component_structure": "src/components/{ComponentName}/index.tsx",
    "test_location": "__tests__/{ComponentName}.test.tsx",
    "style_location": "src/components/{ComponentName}/styles.ts"
  },
  
  "ignore": [
    "node_modules/",
    "build/",
    "dist/",
    "*.generated.*",
    ".next/"
  ]
}
```

## Best Practices

### 1. Use Project-Specific Configuration

```bash
# ✅ Good - consistent style across team
cat > .hanzorc <<EOF
{
  "context": {"language": "Rust", "style": "functional"},
  "instructions": ["Use Result types", "Add property tests"]
}
EOF

# ❌ Avoid - inconsistent suggestions
# (no project configuration)
```

### 2. Provide Context in Prompts

```bash
# ✅ Good - specific and contextual
> Following the error handling pattern in auth.rs, add error handling to payment.rs

# ❌ Avoid - vague
> Add error handling
```

### 3. Review AI-Generated Code

```bash
# ✅ Good - review before applying
hanzo-dev "Refactor this function" > /tmp/suggestion.txt
diff current_file.rs /tmp/suggestion.txt
# Review, then apply if good

# ❌ Avoid - blindly accepting
hanzo-dev "Refactor this function" > current_file.rs  # Don't do this!
```

### 4. Use Streaming for Long Generations

```bash
# ✅ Good - see progress
hanzo-dev --stream "Generate a REST API server"

# ❌ Avoid - wait for everything
hanzo-dev "Generate a REST API server"  # No --stream
```

### 5. Leverage Repository Context

```bash
# ✅ Good - uses project patterns
> Add a new API endpoint following existing patterns

# ❌ Avoid - ignores context
> Add a new API endpoint
```

## Comparison: Traditional vs Hanzo Dev

### Traditional Coding Workflow

```bash
# 1. Search Stack Overflow / docs
google "how to parse JSON in Rust"
# Copy example code
# Adapt to your use case (10-15 minutes)

# 2. Write boilerplate
# Manual typing (5-10 minutes)

# 3. Generate tests
# Write tests manually (10-20 minutes)

# 4. Write documentation
# Write docstrings manually (5-10 minutes)

# 5. Commit with message
git commit -m "update"  # Generic message

# Total time: 30-55 minutes
# Context switches: 5+
```

### Hanzo Dev Workflow

```bash
# 1. Generate with context
> Parse JSON following our error handling patterns
# → Complete, idiomatic code in 5 seconds

# 2. Generate tests
> Generate property-based tests for parse_json
# → Tests generated in 3 seconds

# 3. Generate docs
> Add documentation
# → Docs generated in 2 seconds

# 4. Commit with AI message
hanzo-dev commit
# → "feat: add robust JSON parsing with property-based tests"

# Total time: 1-2 minutes
# Context switches: 0
```

### Productivity Comparison

| Task | Traditional | Hanzo Dev | Speedup |
|------|------------|-----------|---------|
| Code generation | 10-15 min | 5-10 sec | 60-180x |
| Test generation | 10-20 min | 5-10 sec | 60-240x |
| Documentation | 5-10 min | 2-5 sec | 60-300x |
| Commit messages | 1-2 min | 2-3 sec | 20-60x |
| Error explanation | 5-15 min | 5-10 sec | 30-180x |
| Command lookup | 2-5 min | 2-3 sec | 40-150x |

**Overall**: 10-50x faster development with Hanzo Dev

## Troubleshooting

### Hanzo Dev Not Finding Hanzo Node

```bash
# Check if Hanzo Node is running
curl http://localhost:8080/health

# If not running:
hanzo-node start --gpu

# Configure node URL explicitly
cat > ~/.hanzo/dev/config.toml <<EOF
[inference]
provider = "local"
node_url = "http://localhost:8080"
EOF
```

### Slow Completions

```bash
# Enable streaming
hanzo-dev config set performance.streaming true

# Use smaller model
hanzo-dev config set inference.model codellama-7b

# Increase concurrent requests
hanzo-dev config set performance.max_concurrent_requests 8

# Enable completion cache
hanzo-dev config set performance.cache_completions true
```

### Poor Code Quality

```bash
# Lower temperature for more deterministic output
hanzo-dev config set inference.temperature 0.2

# Add project-specific instructions
cat > .hanzorc <<EOF
{
  "instructions": [
    "Follow existing code style",
    "Include comprehensive error handling",
    "Add type annotations",
    "Write tests for all functions"
  ]
}
EOF

# Use larger model
hanzo-node models pull codellama-34b
hanzo-dev config set inference.model codellama-34b
```

### Context Not Being Used

```bash
# Ensure .hanzorc exists and is valid JSON
cat .hanzorc | jq .

# Check ignored files
hanzo-dev context status

# Manually add context
hanzo-dev context add path/to/important/file.rs

# Verify context is being used
hanzo-dev --debug "Generate a function"
# Check logs for context loading
```

## Next Steps

1. **Read Hanzo Node Documentation** - Understanding local inference
2. **Explore Hanzo MCP** - Extending Hanzo Dev with tools
3. **Check Shell Integration** - Terminal productivity features
4. **See Git Integration** - Commit and PR workflows

## Related Skills

- **hanzo-node.md** - Local AI inference infrastructure  
- **hanzo-mcp.md** - Model Context Protocol for tools
- **python-sdk.md** - Python API for Hanzo
- **llm-model-selection.md** - Choosing coding models

---

**Remember**: Hanzo Dev brings AI directly to your terminal with privacy, speed, and context-awareness - dramatically faster than traditional coding workflows.
