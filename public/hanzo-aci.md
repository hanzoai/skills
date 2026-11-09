---
name: hanzo-aci
description: "Agent-Computer Interface for software development agents. AST analysis, code editing, linting, and refactoring with tree-sitter across Python, JS, TS, Ruby, and Go."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-aci",
              "label": "Install Hanzo ACI (pip)",
            },
          ],
      },
  }
---

# Hanzo ACI — Agent-Computer Interface

`pip install hanzo-aci`

Code editing, AST analysis, and linting tools for software development agents. Tree-sitter powered with multi-language support.

## Quick Start

```python
from hanzo_aci import OHEditor

editor = OHEditor(workspace_root="/path/to/project")

# View file contents
content = editor.view("/path/to/file.py")

# Edit with string replacement
editor.str_replace("/path/to/file.py",
    old_str="def old_name(",
    new_str="def new_name("
)

# Insert text at line
editor.insert("/path/to/file.py", line=10, text="    # New comment\n")

# Undo last edit
editor.undo_edit("/path/to/file.py")
```

## Editor Commands

| Command       | Description                          |
| ------------- | ------------------------------------ |
| `view`        | View file contents with line numbers |
| `create`      | Create a new file                    |
| `str_replace` | Replace exact string in file         |
| `insert`      | Insert text at line number           |
| `undo_edit`   | Undo last edit to a file             |

## Linting

```python
from hanzo_aci import DefaultLinter

linter = DefaultLinter()

# Lint Python (flake8-based)
errors = linter.lint("/path/to/file.py")

# Lint TypeScript (tree-sitter-based)
errors = linter.lint("/path/to/file.ts")

# Lint Ruby (tree-sitter-based)
errors = linter.lint("/path/to/file.rb")
```

## Supported Languages

- **Python** — flake8-based linting
- **TypeScript/JavaScript** — tree-sitter AST analysis
- **Ruby** — tree-sitter AST analysis
- **Go** — tree-sitter AST analysis

## Features

- **File History**: Full undo/redo with `FileHistoryManager`
- **Encoding Detection**: Automatic multi-encoding support
- **Binary Detection**: Safe handling of binary files
- **Max File Size**: Configurable (default 10MB)
- **Diff Generation**: Unified diff output
- **Snippet Context**: Smart context windowing for large files

## Semantic Search

```python
# grep-ast integration for semantic code search
from hanzo_aci import grep_ast

results = grep_ast("function_name", path="/path/to/project")
```
