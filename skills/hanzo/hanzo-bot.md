# Hanzo Bot - Automation & Skills Framework

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-agent.md`, `hanzo/hanzo-mcp.md`

## Overview

Hanzo Bot is the **automation framework** powering 739 public skills on `github.com/hanzoskill`. Each skill is a self-contained automation recipe with scripts, metadata, and documentation. The skills monorepo at `hanzobot/skills` contains the full collection with history.

### Why Hanzo Bot?

- **739 skills**: Pre-built automations for common tasks
- **Modular**: Each skill is independent, composable
- **Multi-language**: Skills in Bash, Python, Node.js, Go
- **Discoverable**: Auto-matched by keyword, intent, or file pattern
- **Extensible**: Create custom skills with standard template

### History

Rebranded from ClawdBot. All references updated:
- `clawdbot`/`ClawdBot` → `hanzo-bot`/`Hanzo Bot`
- Config: `~/.bot/`, `bot.json`
- Env vars: `BOT_*` (not `CLAWDBOT_*`)

## When to use

- Running pre-built automation recipes
- Creating reusable automation skills
- Integrating automations into CI/CD pipelines
- Building domain-specific bot behaviors

## Quick reference

| Item | Value |
|------|-------|
| Monorepo | `github.com/hanzobot/skills` |
| Public repos | `github.com/hanzoskill` (739 repos, one per skill) |
| Skills dir | `skills/skills/<author>/<skill>/` |
| Config | `~/.bot/config.json` |
| Env prefix | `BOT_*` |
| Repo | `github.com/hanzoai/bot` |

## Skill Structure

```
skills/<author>/<skill>/
├── SKILL.md          # Skill documentation and instructions
├── _meta.json        # Metadata (name, description, triggers, version)
├── scripts/          # Executable scripts
│   ├── run.sh        # Main entry point
│   ├── setup.sh      # One-time setup
│   └── *.py/js/go    # Language-specific scripts
├── .author           # Author info
└── tests/            # Optional tests
```

### _meta.json Format

```json
{
  "name": "my-skill",
  "description": "Short description of what this skill does",
  "version": "1.0.0",
  "author": "hanzo",
  "triggers": {
    "keywords": ["deploy", "release"],
    "file_patterns": ["*.yaml", "Dockerfile"],
    "intent": "deployment automation"
  },
  "dependencies": ["docker", "kubectl"],
  "language": "bash"
}
```

### SKILL.md Format

```markdown
# Skill Name

## Description
What this skill does and when to use it.

## Usage
How to invoke and configure.

## Examples
Concrete usage examples.

## Requirements
Dependencies and prerequisites.
```

## Skill Categories

| Category | Count | Examples |
|----------|-------|---------|
| DevOps | ~200 | Docker, K8s, CI/CD, monitoring |
| Development | ~150 | Code generation, linting, testing |
| AI/ML | ~100 | Model serving, training, inference |
| Data | ~80 | ETL, databases, migrations |
| Security | ~60 | Scanning, auditing, secrets |
| Cloud | ~50 | AWS, GCP, DO provisioning |
| Misc | ~100 | Formatting, documentation, bots |

## Creating a Skill

```bash
# Use template
cp -r skills/_SKILL_TEMPLATE skills/skills/myorg/my-skill/

# Edit metadata
cat > skills/skills/myorg/my-skill/_meta.json << 'EOF'
{
  "name": "my-skill",
  "description": "Automates widget deployment",
  "version": "1.0.0",
  "author": "myorg",
  "triggers": {"keywords": ["widget", "deploy"]},
  "language": "bash"
}
EOF

# Write the script
cat > skills/skills/myorg/my-skill/scripts/run.sh << 'EOF'
#!/bin/bash
set -euo pipefail
echo "Deploying widget..."
# Your automation here
EOF
chmod +x skills/skills/myorg/my-skill/scripts/run.sh

# Write documentation
cat > skills/skills/myorg/my-skill/SKILL.md << 'EOF'
# Widget Deploy
Automates widget deployment to production.
## Usage
Run `hanzo-bot run my-skill`
EOF
```

## CLI Usage

```bash
# List available skills
hanzo-bot skills list

# Search skills
hanzo-bot skills search "kubernetes deploy"

# Run a skill
hanzo-bot run <skill-name>

# Run with arguments
hanzo-bot run <skill-name> --env production --dry-run

# Install a skill from hanzoskill
hanzo-bot skills install <skill-name>

# Show skill info
hanzo-bot skills info <skill-name>
```

## GitHub Organization

The `hanzoskill` org on GitHub has 739 public repos — one per skill. These are published from the monorepo via CI.

**Rate limits**: GitHub secondary rate limit allows ~150 repo creations per burst, ~30-40min cooldown between bursts.

**Duplicate handling**: If a skill name exists in multiple authors, the second is suffixed (e.g., `discord-doctor-alt`).

## Related Skills

- `hanzo/hanzo-agent.md` - AI agent framework (bots can use agents)
- `hanzo/hanzo-mcp.md` - MCP tools (skills can expose MCP tools)
- `hanzo/hanzo-dev.md` - AI coding agent (uses skill discovery)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: automation, skills, bot, recipes
**Prerequisites**: Bash/Python basics
