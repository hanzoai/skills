# Hanzo Skill - AI Agent Skill Installer CLI

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-bot.md`, `hanzo/hanzo-agent.md`

## Overview

Hanzo Skill (`@hanzo/skill`) is a **TypeScript CLI tool** for installing, managing, and distributing AI agent skills. It clones skill repositories to `~/.hanzo/skills/` as the canonical source of truth, then symlinks them into every supported agent directory (Claude Code, Cursor, Codex, Openclaw, Hanzo Bot). Zero dependencies beyond Node.js standard library.

### Why Hanzo Skill?

- **Single source of truth**: `~/.hanzo/skills/` holds all skill repos; agents see symlinks
- **Multi-agent sync**: One `add` command makes skills available to Claude, Cursor, Codex, Openclaw, and Hanzo Bot simultaneously
- **Git-native**: Skills are Git repos -- `add` clones, re-run pulls latest
- **Zero config**: No auth, no server, just `npx @hanzo/skill add org/repo`
- **Tiny**: 3 source files, zero runtime dependencies

### Tech Stack

- **Language**: TypeScript (ESM)
- **Runtime**: Node.js 18+
- **Package**: `@hanzo/skill` v1.0.1 on npm
- **Dependencies**: None (uses `node:child_process`, `node:fs`, `node:os`, `node:path`)
- **Tests**: Node.js built-in test runner (`node --test`)

## When to use

- Installing shared skill libraries for AI coding agents
- Keeping skills in sync across Claude Code, Cursor, Codex, and Hanzo Bot
- Distributing organizational skill packs (e.g., `hanzoai/skills`, `bootnode/skills`)
- Listing or removing installed skill collections

## Quick reference

| Item | Value |
|------|-------|
| npm package | `@hanzo/skill` v1.0.1 |
| Binary names | `hanzo-skill`, `skill` |
| Canonical dir | `~/.hanzo/skills/` |
| Repo | `github.com/hanzoai/skill` |
| Branch | `main` |
| License | MIT |

## Agent Directories (Symlink Targets)

| Directory | Agent |
|-----------|-------|
| `~/.claude/skills/` | Claude Code |
| `~/.agents/skills/` | Codex, Openclaw |
| `~/.cursor/skills/` | Cursor |
| `~/.hanzo/bot/skills/` | Hanzo Bot |

## Usage

```bash
# Install skills from a GitHub repo
npx @hanzo/skill add hanzoai/skills       # Hanzo ecosystem skills
npx @hanzo/skill add bootnode/skills      # Bootnode blockchain APIs
npx @hanzo/skill add luxfi/skills         # Lux Cloud APIs

# Force reinstall (delete and re-clone)
npx @hanzo/skill add hanzoai/skills --force

# List installed skills
npx @hanzo/skill list

# Remove skills and all symlinks
npx @hanzo/skill remove hanzoai-skills

# Show help
npx @hanzo/skill --help
```

## How It Works

1. **Normalize URL**: Accepts `org/repo`, `github.com/org/repo`, or full HTTPS URL
2. **Clone or pull**: `git clone --depth 1` to `~/.hanzo/skills/<org>-<repo>/`, or `git pull --ff-only` if exists
3. **Count skills**: Walks directory tree counting `SKILL.md` files and `.md` skill files
4. **Symlink**: Creates symlinks from `~/.hanzo/skills/<name>` into each agent skill directory
5. **Remove**: Deletes symlinks from all agent directories, then removes the skill directory

## Repository Structure

```
src/
  cli.ts          # CLI entry point (add, remove, list, help)
  index.ts        # Core library (addSkills, removeSkills, listSkills, symlinkToAgents)
  index.test.ts   # Tests (Node.js built-in test runner)
package.json      # @hanzo/skill v1.0.1, type: module, bin: hanzo-skill/skill
tsconfig.json     # TypeScript config (ESM output)
```

## Library API

The package also exports functions for programmatic use:

```typescript
import { addSkills, removeSkills, listSkills, HANZO_SKILLS_DIR, AGENT_SKILL_DIRS } from "@hanzo/skill"

// Install skills
const result = await addSkills("hanzoai/skills", { force: false })
// result.dirName = "hanzoai-skills"
// result.targetDir = "~/.hanzo/skills/hanzoai-skills"
// result.count = 87 (number of skill files found)
// result.linked = ["~/.claude/skills", ...]

// List installed
const skills = await listSkills()
// [{ name: "hanzoai-skills", path: "...", count: 87 }]

// Remove
await removeSkills("hanzoai-skills")
```

### Exported Functions

| Function | Description |
|----------|-------------|
| `addSkills(url, opts?)` | Clone/pull repo, symlink to agents, return stats |
| `removeSkills(name)` | Remove skill dir and all agent symlinks |
| `listSkills()` | List installed skill directories with counts |
| `symlinkToAgents(name)` | Create symlinks from canonical to all agent dirs |
| `unlinkFromAgents(name)` | Remove symlinks from all agent dirs |
| `normalizeGitUrl(input)` | Convert shorthand to full HTTPS clone URL |
| `extractDirName(input)` | Extract `org-repo` directory name from URL |
| `countSkills(dir)` | Count SKILL.md and .md skill files recursively |

## Build and Test

```bash
# Install and build
npm install
npm run build

# Run tests
npm test

# Development (watch mode)
npm run dev
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `git clone` fails | Private repo or no git | Check git auth, install git |
| Symlink permission error | Restricted home directory | Check permissions on `~/.<agent>/` |
| Skills not visible in agent | Agent not looking at skill dir | Verify agent reads from its skill directory |
| Skill count is 0 | No SKILL.md or .md files | Repo may use different structure |
| `--force` needed | Existing clone is dirty | Use `--force` to delete and re-clone |

## Related Skills

- `hanzo/hanzo-bot.md` - Hanzo Bot (consumes skills from ~/.hanzo/bot/skills/)
- `hanzo/hanzo-agent.md` - Multi-agent SDK
- `hanzo/hanzo-dev.md` - Dev CLI (different tool -- dev is AI agent, skill is package manager)

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: cli, skills, agents, npm, typescript, symlinks
**Prerequisites**: Node.js 18+, git
