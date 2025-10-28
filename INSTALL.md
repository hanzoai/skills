# Installation Guide

## Automatic Installation (Recommended)

Install the plugin with a single command in Claude Code:

```bash
/plugin install https://github.com/hanzoai/skills
```

Claude Code will automatically:
1. Clone the repository to `~/.claude/plugins/hanzo-skills/`
2. Register the `/skills` command
3. Make all 350+ atomic skills available

## Manual Installation

If automatic installation doesn't work, you can install manually:

```bash
# Create plugins directory if it doesn't exist
mkdir -p ~/.claude/plugins
mkdir -p ~/.claude/commands

# Clone or symlink the repository
ln -s /path/to/hanzo/skills ~/.claude/plugins/hanzo-skills

# Register the /skills command
ln -s ~/.claude/plugins/hanzo-skills/commands/skills.md ~/.claude/commands/skills.md
```

## Verification

After installation, verify it works:

```bash
# Check plugin is installed
ls -la ~/.claude/plugins/hanzo-skills

# Check command is registered
ls -la ~/.claude/commands/skills.md

# Try the command (in Claude Code)
/skills
```

## Troubleshooting

### Command Not Found

If `/skills` command doesn't work:

1. Check symlink: `ls -la ~/.claude/commands/skills.md`
2. Verify it points to: `~/.claude/plugins/hanzo-skills/commands/skills.md`
3. Restart Claude Code

### Skills Not Loading

If skills don't load:

1. Check plugin directory: `ls ~/.claude/plugins/hanzo-skills/skills/`
2. Verify it contains skill files
3. Try loading directly: `cat ~/.claude/plugins/hanzo-skills/skills/README.md`

### Permission Issues

If you get permission errors:

```bash
chmod -R u+r ~/.claude/plugins/hanzo-skills
```

## Uninstallation

To remove the plugin:

```bash
rm ~/.claude/plugins/hanzo-skills
rm ~/.claude/commands/skills.md
```

## Next Steps

After installation, see [QUICK_START.md](commands/skills/QUICK_START.md) for usage examples.
