# Migration Guide: Manual Installation → Plugin

This guide is for users who previously installed hanzo-skills manually using `install.sh` and are migrating to the new plugin system.

## Why Migrate?

The plugin system provides:
- ✅ **One-command installation** - No scripts, no manual syncing
- ✅ **Automatic updates** - Pull latest version with `/plugin update`
- ✅ **Cleaner installation** - No files in `~/.claude/skills` or `~/.claude/commands`
- ✅ **Better isolation** - Plugin data stays in `~/.claude/plugins/hanzo-skills/`
- ✅ **Version management** - Track which version you're using
- ✅ **Marketplace distribution** - Discover and share plugins easily

## Migration Steps

### 1. Backup Your Data (Optional)

If you've customized any skills or commands, back them up:

```bash
# Backup skills
cp -r ~/.claude/skills ~/.claude/skills.backup

# Backup commands
cp -r ~/.claude/commands ~/.claude/commands.backup
```

### 2. Uninstall Manual Installation

Remove the manually installed files:

```bash
# Remove skills directory
rm -rf ~/.claude/skills

# Remove /skills command
rm ~/.claude/commands/skills.md
```

**Note:** If you have other custom commands in `~/.claude/commands/`, only remove `skills.md`:

```bash
ls ~/.claude/commands/  # Check what's there
rm ~/.claude/commands/skills.md  # Remove only skills.md
```

### 3. Install Plugin

Install hanzo-skills as a plugin:

```bash
/plugin install https://github.com/hanzoai/skills
```

That's it! The plugin system will:
- Download all 292 skills to `~/.claude/plugins/hanzo-skills~/.hanzo/skills/`
- Register the `/skills` command automatically
- Make everything available immediately

### 4. Verify Installation

Check that the plugin is installed:

```bash
/plugin list
```

You should see:
```
hanzo-skills (v2.0.0) - 292 atomic skills with gateway-based progressive loading
```

Test the `/skills` command:

```bash
/skills
```

You should see skill recommendations and categories.

### 5. Restore Customizations (If Any)

If you customized any skills, you can copy them to the plugin directory:

```bash
# Example: Restore a custom skill
cp ~/.claude/skills.backup/custom/my-skill.md \
   ~/.claude/plugins/hanzo-skills~/.hanzo/skills/custom/my-skill.md
```

**Note:** The plugin directory is at `~/.claude/plugins/hanzo-skills/`, not `~/.claude~/.hanzo/skills/`.

## What Changed

### Installation Location

**Before (Manual):**
```
~/.claude~/.hanzo/skills/          # Skills directory
~/.claude/commands/        # Commands directory
```

**After (Plugin):**
```
~/.claude/plugins/hanzo-skills~/.hanzo/skills/      # Skills
~/.claude/plugins/hanzo-skills/commands/    # Commands
```

### Installation Method

**Before (Manual):**
```bash
git clone https://github.com/hanzoai/skills
cd hanzo-skills
./commands/install.sh
```

**After (Plugin):**
```bash
/plugin install https://github.com/hanzoai/skills
```

### Updating

**Before (Manual):**
```bash
cd ~/src/hanzo-skills
git pull
./commands/install.sh  # Re-run install
```

**After (Plugin):**
```bash
/plugin update hanzo-skills
```

### Uninstallation

**Before (Manual):**
```bash
./commands/uninstall.sh
```

**After (Plugin):**
```bash
/plugin uninstall hanzo-skills
```

## What Stayed the Same

### Skills Structure

The 292 skills across 31 categories are identical:
- Same gateway architecture
- Same progressive loading
- Same skill content and guidance
- Same category organization

### Skills Discovery

All discovery mechanisms work exactly the same:
- `/skills` command (context-aware recommendations)
- `/skills frontend` (browse by category)
- `/skills postgres` (search by keyword)
- Automatic gateway activation
- Manual skill reading with `cat`

### Compatibility

All existing workflows continue to work:
- Gateway skills still activate based on project context
- Skills still compose with each other
- All 28 gateways, 31 categories, 292 skills unchanged

## Troubleshooting

### `/skills` command not found after migration

**Solution:**
```bash
# Verify plugin is installed
/plugin list

# If not listed, install it
/plugin install https://github.com/hanzoai/skills

# Restart Claude Code session
```

### Skills not showing up

**Solution:**
```bash
# Check plugin directory exists
ls ~/.claude/plugins/hanzo-skills~/.hanzo/skills/

# If empty, reinstall plugin
/plugin uninstall hanzo-skills
/plugin install https://github.com/hanzoai/skills
```

### Want to keep both installations?

You can keep the manual installation in `~/.claude~/.hanzo/skills/` for reference, but:
- Only the plugin version will be used by `/skills` command
- Gateway skills will activate from plugin location
- This may cause confusion - recommended to remove manual installation

## Rollback (If Needed)

If you need to rollback to manual installation:

```bash
# 1. Uninstall plugin
/plugin uninstall hanzo-skills

# 2. Restore backup
cp -r ~/.claude/skills.backup ~/.claude/skills
cp ~/.claude/commands.backup/skills.md ~/.claude/commands/skills.md

# 3. Restart Claude Code
```

## Questions?

### How do I know which version I have?

**Plugin:**
```bash
/plugin list  # Shows version number
```

**Manual:**
```bash
cat ~/.claude~/.hanzo/skills/README.md | grep "Version:"
```

### Can I install both?

Technically yes, but not recommended. The plugin system will take precedence for commands and skill discovery.

### Will my custom skills be lost?

No, but you need to manually copy them from `~/.claude~/.hanzo/skills/` to `~/.claude/plugins/hanzo-skills~/.hanzo/skills/` after installing the plugin.

### How do I contribute updates?

Same as before:
1. Fork the repository: https://github.com/hanzoai/skills
2. Make changes to skills
3. Submit a pull request

Once merged, users can update with `/plugin update hanzo-skills`.

## Next Steps

After migration:
1. Explore the plugin system: `/plugin list`, `/plugin help`
2. Test your workflows with the plugin-based installation
3. Remove backups if everything works: `rm -rf ~/.claude/skills.backup ~/.claude/commands.backup`
4. Update any documentation or personal notes about installation

## Support

If you encounter issues during migration:
1. Check this guide's troubleshooting section
2. Review plugin documentation: `cat ~/.claude/plugins/hanzo-skills/PLUGIN.md`
3. Open an issue: https://github.com/hanzoai~/.hanzo/skills/issues

---

**Migration complete!** You're now using hanzo-skills as a Claude Code plugin with cleaner installation, automatic updates, and better version management.
