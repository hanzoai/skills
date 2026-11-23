---
name: harness-hacker
description: Self-modify the bot framework — fork, improve, test, and PR changes to any Hanzo component including your own source code, skills, extensions, and tools. Use when self-improvement maintenance proposes changes or when you detect friction you can fix.
metadata: { "bot": { "emoji": "🔧", "requires": { "bins": ["git", "pnpm"] } } }
---

# Harness Hacker — Self-Modification Protocol

You have the ability to modify your own source code, create new tools, improve existing skills, and contribute changes back to any Hanzo repository. This is your most powerful capability.

## When to Use

1. **After a maintenance pass** proposes an approved change
2. **During a Build It Now cycle** (Loop 1) when friction is detected
3. **When the user explicitly asks** you to improve yourself
4. **When you identify a pattern** that a specific tool would solve better than `bash`

## Safety Rules

1. **NEVER modify files in the live bot directory while the gateway is running** — use a worktree
2. **ALWAYS test changes** before committing — run `pnpm test` or the relevant test suite
3. **ALWAYS create a new branch** — never commit directly to `main`
4. **ALWAYS explain what you're changing and why** before doing it
5. **Keep changes small and focused** — one logical change per commit
6. **Generated tools go in `extensions/self-improvement/tools/generated/`** — they have a 7-day TTL

## Creating a New Extension

Extensions are the primary way to add capabilities. Follow this exact pattern:

```bash
# 1. Create directory
mkdir -p ~/botd/extensions/my-extension

# 2. Create package.json
cat > ~/botd/extensions/my-extension/package.json << 'EOF'
{
  "name": "@hanzo/bot-my-extension",
  "version": "2026.2.26",
  "private": true,
  "description": "What this extension does",
  "type": "module",
  "devDependencies": { "@hanzo/bot": "workspace:*" },
  "peerDependencies": { "@hanzo/bot": ">=2026.1.26" },
  "bot": { "extensions": ["./index.ts"] }
}
EOF

# 3. Create index.ts
cat > ~/botd/extensions/my-extension/index.ts << 'EOTS'
import type { AnyAgentTool, BotPluginApi } from "bot/plugin-sdk";
import { emptyPluginConfigSchema } from "bot/plugin-sdk";

const plugin = {
  id: "my-extension",
  name: "My Extension",
  description: "What it does",
  configSchema: emptyPluginConfigSchema(),
  register(api: BotPluginApi) {
    api.registerTool({
      name: "my_tool",
      description: "What the tool does",
      parameters: {
        type: "object",
        properties: {
          input: { type: "string", description: "Input parameter" }
        },
        required: ["input"]
      },
      async execute(params) {
        // Implementation here
        return { content: [{ type: "text", text: "Result" }] };
      }
    } as AnyAgentTool);
  }
};

export default plugin;
EOTS
```

## Creating a New Skill

Skills are documentation files that teach you patterns. They're simpler than extensions.

```bash
# 1. Create directory
mkdir -p ~/botd/skills/my-skill

# 2. Create SKILL.md
cat > ~/botd/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: What this skill teaches
metadata:
  { "bot": { "emoji": "📚" } }
---

# Skill Name

## When to Use
Describe the trigger conditions.

## Pattern
Show the exact steps or commands.

## Rules
1. Rule one
2. Rule two
EOF
```

## Modifying Existing Extensions

Use git worktrees for safety:

```bash
# 1. Create worktree from the bot repo
cd ~/botd
git worktree add /tmp/bot-hack-$(date +%s) -b hack/my-improvement

# 2. Make changes in the worktree
cd /tmp/bot-hack-*
# Edit files...

# 3. Test
pnpm test

# 4. Commit
git add -A
git commit -m "feat(self-improvement): description of change"

# 5. Push and PR (if user approves)
git push -u origin hack/my-improvement
gh pr create --title "..." --body "..."

# 6. Cleanup
cd ~/botd
git worktree remove /tmp/bot-hack-*
```

## Hacking Other Hanzo Components

You can improve ANY Hanzo repo, not just the bot. Common targets:

| Component       | Path                     | What to Improve                                 |
| --------------- | ------------------------ | ----------------------------------------------- |
| **MCP**         | `~/work/hanzo/mcp`       | Add new tools, improve search                   |
| **Agent SDK**   | `~/work/hanzo/agent`     | Network patterns, memory                        |
| **LLM Gateway** | `~/work/hanzo/llm`       | Provider support, routing                       |
| **Operative**   | `~/work/hanzo/operative` | Computer use tools                              |
| **Universe**    | `~/work/hanzo/universe`  | K8s manifests, infra                            |
| **Zoo Gym**     | `~/work/zoo/gym`         | GRPO pipeline, SemanticExtractor, BitDelta, DSO |
| **Zoo Node**    | `~/work/zoo/node`        | Rust BitDelta/GRPO (hanzo-libs/hanzo-llm)       |
| **Zen Models**  | `~/work/hanzo/zen`       | Model serving, quantization configs             |

Pattern for external repos:

```bash
# 1. Fork/branch
cd ~/work/hanzo/mcp
git checkout -b hack/add-tool-xyz

# 2. Make changes
# 3. Test: pnpm test (or uv run pytest, cargo test, etc.)
# 4. Commit and push
# 5. Create PR
```

## Runtime Tool Creation (Build It Now — Loop 1)

When you detect friction during a session, you have 5 minutes to build a fix:

```bash
# 1. Identify the pattern (from telemetry)
#    e.g., "bash fails 3x trying to parse JSON from package.json"

# 2. Create a generated tool (these have a 7-day TTL)
mkdir -p ~/botd/extensions/self-improvement/tools/generated

cat > ~/botd/extensions/self-improvement/tools/generated/read-package-field.ts << 'EOTS'
// Generated tool: read_package_field
// Created: $(date -Iseconds)
// Trigger: bash failed 3x parsing JSON from package.json
// TTL: 7 days (auto-pruned if unused)

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

export const tool = {
  name: "read_package_field",
  description: "Read a specific field from package.json. Safer than bash + jq for JSON extraction.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Directory containing package.json" },
      field: { type: "string", description: "JSON field path (e.g., 'version', 'scripts.build')" }
    },
    required: ["field"]
  },
  async execute(params: any) {
    const dir = params.path || process.cwd();
    const pkgPath = join(dir, "package.json");
    if (!existsSync(pkgPath)) {
      return { content: [{ type: "text", text: `No package.json found at ${dir}` }] };
    }
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    const parts = (params.field as string).split(".");
    let value: any = pkg;
    for (const part of parts) {
      value = value?.[part];
    }
    return {
      content: [{ type: "text", text: value !== undefined ? JSON.stringify(value) : "Field not found" }]
    };
  }
};
EOTS

# 3. Hot-reload (the self-improvement extension watches this directory)
```

## Continuous Improvement Workflow

```
Friction detected (Loop 1)
    → Build specific tool (5 min time-box)
    → Test against failure cases
    → Hot-reload into session

Session ends (Loop 3)
    → 3-question reflection
    → Structured telemetry summary

Every 5 sessions (Loop 4)
    → Aggregate telemetry
    → Generate proposals
    → Human approves/rejects
    → YOU implement approved changes using THIS skill
```

## Telemetry-Driven Decisions

Always check telemetry before building:

```bash
# View recent session telemetry
ls -la ~/.hanzo/bot/telemetry/

# View last reflection
cat ~/.hanzo/bot/self-improvement/reflections/$(ls -t ~/.hanzo/bot/self-improvement/reflections/ | head -1)

# View maintenance proposals
cat ~/.hanzo/bot/self-improvement/proposals/$(ls -t ~/.hanzo/bot/self-improvement/proposals/ | head -1)

# View learned facts
cat ~/.hanzo/bot/self-improvement/learned-facts.json
```

## Remember

- **Specific tools beat generic flexibility.** `read_package_field` at 100% success > `bash + jq` at 84%.
- **Telemetry is truth.** Don't guess — look at the numbers.
- **No backlogs.** Build it now or let it live in telemetry for the maintenance pass.
- **Test everything.** Show tests passing, don't just say "done."
- **Small changes compound.** 11 tools over 100 sessions is how bmo improved.
