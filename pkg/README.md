# @hanzo/skills

One shared skill data model, parser, registry and loader for Hanzo. Skill DATA
is the single source of truth; the app, the desktop app, and the dev harness all
consume it through this one package instead of maintaining private skill systems.

## What a skill is (portable DATA)

A skill is a folder with a `SKILL.md` and, optionally, two sidecars:

```
my-skill/
  SKILL.md              # YAML frontmatter + markdown body (required)
  agents/openai.yaml    # rich interface / dependencies / policy (optional)
  .skill-meta.json      # install-tracking sidecar (optional)
```

`SKILL.md` frontmatter is a strict superset of every existing Hanzo skill
system, so a skill authored here loads unchanged in all of them:

```yaml
---
name: my-skill # lowercase, hyphenated (portable across dev/app/desktop)
description: What it does and when to use it.
license: MIT # optional (app, desktop)
source: anthropics/skills # optional provenance (desktop, catalog)
metadata:
  short-description: Short label # optional (dev)
---
```

## Consumers

| Consumer    | How it consumes                                                                             |
| ----------- | ------------------------------------------------------------------------------------------- |
| **app**     | `Registry` + `recordSource` (compiled-in / localStorage skills), browser-safe core          |
| **desktop** | `dirSource` over `.cursor/skills` (`@hanzo/skills/node`)                                     |
| **dev**     | reads the JSON manifest emitted by `hanzo-skills build` (or the same on-disk skill folders) |

## API

```ts
import { Registry, recordSource, parseSkillDoc } from "@hanzo/skills";
import { dirSource, scanDir } from "@hanzo/skills/node"; // Node only

// Browser / any runtime — in-memory skills:
const registry = new Registry(recordSource({ planning: skillMarkdown }));
await registry.list(); // SkillMetadata[]
await registry.load("planning"); // Skill (with body)

// Node — discover a directory tree:
const fsRegistry = new Registry(dirSource("./.cursor/skills", { scope: "user" }));
const manifest = await fsRegistry.toManifest(); // SkillManifest (portable JSON)
```

### CLI — the bridge to non-JS runtimes

```bash
hanzo-skills build ./skills -o skills.manifest.json   # emit the portable catalog
hanzo-skills list ./skills                             # id / scope / description
hanzo-skills validate ./skills                         # non-zero exit on any invalid skill
```

The manifest is metadata-only (bodies read on demand from each `path`), mirroring
how the dev Rust loader lists skills and reads bodies lazily — so the Rust
harness reads the same catalog without re-implementing frontmatter parsing.

## Design

- **Data is the single source.** The package unifies the schema, parser,
  registry and loader; it does not own the skills.
- **Runtime-agnostic core.** The main entry point has no Node imports;
  filesystem access lives behind `SkillSource` adapters (`@hanzo/skills/node`).
- **One schema.** Expressed once in zod, so schema, TypeScript types and the
  runtime validator never drift.

## Develop

```bash
pnpm install
pnpm test        # vitest
pnpm typecheck
pnpm build       # tsc -> dist/
```
