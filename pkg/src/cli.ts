#!/usr/bin/env node
/**
 * `hanzo-skills` — a tiny CLI over the shared registry.
 *
 *   hanzo-skills build <dir...> [-o out.json] [--scope <scope>] [--namespace <ns>] [--source <label>]
 *   hanzo-skills list  <dir...> [--scope <scope>]
 *   hanzo-skills validate <dir...>
 *
 * `build` writes the portable {@link SkillManifest} JSON — the artefact a
 * non-JS runtime (the dev Rust harness, CI, a Go service) reads instead of
 * re-implementing frontmatter parsing.
 */
import { writeFile } from "node:fs/promises";
import { dirSource, scanDir, type DirSourceOptions } from "./node.js";
import { Registry } from "./registry.js";
import type { SkillScope } from "./schema.js";

interface Args {
  command: string;
  dirs: string[];
  out?: string;
  scope?: SkillScope;
  namespace?: string;
  source?: string;
}

function parseArgs(argv: string[]): Args {
  const [command = "help", ...rest] = argv;
  const dirs: string[] = [];
  const args: Args = { command, dirs };
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    switch (arg) {
      case "-o":
      case "--out":
        args.out = rest[++i];
        break;
      case "--scope":
        args.scope = rest[++i] as SkillScope;
        break;
      case "--namespace":
        args.namespace = rest[++i];
        break;
      case "--source":
        args.source = rest[++i];
        break;
      default:
        if (arg && !arg.startsWith("-")) dirs.push(arg);
    }
  }
  return args;
}

function dirOptions(args: Args): DirSourceOptions {
  return { scope: args.scope, namespace: args.namespace, source: args.source };
}

const USAGE = `hanzo-skills — shared skill registry

Usage:
  hanzo-skills build <dir...> [-o out.json] [--scope <scope>] [--namespace <ns>] [--source <label>]
  hanzo-skills list  <dir...> [--scope <scope>]
  hanzo-skills validate <dir...>

Scopes: user | repo | system | admin | builtin (default: repo)`;

async function main(): Promise<number> {
  const args = parseArgs(process.argv.slice(2));

  if (args.command === "help" || args.command === "--help" || args.command === "-h") {
    process.stdout.write(`${USAGE}\n`);
    return 0;
  }

  if (args.dirs.length === 0) {
    process.stderr.write(`error: no skill directory given\n\n${USAGE}\n`);
    return 2;
  }

  switch (args.command) {
    case "build": {
      const registry = new Registry(
        args.dirs.map((d) => dirSource(d, dirOptions(args))),
      );
      const manifest = await registry.toManifest();
      const json = `${JSON.stringify(manifest, null, 2)}\n`;
      if (args.out) {
        await writeFile(args.out, json, "utf8");
        process.stderr.write(
          `wrote ${manifest.skills.length} skills to ${args.out}\n`,
        );
      } else {
        process.stdout.write(json);
      }
      return 0;
    }
    case "list": {
      const registry = new Registry(
        args.dirs.map((d) => dirSource(d, dirOptions(args))),
      );
      for (const skill of await registry.list()) {
        process.stdout.write(
          `${skill.id.padEnd(36)} ${skill.scope.padEnd(8)} ${skill.description}\n`,
        );
      }
      return 0;
    }
    case "validate": {
      let errorCount = 0;
      let skillCount = 0;
      for (const dir of args.dirs) {
        const { skills, errors } = await scanDir(dir, dirOptions(args));
        skillCount += skills.length;
        for (const err of errors) {
          errorCount++;
          process.stderr.write(`invalid ${dir}/${err.path}: ${err.message}\n`);
        }
      }
      process.stderr.write(
        `${skillCount} valid, ${errorCount} invalid\n`,
      );
      return errorCount > 0 ? 1 : 0;
    }
    default:
      process.stderr.write(`unknown command: ${args.command}\n\n${USAGE}\n`);
      return 2;
  }
}

main().then(
  (code) => process.exit(code),
  (err) => {
    process.stderr.write(`${err instanceof Error ? err.stack : String(err)}\n`);
    process.exit(1);
  },
);
