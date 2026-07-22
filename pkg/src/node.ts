/**
 * `@hanzo/skills/node` — filesystem discovery of on-disk skills.
 *
 * Walks a root for `SKILL.md` files (one per skill folder, matching dev's
 * `codex-core-skills` loader) and reads the sibling `agents/openai.yaml` and
 * `.skill-meta.json` sidecars. This is how desktop's `.cursor/skills`, the
 * `hanzoai/skills` catalog, and dev's on-disk skill roots are read.
 *
 * Node-only. The browser-safe core is `@hanzo/skills`.
 */
import { readFile, readdir } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import { buildSkill, toMetadata } from "./build.js";
import type { SkillSource } from "./source.js";
import type { Skill, SkillMetadata, SkillScope } from "./schema.js";

export const SKILL_FILENAME = "SKILL.md";
export const AGENTS_METADATA_PATH = join("agents", "openai.yaml");
export const SKILL_META_FILENAME = ".skill-meta.json";

const IGNORED_DIRS = new Set(["node_modules", ".git", "dist", "target"]);
const DEFAULT_MAX_DEPTH = 8;

export interface DirSourceOptions {
  /** Scope assigned to discovered skills. Default `repo`. */
  scope?: SkillScope;
  /** Plugin namespace applied to every id (dev parity). */
  namespace?: string;
  /** Provenance label recorded on each skill. */
  source?: string;
  /** Directory-recursion guard. Default 8. */
  maxDepth?: number;
  /** Skill document filename. Default `SKILL.md` (matches dev). */
  filename?: string;
}

export interface ScanError {
  path: string;
  message: string;
}

export interface ScanResult {
  skills: Skill[];
  errors: ScanError[];
}

/**
 * Recursively read every skill under `root`. Fault-tolerant: a malformed
 * `SKILL.md` is collected in `errors` and does not abort the scan (matching
 * dev's fail-open loader).
 */
export async function scanDir(
  root: string,
  options: DirSourceOptions = {},
): Promise<ScanResult> {
  const scope: SkillScope = options.scope ?? "repo";
  const filename = options.filename ?? SKILL_FILENAME;
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;

  const skills: Skill[] = [];
  const errors: ScanError[] = [];

  const docPaths: string[] = [];
  await walk(root, 0, maxDepth, filename, docPaths);

  for (const docPath of docPaths) {
    try {
      const [content, agents, meta] = await Promise.all([
        readFile(docPath, "utf8"),
        readOptional(join(dirOf(docPath), AGENTS_METADATA_PATH)),
        readOptional(join(dirOf(docPath), SKILL_META_FILENAME)),
      ]);
      skills.push(
        buildSkill({
          content,
          scope,
          namespace: options.namespace,
          source: options.source,
          path: toPosix(relative(root, docPath)),
          agents,
          meta,
        }),
      );
    } catch (err) {
      errors.push({
        path: toPosix(relative(root, docPath)),
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return { skills, errors };
}

/**
 * A {@link SkillSource} backed by a directory tree. Scans once and memoises.
 * Invalid skills are silently skipped; use {@link scanDir} to surface errors.
 */
export function dirSource(
  root: string,
  options: DirSourceOptions = {},
): SkillSource {
  let scanned: Promise<Map<string, Skill>> | undefined;
  const ensure = () => {
    if (!scanned) {
      scanned = scanDir(root, options).then(
        (r) => new Map(r.skills.map((s) => [s.id, s])),
      );
    }
    return scanned;
  };

  return {
    async list(): Promise<SkillMetadata[]> {
      return [...(await ensure()).values()].map(toMetadata);
    },
    async load(id: string): Promise<Skill | undefined> {
      return (await ensure()).get(id);
    },
  };
}

async function walk(
  dir: string,
  depth: number,
  maxDepth: number,
  filename: string,
  out: string[],
): Promise<void> {
  if (depth > maxDepth) return;

  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    if (entry.isDirectory()) {
      const hidden = entry.name.startsWith(".");
      // Skip build/vcs dirs and hidden dirs, but allow `.cursor` so a scan from
      // a repo root reaches desktop's `.cursor/skills`.
      const skip = IGNORED_DIRS.has(entry.name) || (hidden && entry.name !== ".cursor");
      if (skip) continue;
      await walk(join(dir, entry.name), depth + 1, maxDepth, filename, out);
    } else if (entry.isFile() && entry.name === filename) {
      out.push(join(dir, entry.name));
    }
  }
}

async function readOptional(path: string): Promise<string | null> {
  try {
    return await readFile(path, "utf8");
  } catch {
    return null;
  }
}

function dirOf(filePath: string): string {
  const idx = filePath.lastIndexOf(sep);
  return idx === -1 ? "." : filePath.slice(0, idx);
}

function toPosix(p: string): string {
  return p.split(sep).join("/");
}
