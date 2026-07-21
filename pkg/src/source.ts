/**
 * A `SkillSource` is a runtime-agnostic provider of skills: it knows how to
 * enumerate skill metadata and load a full skill by id. The filesystem/blob/
 * network specifics live in the adapter, never in the {@link Registry}.
 *
 * In-memory adapters live here (browser-safe, zero Node imports). The
 * filesystem adapter lives in `@hanzo/skills/node`.
 */
import { buildSkill, toMetadata } from "./build.js";
import type {
  Skill,
  SkillManifest,
  SkillMetadata,
  SkillScope,
} from "./schema.js";
import { splitFrontmatter } from "./parse.js";

export interface SkillSource {
  /** Discover metadata-only entries (no bodies). */
  list(): Promise<SkillMetadata[]>;
  /** Load a full skill (with body) by id; `undefined` if this source lacks it. */
  load(id: string): Promise<Skill | undefined>;
}

/** One skill provided to {@link recordSource}. */
export interface SkillRecord {
  content: string;
  agents?: string | null;
  meta?: string | null;
  scope?: SkillScope;
  source?: string;
  path?: string;
}

export interface RecordSourceOptions {
  /** Default scope for records that do not specify one. */
  scope?: SkillScope;
  /** Plugin namespace applied to every id (dev parity). */
  namespace?: string;
}

/**
 * An in-memory source over a map of `id -> content | SkillRecord`. This backs
 * the app's compiled-in skills and any blob store (e.g. localStorage) once the
 * blobs have been read into memory.
 */
export function recordSource(
  records: Record<string, string | SkillRecord>,
  options: RecordSourceOptions = {},
): SkillSource {
  const defaultScope: SkillScope = options.scope ?? "builtin";

  const skills = new Map<string, Skill>();
  for (const [key, value] of Object.entries(records)) {
    const record: SkillRecord =
      typeof value === "string" ? { content: value } : value;
    const skill = buildSkill({
      content: record.content,
      scope: record.scope ?? defaultScope,
      source: record.source,
      path: record.path,
      id: key,
      namespace: options.namespace,
      agents: record.agents,
      meta: record.meta,
    });
    skills.set(skill.id, skill);
  }

  return {
    async list() {
      return [...skills.values()].map(toMetadata);
    },
    async load(id) {
      return skills.get(id);
    },
  };
}

/**
 * A source over a prebuilt {@link SkillManifest}. Discovery returns the
 * manifest's metadata directly (no re-parse); `load` reads the body on demand
 * via the supplied `readText`, keyed on the entry `path`. This is how a JS
 * consumer reads the same catalog the Rust harness reads.
 */
export function manifestSource(
  manifest: SkillManifest,
  readText: (path: string) => Promise<string>,
): SkillSource {
  const byId = new Map(manifest.skills.map((s) => [s.id, s]));

  return {
    async list() {
      return manifest.skills.map((s) => ({ ...s }));
    },
    async load(id) {
      const entry = byId.get(id);
      if (!entry || !entry.path) return undefined;
      const content = await readText(entry.path);
      const { body } = splitFrontmatter(content);
      return {
        ...entry,
        isBuiltIn: entry.scope === "builtin",
        frontmatter: {
          name: entry.name,
          description: entry.description,
          license: entry.license,
          source: entry.source,
          ...(entry.shortDescription
            ? { metadata: { "short-description": entry.shortDescription } }
            : {}),
        },
        body,
        content,
      };
    },
  };
}
