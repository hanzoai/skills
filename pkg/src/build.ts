/**
 * Pure construction of a {@link Skill} from its raw texts. Shared by every
 * source adapter (in-memory, filesystem, manifest) so parsing lives in exactly
 * one place.
 */
import { parseAgentsFile, parseSkillDoc, parseSkillMeta, shortDescriptionOf } from "./parse.js";
import type { Skill, SkillMetadata, SkillScope } from "./schema.js";

export interface BuildSkillInput {
  /** Raw `SKILL.md` text (frontmatter + body). */
  content: string;
  scope: SkillScope;
  /** Doc path relative to the source root; carried into the manifest. */
  path?: string;
  /** Explicit id; defaults to the frontmatter `name`. */
  id?: string;
  /** Plugin namespace — when set, id becomes `${namespace}:${name}` (dev parity). */
  namespace?: string;
  /** Provenance label; falls back to frontmatter `source`, then `.skill-meta.json` source. */
  source?: string;
  isBuiltIn?: boolean;
  /** Raw `agents/openai.yaml` text, when present. */
  agents?: string | null;
  /** Raw `.skill-meta.json` text, when present. */
  meta?: string | null;
}

/**
 * Parse + assemble a fully-loaded skill. Throws {@link SkillParseError} when the
 * `SKILL.md` frontmatter is missing or invalid.
 */
export function buildSkill(input: BuildSkillInput): Skill {
  const { frontmatter, body } = parseSkillDoc(input.content);

  const baseId = input.id ?? frontmatter.name;
  const id = input.namespace ? `${input.namespace}:${baseId}` : baseId;

  const agents = input.agents ? parseAgentsFile(input.agents) : undefined;
  const meta = input.meta ? parseSkillMeta(input.meta) : undefined;

  const scope: SkillScope = meta?.scope ?? input.scope;
  const source = input.source ?? frontmatter.source ?? meta?.source;

  return {
    id,
    name: frontmatter.name,
    description: frontmatter.description,
    shortDescription: shortDescriptionOf(frontmatter),
    scope,
    source,
    license: frontmatter.license,
    path: input.path,
    interface: agents?.interface,
    dependencies: agents?.dependencies,
    policy: agents?.policy,
    meta,
    frontmatter,
    body,
    content: input.content,
    isBuiltIn: input.isBuiltIn ?? scope === "builtin",
  };
}

/** Project a loaded skill down to its metadata-only view (drops the body). */
export function toMetadata(skill: Skill): SkillMetadata {
  return {
    id: skill.id,
    name: skill.name,
    description: skill.description,
    shortDescription: skill.shortDescription,
    scope: skill.scope,
    source: skill.source,
    license: skill.license,
    path: skill.path,
    interface: skill.interface,
    dependencies: skill.dependencies,
    policy: skill.policy,
    meta: skill.meta,
  };
}
