/**
 * @hanzo/skills — the shared skill data model, parser, registry and loader.
 *
 * Skill DATA (a `SKILL.md` plus optional `agents/openai.yaml` and
 * `.skill-meta.json` sidecars) is the single source of truth; app, desktop and
 * the dev harness all consume it through this one registry/loader instead of
 * maintaining private skill systems.
 *
 * This entry point is runtime-agnostic and browser-safe (no Node imports). For
 * filesystem discovery use `@hanzo/skills/node`.
 */
export * from "./schema.js";
export {
  SkillParseError,
  splitFrontmatter,
  parseSkillDoc,
  parseAgentsFile,
  parseSkillMeta,
  stringifySkillDoc,
  shortDescriptionOf,
  createSkillTemplate,
} from "./parse.js";
export { buildSkill, toMetadata, type BuildSkillInput } from "./build.js";
export {
  recordSource,
  manifestSource,
  type SkillSource,
  type SkillRecord,
  type RecordSourceOptions,
} from "./source.js";
export {
  Registry,
  buildManifest,
  type RegistryOptions,
} from "./registry.js";
