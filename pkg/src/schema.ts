/**
 * Unified skill DATA schema for Hanzo.
 *
 * One schema is the single source of truth for a skill on disk and in memory.
 * It is a strict superset of the three existing systems so a skill authored
 * against it loads unchanged in every consumer:
 *
 *  - dev (Rust `codex-core-skills`): `SKILL.md` frontmatter (`name`, `description`,
 *    `metadata.short-description`) + an optional `agents/openai.yaml` sidecar
 *    carrying `interface` / `dependencies` / `policy`.
 *  - app (`lib/vfs/skills`): `SKILL.md` frontmatter (`name`, `description`, `license`).
 *  - desktop (`.cursor/skills`): `SKILL.md` frontmatter (`name`, `description`,
 *    `source`, `license`) + a `.skill-meta.json` install-tracking sidecar.
 *
 * The schema is expressed with zod so the schema, the TypeScript types, and the
 * runtime validator are one artefact — no drift between them.
 */
import { z } from "zod";

/**
 * Where a skill was discovered. Mirrors dev's `SkillScope` (User/Repo/System/Admin)
 * and adds `builtin` for the app's compiled-in skills (its `isBuiltIn` flag).
 */
export const SkillScopeSchema = z.enum([
  "user",
  "repo",
  "system",
  "admin",
  "builtin",
]);
export type SkillScope = z.infer<typeof SkillScopeSchema>;

/**
 * Portable skill-name convention: lowercase, digits and single hyphens.
 * This is the de-facto intersection already honoured by every source
 * (dev samples, the hanzoai/skills catalog, desktop, app), so a name that
 * validates here is valid everywhere. Plugin namespacing (`plugin:name`,
 * as dev does) is applied at the registry/id layer, not in the name itself.
 */
export const SKILL_NAME_MAX = 64;
export const SKILL_DESCRIPTION_MAX = 1024;
export const SKILL_SHORT_DESCRIPTION_MAX = 1024;

export const SkillNameSchema = z
  .string()
  .min(1)
  .max(SKILL_NAME_MAX)
  .regex(
    /^[a-z0-9]([a-z0-9-]*[a-z0-9])?$/,
    "skill name must be lowercase alphanumerics separated by single hyphens",
  );

/**
 * The nested `metadata:` block of `SKILL.md` frontmatter. The on-disk key is
 * hyphenated (`short-description`) to match dev; it is normalised to
 * `shortDescription` on the parsed {@link Skill}.
 */
export const SkillFrontmatterMetadataSchema = z
  .object({
    "short-description": z.string().max(SKILL_SHORT_DESCRIPTION_MAX).optional(),
  })
  .passthrough();

/**
 * `SKILL.md` YAML frontmatter — the primary, portable descriptor. Unknown keys
 * are preserved (`passthrough`) so bespoke fields survive a round-trip.
 */
export const SkillFrontmatterSchema = z
  .object({
    name: SkillNameSchema,
    description: z.string().min(1).max(SKILL_DESCRIPTION_MAX),
    /** SPDX-ish license string (app + desktop). */
    license: z.string().optional(),
    /** Provenance, e.g. "anthropics/skills" (desktop + hanzoai/skills catalog). */
    source: z.string().optional(),
    metadata: SkillFrontmatterMetadataSchema.optional(),
  })
  .passthrough();
export type SkillFrontmatter = z.infer<typeof SkillFrontmatterSchema>;

/**
 * `.skill-meta.json` — the desktop install-tracking sidecar. Every field is
 * optional so a partial file still loads.
 */
export const SkillMetaSchema = z
  .object({
    source: z.string().optional(),
    name: z.string().optional(),
    installedAt: z.string().optional(),
    updatedAt: z.string().optional(),
    version: z.string().optional(),
    scope: SkillScopeSchema.optional(),
  })
  .passthrough();
export type SkillMeta = z.infer<typeof SkillMetaSchema>;

/**
 * Presentation metadata from the `agents/openai.yaml` sidecar. Field names are
 * snake_case to match the on-disk file and dev's `SkillInterface`.
 */
export const SkillInterfaceSchema = z
  .object({
    display_name: z.string().optional(),
    short_description: z.string().optional(),
    icon_small: z.string().optional(),
    icon_large: z.string().optional(),
    brand_color: z.string().optional(),
    default_prompt: z.string().optional(),
  })
  .passthrough();
export type SkillInterface = z.infer<typeof SkillInterfaceSchema>;

/** One declared tool dependency (dev's `SkillToolDependency`). */
export const SkillToolSchema = z
  .object({
    type: z.string(),
    value: z.string().optional(),
    description: z.string().optional(),
    transport: z.string().optional(),
    command: z.string().optional(),
    url: z.string().optional(),
  })
  .passthrough();
export type SkillTool = z.infer<typeof SkillToolSchema>;

export const SkillDependenciesSchema = z.object({
  tools: z.array(SkillToolSchema).default([]),
});
export type SkillDependencies = z.infer<typeof SkillDependenciesSchema>;

/** Invocation / product gating (dev's `SkillPolicy`). */
export const SkillPolicySchema = z
  .object({
    allow_implicit_invocation: z.boolean().optional(),
    products: z.array(z.string()).default([]),
  })
  .passthrough();
export type SkillPolicy = z.infer<typeof SkillPolicySchema>;

/**
 * The `agents/openai.yaml` sidecar as a whole.
 */
export const SkillAgentsFileSchema = z
  .object({
    interface: SkillInterfaceSchema.optional(),
    dependencies: SkillDependenciesSchema.optional(),
    policy: SkillPolicySchema.optional(),
  })
  .passthrough();
export type SkillAgentsFile = z.infer<typeof SkillAgentsFileSchema>;

/**
 * Metadata-only view of a skill (no body). This is what discovery/listing
 * returns and what a manifest carries — it mirrors dev's `SkillMetadata`,
 * where the body is read lazily. `path` is the doc path relative to the
 * source root (portable across machines).
 */
export const SkillMetadataSchema = z.object({
  id: z.string(),
  name: SkillNameSchema,
  description: z.string(),
  shortDescription: z.string().optional(),
  scope: SkillScopeSchema,
  source: z.string().optional(),
  license: z.string().optional(),
  path: z.string().optional(),
  interface: SkillInterfaceSchema.optional(),
  dependencies: SkillDependenciesSchema.optional(),
  policy: SkillPolicySchema.optional(),
  meta: SkillMetaSchema.optional(),
});
export type SkillMetadata = z.infer<typeof SkillMetadataSchema>;

/**
 * A fully-loaded skill: its metadata plus the raw frontmatter and the markdown
 * body (frontmatter stripped). `content` is the original file text.
 */
export const SkillSchema = SkillMetadataSchema.extend({
  frontmatter: SkillFrontmatterSchema,
  /** Markdown body with the frontmatter block removed. */
  body: z.string(),
  /** Original, unmodified `SKILL.md` text (frontmatter + body). */
  content: z.string(),
  isBuiltIn: z.boolean(),
});
export type Skill = z.infer<typeof SkillSchema>;

/**
 * The portable JSON catalog. A non-JS runtime (the dev Rust harness, CI, a Go
 * service) reads this instead of re-implementing frontmatter parsing. Bodies
 * are not embedded — a consumer reads them on demand from `path`.
 */
export const SKILL_MANIFEST_VERSION = 1 as const;

export const SkillManifestSchema = z.object({
  version: z.literal(SKILL_MANIFEST_VERSION),
  generatedAt: z.string(),
  skills: z.array(SkillMetadataSchema),
});
export type SkillManifest = z.infer<typeof SkillManifestSchema>;
