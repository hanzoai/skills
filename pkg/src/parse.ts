/**
 * Parsing and serialisation for the skill DATA formats: `SKILL.md`
 * (YAML frontmatter + markdown body), the `agents/openai.yaml` sidecar,
 * and the `.skill-meta.json` sidecar.
 *
 * Unlike the app's hand-rolled line parser, this uses a real YAML parser so the
 * nested `metadata:` block (and the `agents/openai.yaml` `interface` /
 * `dependencies` / `policy` trees) parse correctly.
 */
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import {
  SkillAgentsFileSchema,
  SkillFrontmatterSchema,
  SkillMetaSchema,
  type SkillAgentsFile,
  type SkillFrontmatter,
  type SkillMeta,
} from "./schema.js";

export class SkillParseError extends Error {
  override readonly name = "SkillParseError";
}

/** Matches a leading `---\n … \n---\n` YAML frontmatter block. */
const FRONTMATTER_RE = /^﻿?---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?([\s\S]*)$/;

/**
 * Split a `SKILL.md` document into its raw frontmatter text and markdown body.
 * Returns `frontmatter: null` when the document has no frontmatter block.
 */
export function splitFrontmatter(content: string): {
  frontmatter: string | null;
  body: string;
} {
  const match = content.match(FRONTMATTER_RE);
  if (!match) {
    return { frontmatter: null, body: content };
  }
  return { frontmatter: match[1] ?? "", body: (match[2] ?? "").trim() };
}

/**
 * Parse and validate a `SKILL.md` document into its frontmatter and body.
 * Throws {@link SkillParseError} on a missing/invalid frontmatter block or a
 * frontmatter that violates the schema.
 */
export function parseSkillDoc(content: string): {
  frontmatter: SkillFrontmatter;
  body: string;
} {
  const { frontmatter, body } = splitFrontmatter(content);
  if (frontmatter === null) {
    throw new SkillParseError(
      "invalid SKILL.md: missing YAML frontmatter (expected a leading '---' block)",
    );
  }

  let raw: unknown;
  try {
    raw = parseYaml(frontmatter);
  } catch (err) {
    throw new SkillParseError(
      `invalid SKILL.md frontmatter YAML: ${errText(err)}`,
    );
  }

  const parsed = SkillFrontmatterSchema.safeParse(raw ?? {});
  if (!parsed.success) {
    throw new SkillParseError(
      `invalid SKILL.md frontmatter: ${parsed.error.issues
        .map((i) => `${i.path.join(".") || "<root>"}: ${i.message}`)
        .join("; ")}`,
    );
  }
  return { frontmatter: parsed.data, body };
}

/**
 * Normalised short description: the frontmatter `metadata.short-description`.
 */
export function shortDescriptionOf(
  frontmatter: SkillFrontmatter,
): string | undefined {
  const value = frontmatter.metadata?.["short-description"];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Serialise frontmatter + body back into a `SKILL.md` document.
 */
export function stringifySkillDoc(
  frontmatter: SkillFrontmatter,
  body: string,
): string {
  const yaml = stringifyYaml(frontmatter).trimEnd();
  return `---\n${yaml}\n---\n\n${body.trim()}\n`;
}

/** Parse and validate an `agents/openai.yaml` sidecar. */
export function parseAgentsFile(text: string): SkillAgentsFile {
  let raw: unknown;
  try {
    raw = parseYaml(text);
  } catch (err) {
    throw new SkillParseError(`invalid agents/openai.yaml: ${errText(err)}`);
  }
  const parsed = SkillAgentsFileSchema.safeParse(raw ?? {});
  if (!parsed.success) {
    throw new SkillParseError(
      `invalid agents/openai.yaml: ${parsed.error.message}`,
    );
  }
  return parsed.data;
}

/** Parse and validate a `.skill-meta.json` sidecar. */
export function parseSkillMeta(text: string): SkillMeta {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    throw new SkillParseError(`invalid .skill-meta.json: ${errText(err)}`);
  }
  const parsed = SkillMetaSchema.safeParse(raw ?? {});
  if (!parsed.success) {
    throw new SkillParseError(`invalid .skill-meta.json: ${parsed.error.message}`);
  }
  return parsed.data;
}

/**
 * Create a starter `SKILL.md` document. Parity with the app's
 * `createSkillTemplate`, but emits schema-valid frontmatter via the shared
 * serialiser.
 */
export function createSkillTemplate(name: string, description: string): string {
  const id = name.toLowerCase().replace(/\s+/g, "-");
  const body = `# ${name}

## Purpose

Describe what this skill helps with.

## Guidelines

- Guideline 1
- Guideline 2

## Examples

Provide code examples or usage patterns.
`;
  return stringifySkillDoc(
    SkillFrontmatterSchema.parse({ name: id, description }),
    body,
  );
}

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
