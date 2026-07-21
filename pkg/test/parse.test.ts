import { describe, it, expect } from "vitest";
import {
  parseSkillDoc,
  splitFrontmatter,
  stringifySkillDoc,
  parseAgentsFile,
  parseSkillMeta,
  shortDescriptionOf,
  createSkillTemplate,
  SkillParseError,
} from "../src/parse.js";

const DEV_DOC = `---
name: skill-creator
description: Guide for creating effective skills.
metadata:
  short-description: Create or update a skill
---

# Skill Creator

Body text.
`;

describe("splitFrontmatter", () => {
  it("splits frontmatter from body", () => {
    const { frontmatter, body } = splitFrontmatter(DEV_DOC);
    expect(frontmatter).toContain("name: skill-creator");
    expect(body).toBe("# Skill Creator\n\nBody text.");
  });

  it("returns null frontmatter when absent", () => {
    const { frontmatter, body } = splitFrontmatter("# no frontmatter\n");
    expect(frontmatter).toBeNull();
    expect(body).toBe("# no frontmatter\n");
  });
});

describe("parseSkillDoc", () => {
  it("parses the nested metadata.short-description block", () => {
    const { frontmatter } = parseSkillDoc(DEV_DOC);
    expect(frontmatter.name).toBe("skill-creator");
    expect(shortDescriptionOf(frontmatter)).toBe("Create or update a skill");
  });

  it("keeps app + desktop fields (license, source)", () => {
    const { frontmatter } = parseSkillDoc(
      `---\nname: frontend-design\ndescription: Design things.\nsource: anthropics/skills\nlicense: Apache-2.0\n---\nbody\n`,
    );
    expect(frontmatter.source).toBe("anthropics/skills");
    expect(frontmatter.license).toBe("Apache-2.0");
  });

  it("throws on missing frontmatter", () => {
    expect(() => parseSkillDoc("# nope")).toThrow(SkillParseError);
  });

  it("throws on missing name", () => {
    expect(() => parseSkillDoc(`---\ndescription: no name\n---\nbody\n`)).toThrow(
      /name/,
    );
  });

  it("rejects non-portable names (uppercase / spaces)", () => {
    expect(() =>
      parseSkillDoc(`---\nname: Not Portable\ndescription: x\n---\nbody\n`),
    ).toThrow(SkillParseError);
  });
});

describe("stringifySkillDoc round-trip", () => {
  it("re-parses to the same frontmatter and body", () => {
    const { frontmatter, body } = parseSkillDoc(DEV_DOC);
    const out = stringifySkillDoc(frontmatter, body);
    const again = parseSkillDoc(out);
    expect(again.frontmatter.name).toBe(frontmatter.name);
    expect(again.frontmatter.description).toBe(frontmatter.description);
    expect(shortDescriptionOf(again.frontmatter)).toBe(
      "Create or update a skill",
    );
    expect(again.body).toBe(body);
  });
});

describe("parseAgentsFile", () => {
  it("parses interface, dependencies and policy", () => {
    const agents = parseAgentsFile(
      `interface:\n  display_name: Skill Creator\n  brand_color: "#7C5CFF"\ndependencies:\n  tools:\n    - type: mcp\n      value: filesystem\npolicy:\n  allow_implicit_invocation: false\n  products:\n    - dev\n`,
    );
    expect(agents.interface?.display_name).toBe("Skill Creator");
    expect(agents.dependencies?.tools[0]?.type).toBe("mcp");
    expect(agents.policy?.allow_implicit_invocation).toBe(false);
    expect(agents.policy?.products).toEqual(["dev"]);
  });
});

describe("parseSkillMeta", () => {
  it("parses the desktop .skill-meta.json sidecar", () => {
    const meta = parseSkillMeta(
      `{"source":"registry","name":"frontend-design","installedAt":"2026-01-27T18:07:15.552Z","updatedAt":"2026-01-27T18:07:15.552Z"}`,
    );
    expect(meta.source).toBe("registry");
    expect(meta.installedAt).toBe("2026-01-27T18:07:15.552Z");
  });
});

describe("createSkillTemplate", () => {
  it("produces a schema-valid document", () => {
    const doc = createSkillTemplate("My Skill", "Does a thing.");
    const { frontmatter } = parseSkillDoc(doc);
    expect(frontmatter.name).toBe("my-skill");
    expect(frontmatter.description).toBe("Does a thing.");
  });
});
