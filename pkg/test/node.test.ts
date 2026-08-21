import { describe, it, expect } from "vitest";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { scanDir, dirSource } from "../src/node.js";
import { Registry } from "../src/registry.js";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(here, "fixtures");

describe("scanDir over mixed-format fixtures", () => {
  it("discovers every valid SKILL.md and collects the invalid one", async () => {
    const { skills, errors } = await scanDir(FIXTURES, { scope: "repo" });

    const ids = skills.map((s) => s.id).sort();
    expect(ids).toEqual(["frontend-design", "planning", "skill-creator"]);

    // one invalid fixture (missing `name`) is reported, not thrown
    expect(errors).toHaveLength(1);
    expect(errors[0]?.path).toContain("invalid/broken/SKILL.md");
    expect(errors[0]?.message).toMatch(/name/);
  });

  it("parses dev-style skills (metadata.short-description + agents/openai.yaml)", async () => {
    const { skills } = await scanDir(FIXTURES);
    const creator = skills.find((s) => s.id === "skill-creator");
    expect(creator?.shortDescription).toBe("Create or update a skill");
    expect(creator?.interface?.display_name).toBe("Skill Creator");
    expect(creator?.interface?.brand_color).toBe("#7C5CFF");
    expect(creator?.dependencies?.tools[0]?.type).toBe("mcp");
    expect(creator?.policy?.allow_implicit_invocation).toBe(true);
    expect(creator?.policy?.products).toEqual(["dev", "app"]);
    // portable relative posix path
    expect(creator?.path).toBe("dev-style/skill-creator/SKILL.md");
  });

  it("parses desktop-style skills (.skill-meta.json + source/license)", async () => {
    const { skills } = await scanDir(FIXTURES);
    const fd = skills.find((s) => s.id === "frontend-design");
    expect(fd?.source).toBe("anthropics/skills");
    expect(fd?.license).toBe("Apache-2.0");
    expect(fd?.meta?.source).toBe("registry");
    expect(fd?.meta?.installedAt).toBe("2026-01-27T18:07:15.552Z");
  });

  it("parses app-style skills (name/description/license)", async () => {
    const { skills } = await scanDir(FIXTURES);
    const planning = skills.find((s) => s.id === "planning");
    expect(planning?.license).toBe("MIT");
    expect(planning?.description).toBe(
      "Break work into a clear, ordered plan before implementing.",
    );
  });
});

describe("dirSource + Registry", () => {
  it("loads a full skill body through the registry", async () => {
    const registry = new Registry(dirSource(FIXTURES, { scope: "repo" }));
    const loaded = await registry.load("skill-creator");
    expect(loaded?.name).toBe("skill-creator");
    expect(loaded?.body).toContain("# Skill Creator");
    expect(loaded?.body).not.toContain("---");
  });

  it("emits a manifest for the fixture tree", async () => {
    const registry = new Registry(dirSource(FIXTURES));
    const manifest = await registry.toManifest();
    expect(manifest.skills).toHaveLength(3);
    expect(manifest.skills.every((s) => s.path?.endsWith("SKILL.md"))).toBe(true);
  });
});

// The catalogue this repo ships, parsed by the parser this repo ships. The
// block that stood here named absolute paths under another machine's home, so
// skipIf skipped it everywhere and nothing was ever checked. This one resolves
// from the test file, so it runs wherever the repo is checked out.
describe("the catalogue this repo ships", () => {
  const CATALOG = join(here, "..", "..", "skills");

  it("parses every SKILL.md with no errors", async () => {
    const { skills, errors } = await scanDir(CATALOG, { scope: "repo" });
    expect(errors).toEqual([]);
    expect(skills.length).toBeGreaterThan(20);
  });

  it("gives every skill an id that matches its directory", async () => {
    const { skills } = await scanDir(CATALOG, { scope: "repo" });
    const wrong = skills.filter(
      (s) => s.path && !s.path.endsWith(`${s.id}/SKILL.md`),
    );
    expect(wrong.map((s) => s.path)).toEqual([]);
  });
});
