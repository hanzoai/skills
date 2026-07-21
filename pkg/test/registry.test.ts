import { describe, it, expect } from "vitest";
import { Registry, buildManifest } from "../src/registry.js";
import { recordSource, manifestSource } from "../src/source.js";
import { SKILL_MANIFEST_VERSION } from "../src/schema.js";

const PLANNING = `---\nname: planning\ndescription: Plan before implementing.\nlicense: MIT\n---\n# Planning\nDecompose the task.\n`;
const ONE_SHOT = `---\nname: one-shot\ndescription: Build in a single pass.\n---\n# One Shot\nGo.\n`;

describe("recordSource + Registry", () => {
  it("lists, gets and loads built-in skills", async () => {
    const registry = new Registry(
      recordSource({ planning: PLANNING, "one-shot": ONE_SHOT }),
    );

    const list = await registry.list();
    expect(list.map((s) => s.id).sort()).toEqual(["one-shot", "planning"]);

    const planning = await registry.get("planning");
    expect(planning?.scope).toBe("builtin");
    expect(planning?.license).toBe("MIT");

    const loaded = await registry.load("planning");
    expect(loaded?.isBuiltIn).toBe(true);
    expect(loaded?.body).toBe("# Planning\nDecompose the task.");
    expect(loaded?.content).toBe(PLANNING);
  });

  it("returns undefined for unknown ids", async () => {
    const registry = new Registry(recordSource({ planning: PLANNING }));
    expect(await registry.get("nope")).toBeUndefined();
    expect(await registry.load("nope")).toBeUndefined();
  });
});

describe("multi-source dedup", () => {
  const base = recordSource(
    { planning: PLANNING },
    { scope: "system", source: "base" },
  );
  const override = recordSource(
    { planning: `---\nname: planning\ndescription: Overridden.\n---\nbody\n` },
    { scope: "user", source: "user" },
  );

  it("first-wins by default", async () => {
    const registry = new Registry([base, override]);
    const planning = await registry.get("planning");
    expect(planning?.description).toBe("Plan before implementing.");
    expect(planning?.scope).toBe("system");
  });

  it("last-wins when configured", async () => {
    const registry = new Registry([base, override], {
      onDuplicate: "last-wins",
    });
    const planning = await registry.get("planning");
    expect(planning?.description).toBe("Overridden.");
    expect(planning?.scope).toBe("user");
  });
});

describe("namespacing", () => {
  it("prefixes ids with the source namespace (dev parity)", async () => {
    const registry = new Registry(
      recordSource({ planning: PLANNING }, { namespace: "acme" }),
    );
    const list = await registry.list();
    expect(list[0]?.id).toBe("acme:planning");
  });
});

describe("manifest round-trip", () => {
  it("builds a manifest and reads it back via manifestSource", async () => {
    // The manifest carries metadata + a `path` per skill; bodies are resolved
    // on demand. This is the exact shape a non-JS runtime (the Rust harness)
    // consumes: read the catalog, fetch a body by path only when needed.
    const manifest = await buildManifest(
      recordSource(
        {
          planning: { content: PLANNING, path: "planning.md" },
          "one-shot": { content: ONE_SHOT, path: "one-shot.md" },
        },
        { scope: "repo", source: "catalog" },
      ),
    );
    expect(manifest.version).toBe(SKILL_MANIFEST_VERSION);
    expect(manifest.skills).toHaveLength(2);
    expect(manifest.skills.every((s) => typeof s.path === "string")).toBe(true);

    const bodies: Record<string, string> = {
      "planning.md": PLANNING,
      "one-shot.md": ONE_SHOT,
    };
    const registry = new Registry(
      manifestSource(manifest, async (p) => bodies[p] ?? ""),
    );
    const loaded = await registry.load("planning");
    expect(loaded?.name).toBe("planning");
    expect(loaded?.body).toBe("# Planning\nDecompose the task.");
    expect(loaded?.frontmatter.description).toBe("Plan before implementing.");
  });
});
