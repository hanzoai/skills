/**
 * The `Registry` composes one or more {@link SkillSource}s into a single,
 * deduplicated catalog and is the one API every JS consumer talks to. It is
 * runtime-agnostic: give it in-memory sources in the browser, a filesystem
 * source in Node, or a manifest source anywhere.
 */
import type { SkillSource } from "./source.js";
import {
  SKILL_MANIFEST_VERSION,
  type Skill,
  type SkillManifest,
  type SkillMetadata,
} from "./schema.js";

export interface RegistryOptions {
  /** Which source wins when two provide the same id. Default `first-wins`. */
  onDuplicate?: "first-wins" | "last-wins";
}

interface Indexed {
  metadata: SkillMetadata[];
  /** id -> index into the sources array that owns it. */
  owner: Map<string, number>;
}

export class Registry {
  private readonly sources: SkillSource[];
  private readonly onDuplicate: "first-wins" | "last-wins";
  private cache: Indexed | undefined;

  constructor(sources: SkillSource | SkillSource[], options: RegistryOptions = {}) {
    this.sources = Array.isArray(sources) ? sources : [sources];
    this.onDuplicate = options.onDuplicate ?? "first-wins";
  }

  /** Discover and cache the merged, deduplicated metadata catalog. */
  async discover(): Promise<SkillMetadata[]> {
    if (this.cache) return this.cache.metadata;

    const byId = new Map<string, SkillMetadata>();
    const owner = new Map<string, number>();

    for (let i = 0; i < this.sources.length; i++) {
      const source = this.sources[i];
      if (!source) continue;
      for (const meta of await source.list()) {
        const seen = byId.has(meta.id);
        if (!seen || this.onDuplicate === "last-wins") {
          byId.set(meta.id, meta);
          owner.set(meta.id, i);
        }
      }
    }

    this.cache = { metadata: [...byId.values()], owner };
    return this.cache.metadata;
  }

  /** The merged metadata catalog (discovers on first call). */
  async list(): Promise<SkillMetadata[]> {
    return this.discover();
  }

  /** Metadata for one skill, or `undefined`. */
  async get(id: string): Promise<SkillMetadata | undefined> {
    await this.discover();
    return this.cache?.metadata.find((s) => s.id === id);
  }

  /** Fully load one skill (with its markdown body), or `undefined`. */
  async load(id: string): Promise<Skill | undefined> {
    await this.discover();
    const ownerIndex = this.cache?.owner.get(id);
    if (ownerIndex === undefined) return undefined;
    return this.sources[ownerIndex]?.load(id);
  }

  /** Emit the portable JSON catalog other runtimes (Rust, CI, Go) read. */
  async toManifest(): Promise<SkillManifest> {
    const skills = await this.discover();
    return {
      version: SKILL_MANIFEST_VERSION,
      generatedAt: new Date().toISOString(),
      skills,
    };
  }

  /** Drop the discovery cache so the next call re-reads the sources. */
  refresh(): void {
    this.cache = undefined;
  }
}

/** Build a manifest from sources without holding a `Registry`. */
export async function buildManifest(
  sources: SkillSource | SkillSource[],
  options?: RegistryOptions,
): Promise<SkillManifest> {
  return new Registry(sources, options).toManifest();
}
