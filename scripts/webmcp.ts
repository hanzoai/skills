// WebMCP (https://webmachinelearning.github.io/webmcp/): two read-only tools over
// the skills index this site publishes, /.well-known/agent-skills/index.json.
// Registered only where the browser exposes document.modelContext (or the
// earlier navigator.modelContext). `npm run webmcp` writes public/webmcp.js.

interface Skill { name: string; type: string; description: string; url: string; digest: string }

interface Tool {
  name: string;
  title: string;
  description: string;
  inputSchema: object;
  annotations: { readOnlyHint: boolean };
  execute: (input: Record<string, unknown>) => Promise<unknown>;
}

interface ModelContext { registerTool(tool: Tool): Promise<void> | void }

const INDEX = '/.well-known/agent-skills/index.json';
let loaded: Promise<Skill[]> | undefined;

// The index, fetched once per page; a failed fetch is retried on the next call.
function skills(): Promise<Skill[]> {
  loaded ??= fetch(INDEX).then(async (r) => {
    if (!r.ok) throw new Error(`${INDEX}: HTTP ${r.status}`);
    return ((await r.json()) as { skills: Skill[] }).skills;
  });
  loaded.catch(() => { loaded = undefined; });
  return loaded;
}

const hex = (b: ArrayBuffer): string =>
  Array.from(new Uint8Array(b), (x) => x.toString(16).padStart(2, '0')).join('');

const search: Tool = {
  name: 'search_skills',
  title: 'Search Hanzo skills',
  description: 'Search the Hanzo skills published on hanzoskills.com by name and description. ' +
    'Every word must match. Returns the best matches first, each with its name, description and ' +
    'the URL of its Markdown; pass a name to get_skill to read one.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Words to match, e.g. "kai decisions" or "kubernetes deploy".' },
      limit: { type: 'integer', minimum: 1, maximum: 50, default: 10, description: 'Most results to return.' },
    },
    required: ['query'],
  },
  annotations: { readOnlyHint: true },
  async execute(input) {
    const words = String(input.query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) throw new Error('query is empty');
    const limit = Math.min(50, Math.max(1, Math.floor(Number(input.limit ?? 10)) || 10));
    const hits: { s: Skill; score: number }[] = [];
    for (const s of await skills()) {
      const name = s.name.toLowerCase();
      const desc = s.description.toLowerCase();
      let score = name === words.join('-') ? 10 : 0;
      for (const w of words) {
        if (name.includes(w)) score += 2;
        else if (desc.includes(w)) score += 1;
        else { score = -1; break; }
      }
      if (score >= 0) hits.push({ s, score });
    }
    hits.sort((a, b) => b.score - a.score || a.s.name.length - b.s.name.length || a.s.name.localeCompare(b.s.name));
    return {
      query: words.join(' '),
      total: hits.length,
      skills: hits.slice(0, limit).map(({ s }) => ({
        name: s.name, description: s.description, url: new URL(s.url, location.origin).href,
      })),
    };
  },
};

const get: Tool = {
  name: 'get_skill',
  title: 'Read a Hanzo skill',
  description: 'Return one Hanzo skill as Markdown (YAML front matter, then its instructions), ' +
    'verified against the SHA-256 digest in the skills index. Names come from search_skills.',
  inputSchema: {
    type: 'object',
    properties: { name: { type: 'string', description: 'The skill name, e.g. "kai" or "hanzo".' } },
    required: ['name'],
  },
  annotations: { readOnlyHint: true },
  async execute(input) {
    const name = String(input.name ?? '');
    const s = (await skills()).find((x) => x.name === name);
    if (!s) throw new Error(`no skill named "${name}"; search_skills lists the names`);
    const r = await fetch(s.url);
    if (!r.ok) throw new Error(`${s.url}: HTTP ${r.status}`);
    const bytes = await r.arrayBuffer();
    if ('sha256:' + hex(await crypto.subtle.digest('SHA-256', bytes)) !== s.digest) {
      throw new Error(`${s.url}: bytes do not match the index digest`);
    }
    return new TextDecoder().decode(bytes);
  },
};

const mc = (document as Document & { modelContext?: ModelContext }).modelContext
  ?? (navigator as Navigator & { modelContext?: ModelContext }).modelContext;
if (typeof mc?.registerTool === 'function') {
  for (const tool of [search, get]) {
    try {
      Promise.resolve(mc.registerTool(tool)).catch((e) => console.warn('WebMCP:', tool.name, e));
    } catch (e) {
      console.warn('WebMCP:', tool.name, e);
    }
  }
}
