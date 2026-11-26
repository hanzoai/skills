#!/usr/bin/env python3
"""3-tier progressive disclosure catalog generator for hanzoskills.com.

Emits:
  Tier 0 (tiny upfront indexes):
    - public/catalog.json          Compact category tree, counts, philosophy, discovery contract
    - public/llms.txt              Compact agent entrypoint (<5 KB) pointing to Tier 1 & Tier 2
    - public/index.md              Compact human/agent markdown homepage
  Tier 1 (on-demand category & lazy search indexes):
    - public/categories/<slug>.json  Per-category skill metadata (id, name, summary, tags, urls)
    - public/categories/<slug>.md    Per-category markdown index
    - public/search.json             Compact search index fetched lazily only when searching
  Tier 2 (on-demand full skill documents):
    - public/skills/<id>/SKILL.md    Canonical SKILL.md bytes (identical to /<id>.md)
    - public/skills/<id>.json        Structured skill metadata + full markdown content + sha256
"""
import glob
import hashlib
import html
import json
import os
import re

CATEGORIES = [
    ("hanzo", "Hanzo Platform & Cloud",
     "Core Hanzo stack, native Go/Rust/C++ services, Base, Cloud, CLI, MCP, and migration guides."),
    ("ai-agents", "AI, ML & Agent Engineering",
     "LLM APIs, agent workflows, MCP servers, skill authoring, inference, audio, vision, and research."),
    ("engineering", "Systems & Core Engineering",
     "Languages, distributed systems, debugging, testing, code review, protocols, and build tools."),
    ("cloud-infra", "Cloud, Containers & DevOps",
     "Kubernetes, containers, CI/CD, observability, edge deploys, and site architecture."),
    ("data-databases", "Data, Databases & Analytics",
     "SQL, vector search, data engineering, streaming pipelines, and product analytics."),
    ("security", "Security & Cryptography",
     "AppSec, threat modeling, OWASP guides, IAM, KMS, and applied cryptography."),
    ("frontend-design", "Frontend, UI & Design Systems",
     "React, CSS, design systems, Figma workflows, diagrams, mobile UI, and browser automation."),
    ("math-formal", "Mathematics & Formal Methods",
     "Category theory, algebra, proof assistants, formal verification, and optimization."),
    ("growth-product", "Product, Growth & Operations",
     "SEO, CRO, pricing, copywriting, GTM, revops, email, and team operations."),
    ("standards-core", "HIPs: Core AI & Models",
     "Hanzo Improvement Proposals for core architecture, governance, and AI model standards."),
    ("standards-infra", "HIPs: Cloud & Infrastructure",
     "Hanzo Improvement Proposals for compute, networking, storage, and cloud infrastructure."),
    ("standards-interface", "HIPs: Interfaces & Protocols",
     "Hanzo Improvement Proposals for ZAP, MCP, SDKs, APIs, and realtime protocols."),
    ("standards-security", "HIPs: Security, IAM & Crypto",
     "Hanzo Improvement Proposals for identity, KMS, zero-trust, and post-quantum security."),
    ("standards-apps", "HIPs: Platform & Applications",
     "Hanzo Improvement Proposals for developer apps, agent surfaces, and ecosystem products."),
]

CAT_META = {slug: (label, summary) for slug, label, summary in CATEGORIES}
CAT_ORDER = {slug: idx for idx, (slug, _, _) in enumerate(CATEGORIES)}

RAW_TO_CAT = {
    "hanzo": "hanzo",
    "hanzo-api": "hanzo",
    "hanzo-ml-research": "hanzo",
    "ui-port": "hanzo",
    "zoo-gym": "hanzo",
    "bot": "hanzo",

    "ml": "ai-agents",
    "claude-api": "ai-agents",
    "mcp-builder": "ai-agents",
    "skill-creator": "ai-agents",
    "plugin-creator": "ai-agents",
    "cli-creator": "ai-agents",
    "model-usage": "ai-agents",
    "speech": "ai-agents",
    "imagegen": "ai-agents",
    "image": "ai-agents",
    "video": "ai-agents",
    "transcribe": "ai-agents",
    "jupyter-notebook": "ai-agents",
    "research": "ai-agents",
    "modal": "ai-agents",
    "workflow": "ai-agents",
    "ir": "ai-agents",
    "anti-slop": "ai-agents",

    "core": "engineering",
    "engineering": "engineering",
    "distributed-systems": "engineering",
    "debugging": "engineering",
    "testing": "engineering",
    "code-review": "engineering",
    "build-systems": "engineering",
    "zig": "engineering",
    "wasm": "engineering",
    "plt": "engineering",
    "ebpf": "engineering",
    "protocols": "engineering",
    "networking": "engineering",
    "realtime": "engineering",
    "typed-holes-refactor": "engineering",
    "tui": "engineering",
    "api": "engineering",
    "caching": "engineering",
    "proxies": "engineering",
    "winui-app": "engineering",
    "aspnet-core": "engineering",

    "cloud": "cloud-infra",
    "infrastructure": "cloud-infra",
    "containers": "cloud-infra",
    "deployment": "cloud-infra",
    "cicd": "cloud-infra",
    "observability": "cloud-infra",
    "cloudflare-deploy": "cloud-infra",
    "render-deploy": "cloud-infra",
    "netlify-deploy": "cloud-infra",
    "site-architecture": "cloud-infra",

    "data": "data-databases",
    "database": "data-databases",
    "analytics": "data-databases",

    "security": "security",
    "security-best-practices": "security",
    "security-threat-model": "security",
    "security-ownership-map": "security",
    "cryptography": "security",
    "1password": "security",

    "frontend": "frontend-design",
    "mobile": "frontend-design",
    "elegant-design": "frontend-design",
    "theme-factory": "frontend-design",
    "figma-use": "frontend-design",
    "figma-generate-library": "frontend-design",
    "figma-code-connect-components": "frontend-design",
    "figma-create-design-system-rules": "frontend-design",
    "diagrams": "frontend-design",
    "diagram-maker": "frontend-design",
    "control-ui": "frontend-design",
    "playwright": "frontend-design",

    "math": "math-formal",
    "formal": "math-formal",

    "product": "growth-product",
    "marketing-plan": "growth-product",
    "marketing-ideas": "growth-product",
    "ads": "growth-product",
    "ad-creative": "growth-product",
    "ai-seo": "growth-product",
    "seo-audit": "growth-product",
    "programmatic-seo": "growth-product",
    "schema-markup": "growth-product",
    "aso": "growth-product",
    "cro": "growth-product",
    "page-cro": "growth-product",
    "ab-test-setup": "growth-product",
    "offers": "growth-product",
    "pricing": "growth-product",
    "churn-prevention": "growth-product",
    "referral-program": "growth-product",
    "cold-email": "growth-product",
    "emails": "growth-product",
    "copy-editing": "growth-product",
    "content-strategy": "growth-product",
    "competitors": "growth-product",
    "competitor-profiling": "growth-product",
    "customer-research": "growth-product",
    "directory-submissions": "growth-product",
    "free-tool-strategy": "growth-product",
    "lead-magnets": "growth-product",
    "prospecting": "growth-product",
    "public-relations": "growth-product",
    "revops": "growth-product",
    "sales-enablement": "growth-product",
    "social": "growth-product",
    "sms": "growth-product",
    "collaboration": "growth-product",
    "internal-comms": "growth-product",
    "himalaya": "growth-product",
}

FEATURED_IDS = [
    "skill",
    "hanzo",
    "hanzo-base",
    "hanzo-cloud",
    "hanzo-mcp",
    "app-port",
    "cloud-host",
    "ui-port",
]

STOP_TOKENS = {
    "a", "an", "and", "the", "for", "with", "from", "to", "of", "in", "on", "by",
    "guide", "skill", "general", "use", "using", "web", "app", "hanzo", "hip",
}

BAD_DESC = re.compile(
    r"^(go |npm |bd |curl |cargo |pip |brew |git |docker )|[:,]$|^Skill for ", re.I
)


def write_if_changed(path, text):
    data = text.encode("utf-8") if isinstance(text, str) else text
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        try:
            if open(path, "rb").read() == data:
                return False
        except OSError:
            pass
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return True


def parse_frontmatter(text):
    """Best-effort frontmatter parser returning (fields_dict, tags_list, body_str)."""
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, [], text
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", text, re.S)
    if not m:
        return {}, [], text
    fm, body = m.group(1), text[m.end():]
    fields = {}
    tags = []
    lines = fm.splitlines()
    i = 0
    while i < len(lines):
        lm = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", lines[i])
        i += 1
        if not lm:
            continue
        key, val = lm.group(1).lower(), lm.group(2).strip()
        if key == "tags":
            if val.startswith("[") and val.endswith("]"):
                for part in val[1:-1].split(","):
                    t = part.strip().strip("\"'").lower()
                    if t:
                        tags.append(t)
            else:
                while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                    t = re.sub(r"^\s*-\s+", "", lines[i]).strip().strip("\"'").lower()
                    if t:
                        tags.append(t)
                    i += 1
            continue
        if val in ("|", ">", "|-", ">-", "|+", ">+"):
            buf = []
            while i < len(lines) and (lines[i][:1] in (" ", "\t") or lines[i] == ""):
                buf.append(lines[i].strip())
                i += 1
            fields[key] = " ".join(x for x in buf if x).strip()
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        fields[key] = val.strip()
    return fields, tags, body


def first_sentence(body, limit=220):
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or p.startswith(("#", "|", "```", "---", "<!--", "*Note", ">", "**Category", "**Related")):
            continue
        p = re.sub(r"[*_`\[\]]", "", " ".join(p.split()))
        m = re.match(r"(.{20,%d}?[.!?])(\s|$)" % limit, p)
        return (m.group(1) if m else p[:limit]).strip()
    return ""


def map_hip_category(cat, num_str):
    c = (cat or "").lower()
    num = int(num_str) if num_str.isdigit() else 9999
    if c in ("core", "meta", "general", "governance") or num in (0, 1, 2, 3, 6, 7, 9, 39):
        if num in (4, 10, 11, 12, 15, 38):
            return "standards-interface"
        return "standards-core"
    if c in ("security", "cryptography") or num in (5, 26, 27, 54, 84, 85, 86, 87, 88, 111, 118, 143):
        return "standards-security"
    if c in ("infrastructure", "bridge") or num in (14, 20, 24, 28, 29, 30, 33, 36, 101, 102, 106, 114, 117):
        return "standards-infra"
    if c in ("interface",):
        return "standards-interface"
    if c in ("application", "operator", "platform") or num >= 400:
        return "standards-apps"
    return "standards-infra"


def build_source_lookup(source_dir):
    """Map skill id -> (raw_cat, tags) from skills/ source tree in deterministic order."""
    lookup = {}
    if not os.path.isdir(source_dir):
        return lookup
    entries = []
    for dp, dn, fn in os.walk(source_dir):
        dn.sort()
        for f in sorted(fn):
            if not f.endswith(".md") or f in ("INDEX.md", "README.md", "SECURITY.md", "_SKILL_TEMPLATE.md"):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, source_dir).replace(os.sep, "/")
            # Prefer canonical SKILL.md definitions over reference docs, then lexicographical POSIX path.
            entries.append((0 if f == "SKILL.md" else 1, rel, dp, f, full))
    entries.sort(key=lambda item: (item[0], item[1]))
    for _, _, dp, f, p in entries:
        try:
            text = open(p, encoding="utf-8").read()
        except OSError:
            continue
        fm, tags, _ = parse_frontmatter(text)
        if fm.get("name"):
            sid = re.sub(r"[^a-z0-9._-]+", "-", fm["name"].lower()).strip("-")
        else:
            sid = f[:-3].lower()
        rel_parts = os.path.relpath(dp, source_dir).split(os.sep)
        raw_cat = rel_parts[0] if rel_parts and rel_parts[0] != "." else "core"
        if raw_cat == sid:
            raw_cat = "core"
        lookup.setdefault(sid, (raw_cat, tags))
    return lookup


def classify(sid, fm, raw_cat):
    if sid.startswith("hip-"):
        n = sid[4:].lstrip("0") or "0"
        return map_hip_category(fm.get("category", ""), n)
    if sid == "skill" or sid.startswith("hanzo") or sid in ("app-port", "cloud-host", "ui-port", "kai"):
        return "hanzo"
    if raw_cat in RAW_TO_CAT:
        return RAW_TO_CAT[raw_cat]
    if "security" in sid or "owasp" in sid or "crypto" in sid or sid == "1password":
        return "security"
    if "figma" in sid or "css" in sid or "react" in sid or "design" in sid or "ui-" in sid:
        return "frontend-design"
    if "sql" in sid or "postgres" in sid or "sqlite" in sid or "database" in sid:
        return "data-databases"
    if "k8s" in sid or "kubernetes" in sid or "docker" in sid or "deploy" in sid or "terraform" in sid:
        return "cloud-infra"
    if "seo" in sid or "marketing" in sid or "sales" in sid or "pricing" in sid:
        return "growth-product"
    if "math" in sid or "proof" in sid or "lean" in sid or "coq" in sid:
        return "math-formal"
    if "llm" in sid or "agent" in sid or "mcp" in sid or "claude" in sid or "ml-" in sid:
        return "ai-agents"
    return "engineering"


def derive_tags(sid, cat, raw_cat, fm_tags):
    seen = set()
    out = []

    def add(tok):
        t = re.sub(r"[^a-z0-9+-]+", "-", str(tok).lower()).strip("-")
        if not t or len(t) < 2 or t in seen:
            return
        seen.add(t)
        out.append(t)

    add(cat)
    if raw_cat and raw_cat != cat and raw_cat != "core":
        add(raw_cat)
    for t in fm_tags:
        add(t)
    if sid.startswith("hip-"):
        add("hip")
        add("standard")
    for part in sid.split("-"):
        if len(out) >= 6:
            break
        if part not in STOP_TOKENS and not part.isdigit():
            add(part)
    return out[:6]


def summarize(sid, fm, body):
    if sid.startswith("hip-") and fm.get("title"):
        n = str(fm.get("hip", sid[4:])).lstrip("0") or "0"
        status = fm.get("status", "")
        desc = "HIP-%s: %s." % (n, fm["title"].rstrip("."))
        if status:
            desc += " Status %s." % status
        return desc[:220]
    desc = " ".join((fm.get("description") or "").split())
    if desc and not BAD_DESC.search(desc) and len(desc) > 20:
        return (desc[:217] + "...") if len(desc) > 220 else desc
    fs = first_sentence(body, 220)
    return fs or fm.get("title") or sid


def collect_published(root, source_dir):
    lookup = build_source_lookup(source_dir)
    docs = []
    skip = {"index", "all", "auth", "CLAUDE", "AGENTS"}
    for path in sorted(glob.glob(os.path.join(root, "*.md")), key=lambda p: os.path.basename(p)):
        sid = os.path.basename(path)[:-3]
        if sid in skip:
            continue
        raw = open(path, "rb").read()
        text = raw.decode("utf-8", errors="replace")
        fm, fm_tags, body = parse_frontmatter(text)
        raw_cat, src_tags = lookup.get(sid, ("", []))
        cat = classify(sid, fm, raw_cat)
        tags = derive_tags(sid, cat, raw_cat, fm_tags or src_tags)
        summary = summarize(sid, fm, body)
        docs.append({
            "id": sid,
            "name": sid,
            "category": cat,
            "category_label": CAT_META[cat][0],
            "summary": summary,
            "tags": tags,
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "url": "/skills/%s/SKILL.md" % sid,
            "json": "/skills/%s.json" % sid,
            "md": "/%s.md" % sid,
            "html": "/%s" % sid,
            "raw": raw,
            "text": text,
        })
    docs.sort(key=lambda d: (CAT_ORDER.get(d["category"], 99), d["id"]))
    return docs


def emit_progressive(root, source_dir, get_md_text):
    """Emit all Tier 0 / Tier 1 / Tier 2 progressive disclosure artifacts into root."""
    docs = collect_published(root, source_dir)
    by_id = {d["id"]: d for d in docs}
    by_cat = {slug: [] for slug, _, _ in CATEGORIES}
    for d in docs:
        by_cat.setdefault(d["category"], []).append(d)

    # Tier 2: /skills/<id>/SKILL.md and /skills/<id>.json
    skills_dir = os.path.join(root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    for d in docs:
        write_if_changed(os.path.join(skills_dir, d["id"], "SKILL.md"), d["raw"])
        skill_json = {
            "id": d["id"],
            "name": d["name"],
            "category": d["category"],
            "category_label": d["category_label"],
            "summary": d["summary"],
            "tags": d["tags"],
            "sha256": d["sha256"],
            "bytes": d["bytes"],
            "url": d["url"],
            "md": d["md"],
            "html": d["html"],
            "content": d["text"],
        }
        write_if_changed(
            os.path.join(skills_dir, d["id"] + ".json"),
            json.dumps(skill_json, ensure_ascii=False, separators=(",", ":")) + "\n",
        )

    # Tier 1: /categories/<slug>.json and /categories/<slug>.md
    cat_dir = os.path.join(root, "categories")
    os.makedirs(cat_dir, exist_ok=True)
    active_cat_files = set()
    cat_summaries = []
    for slug, label, cat_desc in CATEGORIES:
        items = sorted(by_cat.get(slug, []), key=lambda x: x["id"])
        if not items:
            continue
        active_cat_files.add(slug + ".json")
        active_cat_files.add(slug + ".md")

        # Prefer featured/clean alphabetic canonical ids for sample chips
        sample = []
        if slug == "hanzo":
            for fid in FEATURED_IDS:
                if fid in by_id and by_id[fid]["category"] == slug:
                    sample.append(fid)
        for item in items:
            if len(sample) >= 5:
                break
            sid = item["id"]
            if sid in sample or not re.match(r"^[a-z][a-z0-9-]+$", sid):
                continue
            sample.append(sid)
        for item in items:
            if len(sample) >= 5:
                break
            if item["id"] not in sample:
                sample.append(item["id"])

        cat_payload = {
            "slug": slug,
            "label": label,
            "summary": cat_desc,
            "count": len(items),
            "catalog": "/catalog.json",
            "md": "/categories/%s.md" % slug,
            "skills": [
                {
                    "id": x["id"],
                    "name": x["name"],
                    "category": x["category"],
                    "summary": x["summary"],
                    "tags": x["tags"],
                    "url": x["url"],
                    "json": x["json"],
                    "md": x["md"],
                    "html": x["html"],
                }
                for x in items
            ],
        }
        write_if_changed(
            os.path.join(cat_dir, slug + ".json"),
            json.dumps(cat_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        )

        md_lines = [
            "# %s (%d skills)\n\n" % (label, len(items)),
            "> %s\n" % cat_desc,
            "> Tier 0 catalog: [llms.txt](https://hanzoskills.com/llms.txt) · "
            "[catalog.json](https://hanzoskills.com/catalog.json) · "
            "[JSON](https://hanzoskills.com/categories/%s.json)\n\n" % slug,
            "Fetch a single skill on demand: `curl -fsSL https://hanzoskills.com/skills/<id>/SKILL.md`\n\n",
        ]
        for x in items:
            md_lines.append("- [%s](https://hanzoskills.com/skills/%s/SKILL.md) — %s\n" % (x["id"], x["id"], x["summary"]))
        write_if_changed(os.path.join(cat_dir, slug + ".md"), "".join(md_lines))

        cat_summaries.append({
            "slug": slug,
            "label": label,
            "summary": cat_desc,
            "count": len(items),
            "json": "/categories/%s.json" % slug,
            "md": "/categories/%s.md" % slug,
            "sample": sample,
        })

    # Prune stale files in categories/
    for existing in sorted(os.listdir(cat_dir)):
        if existing not in active_cat_files:
            os.remove(os.path.join(cat_dir, existing))

    # Tier 1 lazy search index (loaded ONLY when user types in #q)
    search_payload = {
        "v": 1,
        "t": len(docs),
        "s": [
            {
                "i": d["id"],
                "c": d["category"],
                "d": d["summary"],
                "g": d["tags"],
            }
            for d in docs
        ],
    }
    write_if_changed(
        os.path.join(root, "search.json"),
        json.dumps(search_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
    )

    # Tier 0: /catalog.json
    featured = []
    for fid in FEATURED_IDS:
        d = by_id.get(fid)
        if d:
            featured.append({
                "id": d["id"],
                "category": d["category"],
                "summary": d["summary"],
                "url": d["url"],
                "json": d["json"],
                "html": d["html"],
            })

    catalog_data = {
        "version": "1.0.0",
        "total": len(docs),
        "philosophy": (
            "Hanzo is an open-source, local-first AI cloud built on native compiled code "
            "(Go, Rust, C++). Agents can build, customize, and mod any part of Hanzo "
            "(apps, mods, extensions, skills, MCP servers) and load skills progressively on demand."
        ),
        "discovery": {
            "tier0": {
                "claude_md": "https://hanzoskills.com/CLAUDE.md",
                "agents_md": "https://hanzoskills.com/AGENTS.md",
                "llms": "https://hanzoskills.com/llms.txt",
                "catalog": "https://hanzoskills.com/catalog.json",
                "manifest": "https://hanzoskills.com/skill.md",
                "hips": "https://hips.hanzo.ai/llms.txt",
                "mcp": "https://hanzoskills.com/mcp/index.json",
                "oss": "https://hanzoskills.com/oss/index.json",
                "oss_md": "https://hanzoskills.com/oss/index.md",
                "forge": "https://forge.hanzo.ai",
                "docs": "https://docs.hanzo.ai/docs/skills",
            },
            "tier1": {
                "category_json": "https://hanzoskills.com/categories/{slug}.json",
                "category_md": "https://hanzoskills.com/categories/{slug}.md",
                "search_index": "https://hanzoskills.com/search.json",
                "detect_api": "POST https://api.hanzo.ai/v1/tool/detect",
            },
            "tier2": {
                "skill_md": "https://hanzoskills.com/skills/{id}/SKILL.md",
                "skill_json": "https://hanzoskills.com/skills/{id}.json",
                "flat_md": "https://hanzoskills.com/{id}.md",
                "html": "https://hanzoskills.com/{id}",
            },
            "commands": [
                "/skill",
                "curl -fsSL https://hanzoskills.com/llms.txt",
                "POST https://api.hanzo.ai/v1/tool/detect",
            ],
        },
        "featured": featured,
        "categories": cat_summaries,
    }
    write_if_changed(
        os.path.join(root, "catalog.json"),
        json.dumps(catalog_data, ensure_ascii=False, indent=2) + "\n",
    )

    # Tier 0: compact /llms.txt (~3.5 KB — ~98.8% smaller than dumping 1,471 rows!)
    llms_lines = [
        "# Hanzo Skills\n\n",
        "> %d skills and HIP specifications across %d categories for building, customizing, and modding Hanzo apps, backends, agents, MCP servers, and cloud infrastructure.\n" % (len(docs), len(cat_summaries)),
        "> Open-source, local-first, native compiled code (Go, Rust, C++). Plain markdown over HTTPS, zero auth.\n\n",
        "## Progressive Discovery (3 Tiers — Do Not Load Everything Up Front)\n\n",
        "1. **Tier 0 (this file)**: Pick a category below or read `GET https://hanzoskills.com/catalog.json`.\n",
        "2. **Tier 1 (category / ranking)**: Fetch ONLY the matching category `GET https://hanzoskills.com/categories/<slug>.md` (or `.json`), invoke `/skill`, or call `POST https://api.hanzo.ai/v1/tool/detect` to rank skills for a prompt.\n",
        "3. **Tier 2 (single skill)**: Fetch ONLY the skill you need via `GET https://hanzoskills.com/skills/<id>/SKILL.md` (or `/skills/<id>.json` or `/<id>.md`).\n\n",
        "## Hanzo Philosophy & Ecosystem\n\n",
        "- [Canonical CLAUDE.md](https://hanzoskills.com/CLAUDE.md): drop-in progressive instructions for Claude Code, Codex (`AGENTS.md`), Cursor, Windsurf, and any agent (`curl -fsSL https://hanzoskills.com/CLAUDE.md >> CLAUDE.md`).\n",
        "- [Onboarding Manifest](https://hanzoskills.com/skill.md): what runs locally, commands to start each part, and hosted cloud boundaries.\n",
        "- [Install Hanzo](https://hanzo.sh): `curl -fsSL https://hanzo.sh | sh` installs the CLI, MCP server, and dev agent.\n",
        "- [HIP Specifications](https://hips.hanzo.ai/llms.txt): Hanzo Improvement Proposals progressive index.\n",
        "- [MCP Progressive Index](https://hanzoskills.com/mcp/index.json): 13 unified HIP-0300 tools (<1.5k tokens).\n",
        "- [Forge Open Source Repositories](https://hanzoskills.com/oss/index.md) ([JSON](https://hanzoskills.com/oss/index.json)): progressive index of all public OSS repos on https://forge.hanzo.ai.\n",
        "- [Documentation & Skills Guide](https://docs.hanzo.ai/docs/skills) ([API llms.txt](https://docs.hanzo.ai/llms.txt)): docs.hanzo.ai guides and api.hanzo.ai operations.\n",
        "- [Hanzo AI](https://hanzo.ai) ([llms.txt](https://hanzo.ai/llms.txt)): the AI operating system for business.\n\n",
        "## Start Here\n\n",
    ]
    for f in featured[:6]:
        llms_lines.append("- [%s](https://hanzoskills.com/skills/%s/SKILL.md): %s\n" % (f["id"], f["id"], f["summary"]))

    llms_lines.append("\n## Categories (Tier 1 Indexes)\n\n")
    for c in cat_summaries:
        samples = ", ".join(c["sample"][:4])
        llms_lines.append(
            "- [%s](https://hanzoskills.com/categories/%s.md) (%d skills · [JSON](https://hanzoskills.com/categories/%s.json)): %s Samples: %s.\n"
            % (c["label"], c["slug"], c["count"], c["slug"], c["summary"], samples)
        )

    llms_lines.extend([
        "\n## Machine Discovery & Optional Corpus\n\n",
        "- [Tier 0 Catalog JSON](https://hanzoskills.com/catalog.json)\n",
        "- [Agent Skills Discovery 0.2.0](https://hanzoskills.com/.well-known/agent-skills/index.json)\n",
        "- [MCP Server Card](https://hanzoskills.com/.well-known/mcp/server-card.json)\n",
        "- [Auth Guide](https://hanzoskills.com/auth.md)\n",
        "- [Full Offline Corpus (16 MB — avoid unless indexing offline)](https://hanzoskills.com/all.md)\n",
    ])
    write_if_changed(os.path.join(root, "llms.txt"), "".join(llms_lines))

    # Canonical progressive-disclosure CLAUDE.md and AGENTS.md
    claude_md = (
        "# Hanzo — Complete Progressive Agent Instructions (`CLAUDE.md`)\n\n"
        "> Canonical drop-in `CLAUDE.md` / `AGENTS.md` for Claude Code, Codex, Cursor, Windsurf, Gemini CLI, Antigravity, Copilot, Hanzo Dev, Hanzo Bot, or any AI agent.\n"
        "> Live at: https://hanzoskills.com/CLAUDE.md · https://hanzo.ai/CLAUDE.md · Docs: https://docs.hanzo.ai/docs/skills\n\n"
        "## 1. Progressive Skill Loading (3 Tiers — Never Load Everything Up Front)\n\n"
        "Hanzo publishes **%d plain-markdown skills** across **%d categories** over HTTPS with zero auth.\n"
        "Keep context small by loading skills **only when a task requires them**:\n\n"
        "1. **Tier 0 (Category Tree, ~8 KB)**: `curl -fsSL https://hanzoskills.com/llms.txt` (or `/catalog.json`).\n"
        "2. **Tier 1 (Single Category)**: Fetch ONLY the matching category `https://hanzoskills.com/categories/<slug>.md` (or rank via `POST https://api.hanzo.ai/v1/tool/detect`).\n"
        "3. **Tier 2 (Single Skill)**: Fetch ONLY the skill you need via `https://hanzoskills.com/skills/<id>/SKILL.md` (or `/<id>.md`).\n"
        "   - Never fetch `/all.md` (16 MB) into an active agent context.\n\n"
        "## 2. Native Agent Setup (Use Each Agent's Native Instructions, Skills & MCP)\n\n"
        "| Agent | Native Instructions File | Native Skills Path | Native MCP Setup |\n"
        "| --- | --- | --- | --- |\n"
        "| **Claude Code** | `./CLAUDE.md` & `~/.claude/CLAUDE.md` | `~/.claude/skills/<id>/SKILL.md` | `claude mcp add --scope user hanzo -- npx -y hanzo mcp` |\n"
        "| **Codex** | `./AGENTS.md` & `~/.codex/AGENTS.md` | `~/.codex/skills/<id>/SKILL.md` | `codex mcp add hanzo -- npx -y hanzo mcp` |\n"
        "| **Cursor** | `.cursor/rules/hanzo.mdc` (`alwaysApply: true`) | `.cursor/rules/<id>.mdc` (`alwaysApply: false`) | `.cursor/mcp.json` (`hanzo` -> `npx -y hanzo mcp`) |\n"
        "| **Windsurf** | `.windsurfrules` / `.windsurf/rules/hanzo.md` | `.windsurf/rules/<id>.md` | `~/.codeium/windsurf/mcp_config.json` |\n"
        "| **Gemini CLI / Antigravity** | `./GEMINI.md` & `~/.gemini/GEMINI.md` | `~/.gemini/antigravity/skills/<id>/SKILL.md` | `~/.gemini/settings.json` (`mcpServers.hanzo`) |\n"
        "| **GitHub Copilot** | `.github/copilot-instructions.md` | `.github/instructions/<id>.instructions.md` | `.vscode/mcp.json` |\n"
        "| **Hanzo Dev** | Native (`AGENTS.md` / `CLAUDE.md`) | `<repo>/.agents/skills/<id>/SKILL.md` or `~/.agents/skills/` | Built-in native MCP + ZAP |\n"
        "| **Hanzo Bot** | Native (`hanzo.bot`) | `<workspace>/skills/` or `~/.hanzo/bot/skills/<id>/SKILL.md` | Built-in native MCP + browser sandbox |\n\n"
        "## 3. Hanzo MCP (Unified Tool Surface — Local Stdio & Hosted HTTP)\n\n"
        "- **Local stdio (recommended, zero pre-install)**: `npx -y hanzo mcp` (or `hanzo mcp` after installing the CLI).\n"
        "  Exposes the **13 HIP-0300 unified tools** (`fs`, `exec`, `code`, `git`, `fetch`, `workspace`, `computer`, `think`, `memory`, `hanzo`, `plan`, `tasks`, `ui`) plus your linked cloud connectors (GitHub, Slack, Google, Discord, X).\n"
        "- **JSON MCP config** (Cursor / Claude Desktop / Windsurf / Gemini CLI / Antigravity):\n"
        "  `{\"mcpServers\":{\"hanzo\":{\"command\":\"npx\",\"args\":[\"-y\",\"hanzo\",\"mcp\"]}}}`\n"
        "- **Hosted Cloud MCP (Streamable HTTP, zero local setup)**: `https://api.hanzo.ai/v1/mcp` (`Authorization: Bearer $HANZO_API_KEY`).\n"
        "- **MCP schemas (progressive)**: `https://hanzoskills.com/mcp/index.json` -> `/mcp/<tool>.json` · Docs: https://docs.hanzo.ai/docs/mcp\n\n"
        "## 4. Install the Hanzo Browser Extension (Chrome, Firefox, Safari, Edge)\n\n"
        "Install the Browser Extension so agents get live tab inspection, CDP, DOM snapshots, click/type, screenshots, and social automation bridged over ZAP/MCP:\n\n"
        "- **Download & unpack Chrome/Chromium extension**:\n"
        "  ```bash\n"
        "  mkdir -p ~/.hanzo/browser-extension && cd ~/.hanzo/browser-extension\n"
        "  URL=$(curl -fsSL https://api.github.com/repos/hanzoai/extension/releases/latest | grep -oE 'https://[^\"]*hanzo-ai-chrome[^\"]*\\.zip' | head -1)\n"
        "  curl -fsSL \"$URL\" -o ext.zip && unzip -oq ext.zip && rm ext.zip\n"
        "  ```\n"
        "  Then open `chrome://extensions` -> enable **Developer mode** -> **Load unpacked** -> select `~/.hanzo/browser-extension` -> sign in via https://hanzo.id.\n"
        "- **All browsers & IDE plugins** (Chrome, Firefox, Safari, Edge, VS Code, Cursor, Windsurf, Antigravity, JetBrains): https://hanzo.ai/extension · https://hanzo.ai/chrome · https://hanzo.ai/download\n\n"
        "## 5. Hanzo Desktop, CLI, Hanzo Dev, Hanzo Bot & Live Usage Visibility\n\n"
        "- **Install native CLI + Dev + MCP + Engine**: `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo`), then `hanzo login` (or headless wallet signup via https://hanzoskills.com/hanzo.md).\n"
        "- **Install Hanzo Desktop** (macOS/Windows/Linux): https://hanzo.ai/download (or https://hanzo.ai/desktop) and sign in with the same Hanzo ID to see every local & cloud agent, live session, bot run, and spend in one control plane.\n"
        "- **Launch any agent through Hanzo CLI so sessions, tools & usage show up everywhere**:\n"
        "  `hanzo dev` · `hanzo --claude` · `hanzo --codex` · `hanzo --cursor` · `hanzo --agy` · `hanzo --gemini` (add `--cloud` to route the agent's model calls on your Hanzo plan).\n"
        "- **Check live usage, plan & balance**: `hanzo usage summary` · `hanzo billing tier` · `hanzo billing balance` · `hanzo billing topup` (or `GET /v1/billing/tier` & `GET /v1/allowance`).\n"
        "- **Hanzo Bot (resident autonomous agents — Slack, Discord, Telegram, WhatsApp, iMessage + browser/terminal/cron)**:\n"
        "  Cloud: https://hanzo.bot (`hanzo bot runs list`) · Local OSS daemon: `npm i -g @hanzo/bot && hanzo-bot onboard --install-daemon` · Docs: https://docs.hanzo.bot\n\n"
        "## 6. Full Platform Fluency & Business AI Operating System (OSS Business Apps)\n\n"
        "Every service shares one API (`https://api.hanzo.ai/v1`), one key (`export HANZO_API_KEY=...`), and one deploy contract (`hanzo deploy .` or self-host):\n\n"
        "- **Zero-auth API discovery**: `GET https://api.hanzo.ai/v1`, `GET /v1/<capability>`, `GET /v1/openapi.json`.\n"
        "- **AI & Core**: `/v1/chat/completions` (`enso-auto`, `zen-free`, `zen-*`, 500+ models), **Kai** calibrated decisions (`POST /v1/decisions`), IAM (`hanzo.id`), KMS (`hanzo kms put`), Base (`/v1/base`), SQL, KV, Vector, S3, PubSub, Tasks, Sandboxes, Billing/USDC.\n"
        "- **Deploy OSS Business Apps for any company** (org setup guide: https://docs.hanzo.ai/docs/run-your-org):\n"
        "  - **Web Analytics** (`forge.hanzo.ai/hanzoai/analytics` · skill `/skills/hanzo-analytics/SKILL.md`): privacy-first, cookieless web & revenue analytics.\n"
        "  - **Product Insights** (`forge.hanzo.ai/hanzoai/insights` · skill `/skills/hanzo-insights/SKILL.md`): funnels, session replay, feature flags, A/B experiments.\n"
        "  - **Marketing, Social & Ads** (`hanzoai/post`, `hanzoai/ads` · skill `/skills/hanzo-post/SKILL.md` + `/categories/growth-product.md`: SEO, CRO, copy, campaigns).\n"
        "  - **Team, Commerce, Cal, Mail, Workflows & Studio**: `hanzoai/team`, `hanzoai/commerce`, `hanzoai/billing`, `hanzoai/cal`, `hanzoai/mail`, `hanzoai/flow`, `hanzoai/studio`, `hanzoai/o11y`.\n\n"
        "## 7. Forge Open Source Repositories (2,832+ Public Repos — Progressive Discovery)\n\n"
        "1. **Tier 0 (Org Directory)**: `curl -fsSL https://hanzoskills.com/oss/index.md` (or `/oss/index.json`).\n"
        "2. **Tier 1 (Single Org)**: `curl -fsSL https://hanzoskills.com/oss/<org>.md` (e.g. `hanzoai`, `hanzoskill`, `luxfi`, `zooai`, `zenlm`, `zap-proto`).\n"
        "3. **Tier 2 (Single Repo)**: `git clone https://forge.hanzo.ai/<org>/<repo>.git` (every public repo serves `/<org>/<repo>/CLAUDE.md`).\n"
        "4. **Local-first runtime & HIPs**: https://hanzoskills.com/skill.md · https://hips.hanzo.ai/llms.txt\n"
    ) % (len(docs), len(cat_summaries))
    write_if_changed(os.path.join(root, "CLAUDE.md"), claude_md)
    write_if_changed(os.path.join(root, "AGENTS.md"), claude_md)

    # Tier 0: compact /index.md
    idx_lines = [
        "# Hanzo Skills\n\n",
        "Native compiled cloud skills (%d skills across %d categories). Progressive disclosure by default — load only what your task needs.\n\n" % (len(docs), len(cat_summaries)),
        get_md_text,
        "    curl -fsSL https://hanzoskills.com/llms.txt                  # Tier 0: compact category tree\n",
        "    curl -fsSL https://hanzoskills.com/categories/hanzo.md       # Tier 1: one category\n",
        "    curl -fsSL https://hanzoskills.com/skills/hanzo/SKILL.md     # Tier 2: one skill\n",
        "    curl -fsSL https://hips.hanzo.ai/llms.txt                    # HIP architectural specs\n\n",
        "## Start Here\n\n",
    ]
    for f in featured[:6]:
        idx_lines.append("- [`%s`](skills/%s/SKILL.md) — %s\n" % (f["id"], f["id"], f["summary"]))
    idx_lines.append("\n## Categories\n\n")
    for c in cat_summaries:
        idx_lines.append(
            "- [%s](categories/%s.md) (%d skills · [JSON](categories/%s.json)) — %s\n"
            % (c["label"], c["slug"], c["count"], c["slug"], c["summary"])
        )
    write_if_changed(os.path.join(root, "index.md"), "".join(idx_lines))

    return catalog_data


def render_index_body(catalog_data):
    """Render the compact Tier 0 interactive landing page body (zero eager skill rows)."""
    total = catalog_data["total"]
    cats = catalog_data["categories"]
    cards = []
    for c in cats:
        chips = "".join(
            '<span class="chip">%s</span>' % html.escape(s) for s in c["sample"][:4]
        )
        cards.append(
            '<button class="catcard" type="button" data-cat="%s" aria-expanded="false">'
            '<div class="cathead"><span class="catname">%s</span>'
            '<span class="catcnt">%d</span></div>'
            '<p class="catdesc">%s</p>'
            '<div class="chips">%s</div>'
            '<div class="catfoot"><span>Open category</span>'
            '<span class="catfmt">json · md</span></div>'
            '</button>'
            % (
                html.escape(c["slug"]),
                html.escape(c["label"]),
                c["count"],
                html.escape(c["summary"]),
                chips,
            )
        )

    tiers = (
        '<section class="tiers" aria-label="Progressive disclosure">'
        '<div class="tier">'
        '<div class="tmeta"><b>Tier 0 · Index</b><span>~4 KB</span></div>'
        '<h3>Category Tree</h3>'
        '<p>Load the compact overview first — zero context bloat.</p>'
        '<div class="tlinks">'
        '<a href="/CLAUDE.md">/CLAUDE.md</a>'
        '<a href="/llms.txt">/llms.txt</a>'
        '<a href="/catalog.json">/catalog.json</a>'
        '<a href="/oss/index.md">/oss/index.md</a>'
        '</div></div>'
        '<div class="tier">'
        '<div class="tmeta"><b>Tier 1 · Category</b><span>On demand</span></div>'
        '<h3>Filtered Metadata</h3>'
        '<p>Fetch one category or rank matches with <code>POST /v1/tool/detect</code> or <code>/skill</code>.</p>'
        '<div class="tlinks">'
        '<a href="/categories/hanzo.json">/categories/&lt;slug&gt;.json</a>'
        '<a href="/categories/hanzo.md">/categories/&lt;slug&gt;.md</a>'
        '</div></div>'
        '<div class="tier">'
        '<div class="tmeta"><b>Tier 2 · Skill</b><span>Lazy fetch</span></div>'
        '<h3>Full Instructions</h3>'
        '<p>Read full <code>SKILL.md</code> only when inspecting or executing a task.</p>'
        '<div class="tlinks">'
        '<a href="/skills/hanzo/SKILL.md">/skills/&lt;id&gt;/SKILL.md</a>'
        '<a href="/skills/hanzo.json">/skills/&lt;id&gt;.json</a>'
        '<a href="/oss/index.json">/oss/index.json</a>'
        '</div></div>'
        '</section>'
    )

    compact_boot = {
        "total": total,
        "categories": [
            {"slug": c["slug"], "label": c["label"], "summary": c["summary"], "count": c["count"]}
            for c in cats
        ],
    }
    boot_json = json.dumps(compact_boot, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    return (
        tiers
        + '<div class="searchbar">'
        + '<input id="q" type="search" placeholder="Search %d skills, standards &amp; HIPs (lazy-loaded on type)" autocomplete="off" spellcheck="false">' % total
        + '<div class="statusrow"><div id="count">%d categories · %d skills · progressive disclosure</div>' % (len(cats), total)
        + '<button class="clearbtn" id="clearbtn" type="button" hidden>All categories</button></div>'
        + '</div>'
        + '<section id="catgrid" class="catgrid" aria-label="Skill categories">%s</section>' % "".join(cards)
        + '<section id="browser" class="browser" aria-live="polite" hidden></section>'
        + '<script type="application/json" id="catalog-boot">%s</script>' % boot_json
    )


BROWSER_JS = """
<script id="browserjs">
(function(){
var boot=JSON.parse(document.getElementById('catalog-boot').textContent);
var q=document.getElementById('q'),cnt=document.getElementById('count'),
    grid=document.getElementById('catgrid'),br=document.getElementById('browser'),
    clr=document.getElementById('clearbtn');
var PAGE=24,catCache=new Map(),skillCache=new Map(),searchCache=null,searchPromise=null;
var state={mode:'categories',cat:null,query:'',items:[],shown:0,title:'',subtitle:'',metaLinks:''};
var catBySlug={};
boot.categories.forEach(function(c){catBySlug[c.slug]=c;});

function esc(s){return String(s).replace(/[&<>"']/g,function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m];});}

function setHash(h){try{if(h)history.replaceState(null,'','#'+h);else history.replaceState(null,'',location.pathname);}catch(_){}}

function showCategories(){
  state.mode='categories';state.cat=null;state.query='';state.items=[];state.shown=0;
  q.value='';grid.hidden=false;br.hidden=true;br.innerHTML='';clr.hidden=true;
  cnt.textContent=boot.categories.length+' categories · '+boot.total+' skills · progressive disclosure';
  grid.querySelectorAll('.catcard').forEach(function(b){b.setAttribute('aria-expanded','false');});
  setHash('');
}

function renderRow(it){
  var tags=(it.tags||[]).slice(0,4).map(function(t){return '<span class="tag">'+esc(t)+'</span>';}).join('');
  return '<article class="srow" data-id="'+esc(it.id)+'">'
    +'<div class="smain">'
    +'<div class="shead"><a class="nm" href="/'+esc(it.id)+'">'+esc(it.id)+'</a>'
    +'<span class="stag">'+esc(it.category)+'</span></div>'
    +'<p class="ds">'+esc(it.summary)+'</p>'
    +(tags?'<div class="tags">'+tags+'</div>':'')
    +'</div>'
    +'<div class="sact">'
    +'<button class="sbtn inspect" type="button" data-id="'+esc(it.id)+'" aria-expanded="false">Inspect SKILL.md</button>'
    +'<a class="sbtn" href="/skills/'+esc(it.id)+'/SKILL.md">Raw .md</a>'
    +'</div>'
    +'<div class="drawer" id="dr-'+esc(it.id)+'" hidden></div>'
    +'</article>';
}

function renderBrowser(){
  grid.hidden=true;br.hidden=false;clr.hidden=false;
  var slice=state.items.slice(0,state.shown);
  var more=state.items.length>state.shown;
  var rem=state.items.length-state.shown;
  br.innerHTML='<div class="bhead"><div><h2>'+esc(state.title)+'</h2>'
    +(state.subtitle?'<p class="bsub">'+esc(state.subtitle)+'</p>':'')+'</div>'
    +'<div class="blinks">'+state.metaLinks+'</div></div>'
    +(slice.length?'<div class="slist">'+slice.map(renderRow).join('')+'</div>'
      :'<p class="empty">No matching skills found.</p>')
    +(more?'<div class="morewrap"><button class="btn" id="morebtn" type="button">Load more ('+rem+' remaining)</button></div>':'');
}

function openCategory(slug){
  var meta=catBySlug[slug];if(!meta)return;
  state.mode='category';state.cat=slug;state.query='';
  grid.hidden=true;br.hidden=false;clr.hidden=false;
  cnt.textContent='Loading '+meta.label+' ('+meta.count+' skills)...';
  br.innerHTML='<p class="loading">Fetching /categories/'+esc(slug)+'.json...</p>';
  setHash('cat='+encodeURIComponent(slug));
  var p=catCache.has(slug)?Promise.resolve(catCache.get(slug)):fetch('/categories/'+slug+'.json').then(function(r){
    if(!r.ok)throw new Error('HTTP '+r.status);return r.json();
  }).then(function(d){catCache.set(slug,d);return d;});
  p.then(function(d){
    if(state.mode!=='category'||state.cat!==slug)return;
    state.items=d.skills||[];state.shown=Math.min(PAGE,state.items.length);
    state.title=d.label+' ('+state.items.length+')';
    state.subtitle=d.summary||'';
    state.metaLinks='<a class="sbtn" href="/categories/'+esc(slug)+'.json">Category JSON</a>'
      +'<a class="sbtn" href="/categories/'+esc(slug)+'.md">Category .md</a>';
    cnt.textContent='Showing '+state.shown+' of '+state.items.length+' in '+d.label+' · Tier 1 loaded';
    renderBrowser();
  }).catch(function(err){
    br.innerHTML='<p class="empty">Failed to load category: '+esc(err.message)+'</p>';
  });
}

function loadSearchIndex(){
  if(searchCache)return Promise.resolve(searchCache);
  if(!searchPromise){
    searchPromise=fetch('/search.json').then(function(r){
      if(!r.ok)throw new Error('HTTP '+r.status);return r.json();
    }).then(function(d){searchCache=d.s||[];return searchCache;})
    .catch(function(e){searchPromise=null;throw e;});
  }
  return searchPromise;
}

function runSearch(raw){
  var v=raw.trim().toLowerCase();
  if(!v){showCategories();return;}
  state.mode='search';state.cat=null;state.query=v;
  grid.hidden=true;br.hidden=false;clr.hidden=false;
  cnt.textContent='Searching "'+raw.trim()+'"...';
  setHash('q='+encodeURIComponent(raw.trim()));
  loadSearchIndex().then(function(list){
    if(state.mode!=='search'||state.query!==v)return;
    var words=v.split(/\\s+/).filter(Boolean),hits=[];
    for(var i=0;i<list.length;i++){
      var s=list[i],id=s.i,cat=s.c,ds=s.d.toLowerCase(),tg=(s.g||[]).join(' ');
      var sc=id===v?20:id.indexOf(v)===0?12:0,ok=true;
      for(var w=0;w<words.length;w++){
        var wd=words[w];
        if(id.indexOf(wd)>=0)sc+=4;
        else if(cat.indexOf(wd)>=0||tg.indexOf(wd)>=0)sc+=2;
        else if(ds.indexOf(wd)>=0)sc+=1;
        else{ok=false;break;}
      }
      if(ok)hits.push({sc:sc,s:s});
    }
    hits.sort(function(a,b){return b.sc-a.sc||a.s.i.length-b.s.i.length||a.s.i.localeCompare(b.s.i);});
    state.items=hits.map(function(h){return {id:h.s.i,name:h.s.i,category:h.s.c,summary:h.s.d,tags:h.s.g||[]};});
    state.shown=Math.min(PAGE,state.items.length);
    state.title='Search results for "'+raw.trim()+'" ('+state.items.length+')';
    state.subtitle='Lazy search index loaded on demand. Click any skill to inspect its SKILL.md.';
    state.metaLinks='<a class="sbtn" href="/catalog.json">Catalog JSON</a><a class="sbtn" href="/llms.txt">llms.txt</a>';
    cnt.textContent=state.items.length+(state.items.length===1?' match':' matches')+' for "'+raw.trim()+'"';
    renderBrowser();
  }).catch(function(err){
    br.innerHTML='<p class="empty">Search index unavailable: '+esc(err.message)+'</p>';
  });
}

function toggleInspect(btn){
  var id=btn.dataset.id,dr=document.getElementById('dr-'+id);if(!dr)return;
  if(!dr.hidden){dr.hidden=true;btn.setAttribute('aria-expanded','false');btn.textContent='Inspect SKILL.md';return;}
  dr.hidden=false;btn.setAttribute('aria-expanded','true');btn.textContent='Hide SKILL.md';
  if(dr.dataset.loaded==='1')return;
  dr.innerHTML='<p class="loading">Fetching /skills/'+esc(id)+'/SKILL.md (Tier 2)...</p>';
  var p=skillCache.has(id)?Promise.resolve(skillCache.get(id)):fetch('/skills/'+id+'/SKILL.md').then(function(r){
    if(!r.ok)throw new Error('HTTP '+r.status);return r.text();
  }).then(function(t){skillCache.set(id,t);return t;});
  p.then(function(md){
    var cmd='curl -fsSL https://hanzoskills.com/skills/'+id+'/SKILL.md';
    dr.dataset.loaded='1';
    dr.innerHTML='<div class="dbar"><code>'+esc(cmd)+'</code>'
      +'<div class="dacts">'
      +'<button class="sbtn dcopy" type="button" data-copy="cmd">Copy curl</button>'
      +'<button class="sbtn dcopy" type="button" data-copy="md">Copy SKILL.md</button>'
      +'<a class="sbtn" href="/skills/'+esc(id)+'.json">JSON</a>'
      +'<a class="sbtn" href="/'+esc(id)+'">Full page</a>'
      +'</div></div>'
      +'<pre class="dpre"><code>'+esc(md)+'</code></pre>';
  }).catch(function(err){
    dr.innerHTML='<p class="empty">Failed to fetch SKILL.md: '+esc(err.message)+'</p>';
  });
}

grid.addEventListener('click',function(e){
  var c=e.target.closest('.catcard');if(c)openCategory(c.dataset.cat);
});
clr.addEventListener('click',showCategories);
br.addEventListener('click',function(e){
  var more=e.target.closest('#morebtn');
  if(more){
    state.shown=Math.min(state.items.length,state.shown+PAGE);
    cnt.textContent='Showing '+state.shown+' of '+state.items.length;
    renderBrowser();return;
  }
  var ins=e.target.closest('.inspect');
  if(ins){toggleInspect(ins);return;}
  var cp=e.target.closest('.dcopy');
  if(cp){
    var art=cp.closest('.srow'),id=art&&art.dataset.id;
    var text=cp.dataset.copy==='cmd'?'curl -fsSL https://hanzoskills.com/skills/'+id+'/SKILL.md':(skillCache.get(id)||'');
    navigator.clipboard.writeText(text).then(function(){
      var prev=cp.textContent;cp.textContent='Copied';setTimeout(function(){cp.textContent=prev;},1400);
    });
  }
});
var timer=0;
q.addEventListener('input',function(){
  clearTimeout(timer);
  var val=q.value;
  timer=setTimeout(function(){runSearch(val);},80);
});
document.addEventListener('keydown',function(e){
  if(e.key==='/'&&document.activeElement!==q&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)){
    e.preventDefault();q.focus();
  }else if(e.key==='Escape'&&document.activeElement===q&&q.value){
    showCategories();
  }
});
if(location.hash){
  var hm=location.hash.match(/^#cat=([^&]+)/),qm=location.hash.match(/^#q=([^&]+)/);
  if(hm)openCategory(decodeURIComponent(hm[1]));
  else if(qm){q.value=decodeURIComponent(qm[1]);runSearch(q.value);}
}
})();
</script>"""
