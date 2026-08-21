#!/usr/bin/env python3
"""
Generate the hanzo-api skill category from the served OpenAPI document.

Input is https://api.hanzo.ai/v1/openapi.json — the document each subsystem
projects from its own router. Every endpoint, parameter, summary and example
written into a skill comes from that document. Nothing here invents API surface.

The document partitions its operations by `x-app`: the fleet app that serves the
operation. That partition IS the product grouping — one skill per app, families
(`/v1/<family>`) as sections inside it. 185 families, 121 products.

Hand-written prose is confined to NOTES below: the domain a product belongs to,
when to reach for it, and how it composes. A document cannot supply those. It
supplies everything else, so everything else is generated and regenerating is
the way to fix a stale skill.

Output is the repository's existing three tiers, no new format:

    skills/discover-hanzo-api/SKILL.md   gateway, auto-activates on keywords
    skills/hanzo-api/INDEX.md            category index, products by domain
    skills/hanzo-api/<product>.md        one skill per product

Usage:

    python3 scripts/generate_hanzo_api_skills.py --plan
    python3 scripts/generate_hanzo_api_skills.py --only kms,sandbox,ai
    python3 scripts/generate_hanzo_api_skills.py
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

DOC_URL = "https://api.hanzo.ai/v1/openapi.json"
CATEGORY = "hanzo-api"
METHODS = ("get", "post", "put", "patch", "delete", "head", "options")

# Detail budget. A product at or under this many operations gets a summary line
# per operation; above it, operations are listed method+path and the reader is
# pointed at the live document for the prose. Keeps every skill near the
# repository's ~320-line norm without truncating the surface.
DETAIL_MAX = 40


# --- hand-written prose -------------------------------------------------------
#
# `domain` groups products in the INDEX and the gateway. `when` and `composes`
# are the two things the document cannot know: why a caller reaches for this
# product, and what it is used with. Both optional — a product without them
# generates without those sections rather than with filler.
#
# `name` overrides the derived skill name. Needed where singularising two app
# names collides — the generator refuses to run rather than guess — and where a
# name only looks plural: `analytics` is a mass noun, not analytics of analytic.
# Initialisms are handled by the rule (a stem with no vowel is never a plural),
# so kms and dns need no entry.

NOTES = {
    # --- model -------------------------------------------------------------
    "ai": {
        "title": "AI",
        "domain": "model",
        "when": [
            "Calling a model — chat, messages, embeddings, images, audio, video, rerank",
            "You want one credential and one base URL across every provider",
            "Porting code written against the OpenAI or Anthropic SDK: point base_url here",
            "Choosing a model: /v1/models carries context window, modalities and price",
        ],
        "composes": [
            "`pricing` prices a model before you call it; `usage` and `billing` report what the call cost",
            "`router` paths inside this product pick a model per request; `plan` decides which are allowed",
            "`sandbox` runs code the model writes; `tool` lists what the model may call",
            "`o11y` receives the traces — a model call is the span most worth keeping",
        ],
    },
    "engine": {"domain": "model"},
    "ml": {"domain": "model"},
    "benchmark": {"domain": "model"},
    "evals": {"domain": "model"},
    "prompts": {"domain": "model"},
    "research": {"domain": "model"},
    "experiments": {"domain": "model"},
    "translate": {"domain": "model"},
    # --- agent -------------------------------------------------------------
    "agents": {"domain": "agent"},
    "sandboxes": {
        "domain": "agent",
        "when": [
            "Running code you did not write — a model wrote it, or a user submitted it",
            "You need a filesystem, a terminal or a screen that outlives a single call",
            "Isolating a long-running job from the caller's process and from other tenants",
            "Driving a desktop: the screen and terminal websockets are on the same object",
        ],
        "composes": [
            "`ai` writes the code, this runs it — the pairing behind every code-interpreter loop",
            "`exec` is the one-shot form of the same primitive: no lease, no id, no lifetime to manage",
            "`storage` and `s3` hold what a sandbox produced after the sandbox is gone",
            "`visor` rents the machines underneath; a sandbox is a pod, not a VM you manage",
        ],
    },
    "exec": {"domain": "agent"},
    "tools": {"domain": "agent"},
    "skills": {"domain": "agent"},
    "flow": {"domain": "agent"},
    "auto": {"domain": "agent"},
    "automations": {"domain": "agent"},
    "tasks": {"domain": "agent"},
    "marketplace": {"domain": "agent"},
    "code": {"domain": "agent"},
    "lsp": {"title": "LSP", "domain": "agent"},
    "bot": {"domain": "agent", "name": "node"},
    "bots": {"domain": "agent", "name": "bot"},
    # --- search ------------------------------------------------------------
    "index": {"domain": "search"},
    "search": {"domain": "search"},
    "product": {"domain": "search"},
    "knowledge": {"domain": "search"},
    "crawl": {"domain": "search"},
    "websearch": {"domain": "search"},
    "world": {"domain": "search"},
    "ask": {"domain": "search"},
    # --- identity ----------------------------------------------------------
    "iam": {"title": "IAM", "domain": "identity"},
    "account": {"domain": "identity"},
    "authz": {"title": "Authz", "domain": "identity"},
    "entitlements": {"domain": "identity"},
    "prefs": {"domain": "identity"},
    "settings": {"domain": "identity"},
    "admin": {"domain": "identity"},
    "plugins": {"domain": "identity"},
    # --- secret ------------------------------------------------------------
    "kms": {
        "title": "KMS",
        "domain": "secret",
        "when": [
            "Your code needs a credential and you refuse to put it in a manifest or an env file",
            "Rotating a secret without redeploying whatever reads it",
            "You need a signature from a key the caller is never allowed to hold",
            "Auditing what secrets an org holds without revealing any value",
        ],
        "composes": [
            "Everything that takes a third-party credential reads it from here — `integrations`, `notify`, `settings`",
            "`iam` decides who the caller is; this decides what that caller may unseal",
            "`audit` records the read; the two together are the evidence trail",
            "`platform` and `functions` inject secrets into a deployment by reference, never by value",
        ],
    },
    "wallets": {"domain": "secret"},
    "security": {"domain": "secret"},
    "sbom": {"title": "SBOM", "domain": "secret"},
    "audit": {"domain": "secret"},
    # --- money -------------------------------------------------------------
    "commerce": {"domain": "money"},
    "billing": {"domain": "money"},
    "plan": {"domain": "money"},
    "pricing": {"domain": "money"},
    "allowance": {"domain": "money"},
    "usage": {"domain": "money"},
    "leaderboard": {"domain": "money"},
    "books": {"domain": "money"},
    "treasury": {"domain": "money"},
    "x402": {"title": "x402", "domain": "money"},
    "licensing": {"domain": "money"},
    "affiliates": {"domain": "money"},
    "referrals": {"domain": "money"},
    "authors": {"domain": "money"},
    # --- compute -----------------------------------------------------------
    "visor": {"domain": "compute"},
    "venue": {"domain": "compute"},
    "provisioning": {"domain": "compute"},
    "storage": {"domain": "compute"},
    "registry": {"domain": "compute"},
    "mq": {"title": "MQ", "domain": "compute"},
    "pubsub": {"domain": "compute"},
    "sync": {"domain": "compute"},
    # --- deploy ------------------------------------------------------------
    "platform": {"domain": "deploy"},
    "projects": {"domain": "deploy"},
    "deploy": {"domain": "deploy"},
    "functions": {"domain": "deploy"},
    "blueprint": {"domain": "deploy"},
    "templates": {"domain": "deploy"},
    "catalog": {"domain": "deploy"},
    "base": {"domain": "deploy"},
    "git": {"domain": "deploy"},
    "ingress": {"domain": "deploy"},
    "gateway": {"domain": "deploy"},
    "dns": {"domain": "deploy"},
    "domain": {"domain": "deploy"},
    "cloudflare": {"domain": "deploy"},
    "zt": {"title": "Zero Trust", "domain": "deploy"},
    "share": {"domain": "deploy"},
    # --- telemetry ---------------------------------------------------------
    "o11y": {"title": "o11y", "domain": "telemetry"},
    "metrics": {"domain": "telemetry"},
    "analytics": {"domain": "telemetry", "name": "analytics"},
    "destinations": {"domain": "telemetry"},
    # --- reach -------------------------------------------------------------
    "marketing": {"domain": "reach"},
    "campaign": {"domain": "reach"},
    "ads": {"domain": "reach"},
    "content": {"domain": "reach"},
    "social": {"domain": "reach"},
    "notify": {"domain": "reach"},
    "tel": {"domain": "reach"},
    "meet": {"domain": "reach"},
    "channels": {"domain": "reach"},
    "help": {"domain": "reach"},
    "webhooks": {"domain": "reach"},
    "link": {"domain": "reach"},
    "integrations": {"domain": "reach"},
    "crm": {"domain": "reach"},
    # --- company -----------------------------------------------------------
    "company": {"domain": "company"},
    "legal": {"domain": "company"},
    "compliance": {"domain": "company"},
    "esign": {"domain": "company"},
    "captable": {"domain": "company"},
    "dataroom": {"domain": "company"},
    "guide": {"domain": "company"},
    "framework": {"domain": "company"},
    "taxonomy": {"domain": "company"},
    "todo": {"domain": "company"},
    "team": {"domain": "company"},
    "risk": {"domain": "company"},
    "dataset": {"domain": "company"},
    "label": {"domain": "company"},
    "reference": {"domain": "company"},
    "flags": {"domain": "company"},
    "admission": {"domain": "company"},
    # --- chain -------------------------------------------------------------
    "web3": {"domain": "chain"},
    "explorer": {"domain": "chain"},
    "validators": {"domain": "chain"},
    # --- meta --------------------------------------------------------------
    "openapi": {"title": "OpenAPI", "domain": "meta"},
}

DOMAIN_ORDER = [
    ("model", "Models and inference"),
    ("agent", "Agents, code and workflow"),
    ("search", "Retrieval"),
    ("identity", "Identity and access"),
    ("secret", "Secrets and assurance"),
    ("money", "Money"),
    ("compute", "Compute and data plumbing"),
    ("deploy", "Shipping and the front door"),
    ("telemetry", "Telemetry"),
    ("reach", "Reaching people"),
    ("company", "Running a company"),
    ("chain", "Chain"),
    ("meta", "The document itself"),
]

AUTH = """## Authentication

Every operation takes a Hanzo IAM bearer token. `hanzo auth token` prints the
active short-lived access token for the identity you are signed in as; the org
is carried in the token's claim, never in a path or a query parameter.

```bash
TOKEN=$(hanzo auth token)
curl -sS https://api.hanzo.ai/v1/... -H "Authorization: Bearer $TOKEN"
```

The document declares one scheme, `bearer`, at the top level: `Authorization:
Bearer <token>`. `X-Authorization` and HTTP Basic carrying the token as the
password are accepted spellings of the same header; a browser presents the
session cookie instead. There is no product-specific key scheme."""


# --- reading the document -----------------------------------------------------

def load_doc(source):
    """Read the served document from a URL or a local path."""
    if re.match(r"^https?://", source):
        with urllib.request.urlopen(source, timeout=60) as r:
            return json.load(r)
    return json.loads(Path(source).read_text())


def family_of(path):
    """The product family a path belongs to: the segment after /v1/."""
    seg = path.split("/")
    if len(seg) > 2 and seg[1] == "v1":
        return seg[2]
    return "/" + seg[1] if len(seg) > 1 and seg[1] else "/"


def title_of(name):
    """`kms` is not a word to capitalise, it is an initialism. A stem with no
    vowel says so on its own; the rest that only look like words take a `title`
    override in NOTES."""
    words = []
    for w in name.replace("-", " ").split():
        words.append(w.upper() if not any(c in "aeiouy" for c in w) else w.title())
    return " ".join(words)


def singular(name):
    """Strip a plural. A stem with no vowel is an initialism, not a plural, so
    kms and dns are left alone; anything else genuinely irregular takes a `name`
    override in NOTES."""
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith(("sses", "shes", "ches", "xes")):
        return name[:-2]
    if name.endswith("s") and not name.endswith(("ss", "us")):
        stem = name[:-1]
        if any(c in "aeiouy" for c in stem):
            return stem
    return name


def collect(doc):
    """Partition the document's operations by the app that serves them."""
    tag_desc = {t["name"]: (t.get("description") or "") for t in doc.get("tags", [])}
    apps = {}
    for path, item in doc["paths"].items():
        for method, op in item.items():
            if method not in METHODS:
                continue
            app = op.get("x-app") or "unknown"
            apps.setdefault(app, []).append((path, method, op))
    products = {}
    for app, ops in apps.items():
        ops.sort(key=lambda t: (family_of(t[0]), t[0], t[1]))
        products[app] = {
            "app": app,
            "name": NOTES.get(app, {}).get("name") or singular(app),
            "ops": ops,
            "families": sorted({family_of(p) for p, _, _ in ops}),
            "purpose": purpose_of(app, ops, tag_desc),
        }
    return products


def purpose_of(app, ops, tag_desc):
    """The product's one-line purpose, taken from the document.

    A package's own doc comment reaches the document as the description of every
    tag that package serves. Prefer the line that names this app, then the line
    on the tag spelled like the app, then the app's own first summary — never a
    neighbouring package's line silently standing in for a missing one.
    """
    tags = []
    for _, _, op in ops:
        for t in op.get("tags", []):
            if t not in tags:
                tags.append(t)
    for prefix in (f"Package {app} is ", f"Package {singular(app)} is ",
                   f"Package {app} ", f"Package {singular(app)} "):
        for t in tags:
            if tag_desc.get(t, "").startswith(prefix):
                return strip_package(tag_desc[t])
    if tag_desc.get(app):
        return strip_package(tag_desc[app])
    return ops[0][2].get("summary", "").strip()


def strip_package(text):
    text = text.split("\n\n")[0].strip()
    text = re.sub(r"^Package \w+ (is|mounts) ", "", text)
    return text[:1].upper() + text[1:] if text else text


def count(n, noun):
    return f"{n} {noun}" + ("" if n == 1 else "s")


def one_line(text, limit=150):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


# --- example calls ------------------------------------------------------------

def example_call(path, method, op, schemas):
    """A curl built only from what the document states.

    A path parameter is filled with the document's own example when it carries
    one and left as its literal `{name}` placeholder when it does not — an
    unfilled placeholder is honest, a guessed value is not. Query parameters
    appear only when the document marks them required. The body is the
    document's own example; a body with no example is named by its schema
    instead of being invented.
    """
    url = path
    params = op.get("parameters") or []
    for p in params:
        if p.get("in") == "path" and p.get("example") is not None:
            url = url.replace("{" + p["name"] + "}", str(p["example"]))
    query = []
    for p in params:
        if p.get("in") == "query" and p.get("required"):
            val = p.get("example")
            query.append(f"{p['name']}={val if val is not None else '<' + p['name'] + '>'}")
    if query:
        url = url + "?" + "&".join(query)

    # The document decides whether a call carries a token: a top-level `security`
    # of bearer unless the operation overrides it with `security: []` or marks
    # itself public. Prose that CLAIMS a route is open does not count.
    open_route = op.get("security") == [] or op.get("x-public") is True

    lines = []
    verb = "" if method == "get" else f" -X {method.upper()}"
    lines.append(f'curl -sS{verb} "https://api.hanzo.ai{url}"'
                 + ("" if open_route else " \\"))
    if not open_route:
        lines.append('  -H "Authorization: Bearer $TOKEN"')

    body, schema = request_body(op)
    if body is not None:
        lines[-1] += " \\"
        lines.append("  -H 'Content-Type: application/json' \\")
        payload = json.dumps(body, indent=2, ensure_ascii=False)
        payload = "\n".join(["  -d '" + payload.split("\n")[0]] +
                            ["  " + ln for ln in payload.split("\n")[1:]]) + "'"
        lines.append(payload)
    trailer = ""
    if body is None and schema:
        rows = schema_table(schemas, schema)
        trailer = f"\nBody — `{schema}`:\n\n" + "\n".join(rows) if rows else (
            f"\nThe body is `{schema}`; the document declares no fields for it.")
    return "\n".join(lines), trailer


def request_body(op):
    rb = op.get("requestBody") or {}
    for _, media in (rb.get("content") or {}).items():
        if media.get("example") is not None:
            return media["example"], None
        ref = (media.get("schema") or {}).get("$ref")
        if ref:
            return None, ref.rsplit("/", 1)[-1]
    return None, None


def rank(path, method, op, i, purpose=""):
    """How worth showing an operation is, decided only by what the document says
    about it: an operation it illustrates, then one it publishes without a
    token, then one it surfaces as an agent tool, then a plain collection read."""
    body, _ = request_body(op)
    illustrated = body is not None or any(
        p.get("example") is not None for p in (op.get("parameters") or []))
    return (0 if illustrated else 1,
            0 if path in purpose else 1,
            0 if op.get("x-public") else 1,
            0 if op.get("x-tool") else 1,
            0 if "{" not in path else 1,
            -len(op.get("description") or ""),
            i)


def schema_table(schemas, name, limit=10):
    """The body's fields, straight out of the document's own schema. A caller
    reading `The body is RunIn` still cannot make the call; a caller reading the
    field list can."""
    schema = schemas.get(name) or {}
    props = schema.get("properties") or {}
    if not props:
        return []
    required = set(schema.get("required") or [])
    rows = ["| Field | Type | Required | What it is |", "|---|---|---|---|"]
    for field, spec in list(props.items())[:limit]:
        kind = spec.get("type", "")
        if kind == "array":
            kind = f"array of {(spec.get('items') or {}).get('type', 'object')}"
        if not kind and spec.get("$ref"):
            kind = spec["$ref"].rsplit("/", 1)[-1]
        first = (spec.get("description") or "").strip().split("\n\n")[0]
        rows.append(f"| `{field}` | {kind or 'object'} | "
                    f"{'yes' if field in required else '—'} | {one_line(first, 110)} |")
    if len(props) > limit:
        rows.append(f"| … | | | {len(props) - limit} more fields — "
                    f"`jq '.components.schemas.{name}' `on the document |")
    return rows


def pick_examples(ops, want=3, purpose=""):
    """The best operations to show, with a write among them when the product has
    one — a product read but never written is a product half taught."""
    scored = sorted(((rank(p, m, o, i, purpose), (p, m, o))
                     for i, (p, m, o) in enumerate(ops)), key=lambda t: t[0])
    picks = [op for _, op in scored[:want]]
    if not any(m != "get" for _, m, _ in picks):
        writes = [op for _, op in scored if op[1] != "get"]
        if writes:
            picks = picks[:want - 1] + [writes[0]]
    return picks


# --- rendering ----------------------------------------------------------------

def render_skill(prod, products, schemas, repo, today):
    name, app = prod["name"], prod["app"]
    note = NOTES.get(app, {})
    ops = prod["ops"]
    fams = prod["families"]
    title = note.get("title") or title_of(name)
    detail = len(ops) <= DETAIL_MAX

    out = []
    out.append("---")
    out.append(f"name: {CATEGORY}-{name}")
    out.append(f"description: {one_line(prod['purpose'], 200)}")
    out.append("---")
    out.append("")
    out.append(f"# Hanzo {title}")
    out.append("")
    out.append(f"**Scope**: {one_line(prod['purpose'], 120)}")
    shown = ["`" + ("/v1/" + f if not f.startswith("/") else f) + "`" for f in fams]
    more = "" if len(shown) <= 8 else f" and {len(shown) - 8} more"
    out.append(f"**Surface**: {count(len(ops), 'operation')} on "
               f"{', '.join(shown[:8])}{more}")
    out.append("**Lines**: ~LINES")
    out.append(f"**Last Updated**: {today}")
    out.append(f"**Source**: {DOC_URL} (`x-app: {app}`) — generated, do not hand-edit")
    out.append("")
    out.append("## When to Use This Skill")
    out.append("")
    out.append("Activate this skill when:")
    for w in note.get("when", []):
        out.append(f"- {w}")
    ranked = sorted(fams, key=lambda f: -sum(1 for p, _, _ in ops if family_of(p) == f))
    for f in ranked[:6]:
        out.append(f"- Calling `{'/v1/' + f if not f.startswith('/') else f}` "
                   f"({count(sum(1 for p, _, _ in ops if family_of(p) == f), 'operation')})")
    if len(ranked) > 6:
        out.append(f"- Calling any of this product's {len(ranked) - 6} smaller families "
                   f"(see Surface below)")
    out.append("")
    out.append(AUTH)
    out.append("")

    # --- surface -------------------------------------------------------------
    out.append("## Surface")
    out.append("")
    out.append("| Family | Operations | Addresses under it |")
    out.append("|---|---|---|")
    for f in fams:
        fops = [o for o in ops if family_of(o[0]) == f]
        shown = "/v1/" + f if not f.startswith("/") else f
        subs = []
        for path, _, _ in fops:
            rest = path[len(shown):].strip("/").split("/")[0]
            if rest and not rest.startswith("{") and rest not in subs:
                subs.append(rest)
        under = ", ".join(f"`{x}`" for x in subs[:8]) or "the family root"
        if len(subs) > 8:
            under += f", +{len(subs) - 8}"
        out.append(f"| `{shown}` | {len(fops)} | {under} |")
    out.append("")

    # --- operations ----------------------------------------------------------
    out.append("## Operations")
    out.append("")
    if not detail:
        out.append(f"{len(ops)} operations — listed by address. Read any one's prose from "
                   f"the document:")
        out.append("")
        out.append("```bash")
        out.append(f"curl -sS {DOC_URL} \\")
        out.append(f"  | jq '.paths | to_entries[] | select(.value[].\"x-app\" == \"{app}\") "
                   f"| {{path: .key, ops: (.value | map_values(.summary))}}'")
        out.append("```")
        out.append("")
    for f in fams:
        fops = [o for o in ops if family_of(o[0]) == f]
        out.append(f"### `{'/v1/' + f if not f.startswith('/') else f}`")
        out.append("")
        if detail:
            out.append("| Method | Path | Summary |")
            out.append("|---|---|---|")
            for path, method, op in fops:
                out.append(f"| `{method.upper()}` | `{path}` | "
                           f"{one_line(op.get('summary', ''), 110)} |")
        else:
            for path, method, op in fops:
                out.append(f"- `{method.upper()} {path}`")
        out.append("")

    # --- examples ------------------------------------------------------------
    out.append("## Example Calls")
    out.append("")
    out.append("```bash")
    out.append("TOKEN=$(hanzo auth token)")
    out.append("```")
    out.append("")
    for path, method, op in pick_examples(ops, purpose=prod["purpose"]):
        out.append(f"**{one_line(op.get('summary', path), 110)}**")
        out.append("")
        call, trailer = example_call(path, method, op, schemas)
        out.append("```bash")
        out.append(call)
        out.append("```")
        if trailer:
            out.append(trailer)
        params = [p for p in (op.get("parameters") or [])
                  if p.get("in") == "path" and p.get("example") is None]
        for p in params:
            out.append(f"- `{{{p['name']}}}` — {one_line(p.get('description', ''), 120)}")
        desc = (op.get("description") or "").strip()
        # A write the document gives no schema for still states its fields — in
        # prose. Carry all of it rather than leave the caller a bodiless POST.
        body_unstated = method != "get" and not op.get("requestBody")
        paras = desc.split("\n\n") if body_unstated else desc.split("\n\n")[:1]
        for para in paras:
            if para.strip() and para.strip() != op.get("summary", "").strip():
                out.append("")
                out.append(one_line(para, 420))
        out.append("")

    # --- what the document says about answers --------------------------------
    codes = {}
    public = 0
    for _, _, op in ops:
        for c in (op.get("responses") or {}):
            codes[c] = codes.get(c, 0) + 1
        if op.get("x-public"):
            public += 1
    if codes or public:
        out.append("## Answers")
        out.append("")
        if codes:
            out.append("Documented status codes across this product: " +
                       ", ".join(f"`{c}` ({n})" for c, n in sorted(codes.items())) + ".")
            out.append("")
        if public:
            out.append(f"{public} of these operations are marked public in the document — "
                       f"they are reachable without a bearer token. The rest are not.")
            out.append("")

    # --- composition ---------------------------------------------------------
    if note.get("composes"):
        out.append("## Composition")
        out.append("")
        for c in note["composes"]:
            out.append(f"- {c}")
        out.append("")

    # --- related -------------------------------------------------------------
    out.append("## Related Skills")
    out.append("")
    siblings = [p for p in products.values()
                if p["app"] != app
                and NOTES.get(p["app"], {}).get("domain") == note.get("domain")]
    siblings.sort(key=lambda p: -len(p["ops"]))
    for s in siblings[:4]:
        out.append(f"- `{CATEGORY}/{s['name']}.md` — {one_line(s['purpose'], 90)}")
    legacy = repo / "skills" / "hanzo" / f"hanzo-{name}.md"
    if legacy.exists():
        out.append(f"- `hanzo/hanzo-{name}.md` — the repo and how it is operated, "
                   f"not the API surface")
    out.append("- `hanzo-cloud-architecture/SKILL.md` — how one binary serves all of this")
    out.append(f"- `{CATEGORY}/INDEX.md` — every Hanzo product, by domain")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"**Last Updated**: {today}")
    out.append(f"**Generated from**: {DOC_URL}")
    out.append("")

    text = "\n".join(out)
    return text.replace("~LINES", f"~{text.count(chr(10)) + 1}")


def render_index(products, written, today):
    total_ops = sum(len(p["ops"]) for p in products.values())
    fams = {f for p in products.values() for f in p["families"]}
    v1 = {f for f in fams if not f.startswith("/")}
    out = []
    out.append("# Hanzo API Skills")
    out.append("")
    out.append("How to call Hanzo. One skill per product, generated from the document "
               "`api.hanzo.ai` serves.")
    out.append("")
    out.append("## Category Overview")
    out.append("")
    out.append(f"**Total Skills**: {len(products)}")
    out.append(f"**Surface**: {len(v1)} `/v1` families plus {len(fams) - len(v1)} "
               f"roots outside `/v1`, {total_ops} operations, one bearer token")
    out.append(f"**Source**: {DOC_URL}")
    out.append("**Focus**: Reaching Hanzo's own products from code and from an agent")
    out.append("")
    out.append("The document partitions its operations by the app that serves them. "
               "That partition is this catalogue: a product is an app, a family is a "
               "`/v1/<family>` prefix inside it. Regenerate with "
               "`python3 scripts/generate_hanzo_api_skills.py`.")
    out.append("")
    out.append("## Skills in This Category")
    out.append("")
    byd = {}
    for p in products.values():
        byd.setdefault(NOTES.get(p["app"], {}).get("domain", "meta"), []).append(p)
    for key, label in DOMAIN_ORDER:
        group = sorted(byd.get(key, []), key=lambda p: -len(p["ops"]))
        if not group:
            continue
        out.append(f"### {label} ({len(group)} skills)")
        out.append("")
        out.append("| Skill | Serves | Operations | Purpose |")
        out.append("|---|---|---|---|")
        for p in group:
            mark = "" if p["app"] in written else " *(not yet written)*"
            fams = ", ".join("`/v1/" + f + "`" for f in p["families"]
                             if not f.startswith("/"))[:70]
            out.append(f"| `{p['name']}.md`{mark} | {fams} | {len(p['ops'])} | "
                       f"{one_line(p['purpose'], 90)} |")
        out.append("")
    out.append("---")
    out.append("")
    out.append(f"**Last Updated**: {today}")
    out.append("")
    return "\n".join(out)


def render_gateway(products, today):
    total_ops = sum(len(p["ops"]) for p in products.values())
    byd = {}
    for p in products.values():
        byd.setdefault(NOTES.get(p["app"], {}).get("domain", "meta"), []).append(p)
    keywords = ", ".join(p["name"] for p in
                         sorted(products.values(), key=lambda p: -len(p["ops"]))[:28])
    out = []
    out.append("---")
    out.append(f"name: discover-{CATEGORY}")
    out.append("description: Automatically discover how to CALL Hanzo's own products "
               "over api.hanzo.ai — models, agents, sandboxes, secrets, billing, "
               "deployments, identity, telemetry. Activates when a task needs a Hanzo "
               "API rather than code written from scratch: " + keywords + ".")
    out.append("---")
    out.append("")
    out.append("# Hanzo API Skills Discovery")
    out.append("")
    out.append(f"Provides automatic access to {len(products)} product skills covering "
               f"{total_ops} operations on `api.hanzo.ai`, generated from the document "
               f"the API serves.")
    out.append("")
    out.append("## When This Skill Activates")
    out.append("")
    out.append("This skill auto-activates when you're working with:")
    out.append("- Calling a Hanzo product instead of building the capability yourself")
    out.append("- `api.hanzo.ai`, `/v1/...` paths, or a `hanzo` CLI command")
    out.append("- Authenticating to Hanzo (`hanzo auth token`, IAM bearer tokens)")
    out.append("- Choosing which Hanzo product owns a capability you need")
    out.append("")
    out.append("## Available Skills")
    out.append("")
    out.append("### By Domain")
    out.append("")
    for key, label in DOMAIN_ORDER:
        group = sorted(byd.get(key, []), key=lambda p: -len(p["ops"]))
        if not group:
            continue
        names = ", ".join(f"`{p['name']}`" for p in group)
        out.append(f"**{label}** — {names}")
        out.append("")
    out.append("### Load Full Category Details")
    out.append("")
    out.append("```bash")
    out.append(f"cat skills/{CATEGORY}/INDEX.md")
    out.append("```")
    out.append("")
    out.append("### Load Specific Skills")
    out.append("")
    out.append("```bash")
    biggest = sorted(products.values(), key=lambda p: -len(p["ops"]))[:6]
    for p in biggest:
        out.append(f"cat skills/{CATEGORY}/{p['name']}.md")
    out.append("```")
    out.append("")
    out.append("## Authentication")
    out.append("")
    out.append(AUTH.split("\n", 2)[2])
    out.append("")
    out.append("## Skill Selection Guide")
    out.append("")
    out.append("**Do not write it — check whether Hanzo serves it.** The surface is "
               "wider than most callers expect: incorporation, cap tables, DNS, "
               "e-signature, telephony and double-entry accounting are products here, "
               "not just models and storage.")
    out.append("")
    out.append("**Find the owner of a capability** without loading anything:")
    out.append("")
    out.append("```bash")
    out.append(f"curl -sS {DOC_URL} \\")
    out.append("  | jq -r '.paths | to_entries[] | .key as $p | .value | to_entries[] "
               "| \"\\(.value[\"x-app\"])\\t\\($p)\\t\\(.value.summary)\"' \\")
    out.append("  | grep -i '<what you need>'")
    out.append("```")
    out.append("")
    out.append("## Integration with Other Skills")
    out.append("")
    out.append("- `discover-hanzo` — what each Hanzo repo IS and how it is operated; "
               "this gateway is how to CALL it")
    out.append("- `hanzo-cloud-architecture/SKILL.md` — one binary, ZAP transport, "
               "why every product answers on one host")
    out.append("- `discover-api` — designing your own API; this one consumes ours")
    out.append("")
    out.append("## Progressive Loading")
    out.append("")
    out.append("- **Level 1**: This gateway (~200 lines)")
    out.append(f"- **Level 2**: `skills/{CATEGORY}/INDEX.md` — all {len(products)} "
               f"products by domain")
    out.append(f"- **Level 3**: `skills/{CATEGORY}/<product>.md` — one product's "
               f"whole surface")
    out.append(f"- **Level 4**: `{DOC_URL}` — the served document, always current")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"**Next Steps**: Run `cat skills/{CATEGORY}/INDEX.md` to see every "
               f"product, or load a product skill directly.")
    out.append("")
    out.append(f"**Last Updated**: {today}")
    out.append("")
    return "\n".join(out)


# --- driver -------------------------------------------------------------------

def check_notes(products):
    dead = sorted(set(NOTES) - set(products))
    if dead:
        print(f"\u2717 NOTES names no app the document serves: {', '.join(dead)}",
              file=sys.stderr)
        raise SystemExit(1)


def check_names(products):
    seen = {}
    for p in products.values():
        seen.setdefault(p["name"], []).append(p["app"])
    bad = {n: a for n, a in seen.items() if len(a) > 1}
    if bad:
        for n, apps in sorted(bad.items()):
            print(f"✗ name collision: {apps} both singularise to '{n}'", file=sys.stderr)
        print("Add a `name` override in NOTES for one of them, naming the product for "
              "what its paths address.", file=sys.stderr)
        raise SystemExit(1)


def plan(products):
    byd = {}
    for p in products.values():
        byd.setdefault(NOTES.get(p["app"], {}).get("domain", "«unassigned»"), []).append(p)
    fams = {f for p in products.values() for f in p["families"]}
    v1 = {f for f in fams if not f.startswith("/")}
    ops = sum(len(p["ops"]) for p in products.values())
    print(f"{len(v1)} /v1 families + {len(fams) - len(v1)} non-/v1 roots "
          f"-> {len(products)} products -> {len(byd)} domains, {ops} operations")
    for key, label in DOMAIN_ORDER + [("«unassigned»", "UNASSIGNED")]:
        group = sorted(byd.get(key, []), key=lambda p: -len(p["ops"]))
        if not group:
            continue
        print(f"\n{label} ({len(group)})")
        for p in group:
            print(f"  {p['name']:<16} {len(p['ops']):>4} ops  "
                  f"x-app={p['app']:<14} {one_line(p['purpose'], 70)}")
    missing = [p["name"] for p in products.values() if not p["purpose"]]
    if missing:
        print(f"\n! no purpose line in the document: {', '.join(sorted(missing))}")
    thin = [p["name"] for p in products.values()
            if not NOTES.get(p["app"], {}).get("when")]
    print(f"\n{len(products) - len(thin)}/{len(products)} products carry hand-written "
          f"'when to reach for this'; the rest generate without that section.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", default=DOC_URL, help="URL or path of the OpenAPI document")
    ap.add_argument("--out", default=None, help="repository root (default: cwd)")
    ap.add_argument("--only", default=None,
                    help="comma-separated product names to write (default: all)")
    ap.add_argument("--plan", action="store_true", help="print the grouping and exit")
    args = ap.parse_args()

    doc = load_doc(args.doc)
    products = collect(doc)
    check_notes(products)
    check_names(products)

    if args.plan:
        plan(products)
        return

    repo = Path(args.out or ".").resolve()
    today = date.today().isoformat()
    only = set(args.only.split(",")) if args.only else None
    chosen = {a: p for a, p in products.items() if only is None or p["name"] in only}
    if only and len(chosen) != len(only):
        missing = only - {p["name"] for p in chosen.values()}
        print(f"✗ no such product: {', '.join(sorted(missing))}", file=sys.stderr)
        raise SystemExit(1)

    cat = repo / "skills" / CATEGORY
    cat.mkdir(parents=True, exist_ok=True)
    for p in sorted(chosen.values(), key=lambda p: p["name"]):
        (cat / f"{p['name']}.md").write_text(
            render_skill(p, products, doc.get("components", {}).get("schemas", {}),
                         repo, today))
        print(f"✓ {CATEGORY}/{p['name']}.md  ({len(p['ops'])} operations)")

    (cat / "INDEX.md").write_text(render_index(products, set(chosen), today))
    print(f"✓ {CATEGORY}/INDEX.md  ({len(products)} products)")

    gw = repo / "skills" / f"discover-{CATEGORY}"
    gw.mkdir(parents=True, exist_ok=True)
    (gw / "SKILL.md").write_text(render_gateway(products, today))
    print(f"✓ discover-{CATEGORY}/SKILL.md")


if __name__ == "__main__":
    main()
