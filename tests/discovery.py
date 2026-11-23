#!/usr/bin/env python3
"""The agent discovery documents: well formed in public/, and served as committed.

    python3 tests/discovery.py public                         # the files
    python3 tests/discovery.py public https://hanzoskills.com # and the live host

/auth.md (H1 names auth.md), /.well-known/api-catalog (RFC 9727 linkset),
/.well-known/mcp/server-card.json (MCP Server Card v1) and
/.well-known/agent-skills/index.json (Agent Skills Discovery 0.2.0, written by
scripts/site.py). Live, each must answer 200 with the committed bytes, and every
skill's digest must match the bytes the host serves for it.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = sys.argv[1] if len(sys.argv) > 1 else "public"
BASE = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DOCS = ["auth.md", ".well-known/api-catalog", ".well-known/mcp/server-card.json",
        ".well-known/agent-skills/index.json"]

failed = 0


def ok(name, cond, detail=""):
    global failed
    failed += not cond
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, "  - " + detail if detail else ""))


def read(rel):
    return open(os.path.join(ROOT, rel), "rb").read()


def get(rel):
    req = urllib.request.Request(BASE + "/" + rel, headers={"User-Agent": "hanzoskills-deploy"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), b""


auth = read("auth.md").decode("utf-8")
ok("auth.md: first line is an H1 naming auth.md", re.match(r"^# .*auth\.md", auth) is not None)

catalog = json.loads(read(".well-known/api-catalog"))
links = catalog.get("linkset")
ok("api-catalog: a non-empty linkset", isinstance(links, list) and len(links) > 0)
for entry in links or []:
    hrefs = [t["href"] for k, v in entry.items() if k != "anchor" for t in v]
    ok("api-catalog: %s has an https anchor and https targets" % entry.get("anchor"),
       str(entry.get("anchor", "")).startswith("https://") and hrefs
       and all(h.startswith("https://") for h in hrefs))

card = json.loads(read(".well-known/mcp/server-card.json"))
ok("server card: v1 schema", card.get("$schema") ==
   "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json")
ok("server card: reverse-DNS name with one slash",
   re.match(r"^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$", card.get("name", "")) is not None)
ok("server card: version and a description of at most 100 characters",
   bool(card.get("version")) and 0 < len(card.get("description", "")) <= 100)
ok("server card: every remote is https streamable-http",
   bool(card.get("remotes")) and all(r["type"] == "streamable-http" and r["url"].startswith("https://")
                                     for r in card["remotes"]))

index = json.loads(read(".well-known/agent-skills/index.json"))
skills = index.get("skills") or []
ok("skills index: 0.2.0 schema", index.get("$schema") ==
   "https://schemas.agentskills.io/discovery/0.2.0/schema.json")
ok("skills index: lists skills", len(skills) > 0, "%d" % len(skills))
bad = [s.get("name") for s in skills
       if not (NAME.match(s.get("name", "")) and len(s["name"]) <= 64 and s.get("type") == "skill-md"
               and 0 < len(s.get("description", "")) <= 1024 and s.get("url") == "/%s.md" % s["name"]
               and s.get("digest") == "sha256:" + hashlib.sha256(read(s["name"] + ".md")).hexdigest())]
ok("skills index: every entry well formed, its digest the committed file's", not bad, ", ".join(bad[:5]))
ok("skills index: names are unique", len({s["name"] for s in skills}) == len(skills))

# The index is site.py's: a skill is a SKILL.md served whole, with every file it
# points at published beside it.
spec = importlib.util.spec_from_file_location("site_chrome", os.path.join(os.path.dirname(__file__), "..", "scripts", "site.py"))
chrome = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chrome)
loose = ["%s -> %s" % (s["name"], r) for s in skills
         for r in sorted(chrome.refs(read(s["name"] + ".md").decode("utf-8")))
         if not os.path.isfile(os.path.join(ROOT, r))]
ok("skills index: every file a listed skill points at is published", not loose, ", ".join(loose[:5]))
ok("skills index: the index site.py derives", read(".well-known/agent-skills/index.json").decode("utf-8")
   == chrome.skills_index(ROOT))

# A crawler reads only the group that names it (RFC 9309), so every group states
# the Content Signals.
groups = [g for g in re.split(r"\n\s*\n", read("robots.txt").decode("utf-8"))
          if re.search(r"^User-agent:", g, re.M | re.I)]
ok("robots.txt: every group states Content-Signal", groups and all(re.search(r"^Content-Signal: ", g, re.M) for g in groups),
   "%d groups" % len(groups))

# 3-tier progressive disclosure: Tier 0 (llms.txt + catalog.json), Tier 1 (categories/*.{json,md} + search.json),
# Tier 2 (skills/<id>/SKILL.md + skills/<id>.json).
llms_raw = read("llms.txt")
llms_text = llms_raw.decode("utf-8")
ok("tier0 llms.txt: compact (< 16 KB)", 500 < len(llms_raw) < 16384, "%d bytes" % len(llms_raw))
ok("tier0 llms.txt: teaches 3-tier discovery, HIPs, and detect API",
   "/catalog.json" in llms_text and "/categories/" in llms_text
   and "/skills/<id>/SKILL.md" in llms_text and "https://hips.hanzo.ai/llms.txt" in llms_text
   and "/v1/tool/detect" in llms_text)

cat_raw = read("catalog.json")
cat_doc = json.loads(cat_raw)
cats = cat_doc.get("categories") or []
ok("tier0 catalog.json: compact (< 16 KB) with non-empty categories",
   len(cat_raw) < 16384 and len(cats) >= 8 and cat_doc.get("total", 0) > 100,
   "%d bytes, %d categories, %d skills" % (len(cat_raw), len(cats), cat_doc.get("total", 0)))
ok("tier0 catalog.json: category counts sum to total",
   sum(c["count"] for c in cats) == cat_doc["total"])

missing_cats = []
bad_skills = []
for c in cats:
    cj = os.path.join(ROOT, c["json"].lstrip("/"))
    cm = os.path.join(ROOT, c["md"].lstrip("/"))
    if not (os.path.isfile(cj) and os.path.isfile(cm)):
        missing_cats.append(c["slug"])
        continue
    cd = json.loads(open(cj, encoding="utf-8").read())
    if len(cd.get("skills", [])) != c["count"]:
        missing_cats.append(c["slug"] + ":count")
    for s in cd.get("skills", [])[:3]:
        smd = os.path.join(ROOT, s["url"].lstrip("/"))
        sjs = os.path.join(ROOT, s["json"].lstrip("/"))
        fmd = os.path.join(ROOT, s["md"].lstrip("/"))
        if not (os.path.isfile(smd) and os.path.isfile(sjs) and open(smd, "rb").read() == open(fmd, "rb").read()):
            bad_skills.append(s["id"])
ok("tier1 categories: every category json & md exists and matches count", not missing_cats, ", ".join(missing_cats[:5]))
ok("tier2 skills: sampled SKILL.md and .json exist and match flat .md", not bad_skills, ", ".join(bad_skills[:5]))
index_html = read("index.html")
ok("tier0 index.html: compact (< 100 KB, no eager skill dump)",
   len(index_html) < 102400 and b'class="row"' not in index_html, "%d bytes" % len(index_html))

if BASE:
    for rel in DOCS:
        status, ctype, body = get(rel)
        ok("live %s: 200, the committed bytes" % rel, status == 200 and body == read(rel),
           "HTTP %d, %d bytes" % (status, len(body)))
    status, ctype, _ = get("auth.md")
    ok("live auth.md: served as markdown", ctype.startswith("text/markdown"), ctype)

    def answers(url):
        req = urllib.request.Request(url, headers={"User-Agent": "hanzoskills-deploy"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return 0

    hrefs = sorted({t["href"] for e in links for k, v in e.items() if k != "anchor" for t in v})
    with ThreadPoolExecutor(8) as pool:
        dead = [h for h, code in zip(hrefs, pool.map(answers, hrefs)) if code != 200]
    ok("live api-catalog: every link answers 200", not dead, ", ".join(dead))
    def served(s):
        status, _, body = get(s["url"].lstrip("/"))
        return status == 200 and "sha256:" + hashlib.sha256(body).hexdigest() == s["digest"]

    with ThreadPoolExecutor(16) as pool:
        wrong = [s["name"] for s, good in zip(skills, pool.map(served, skills)) if not good]
    ok("live skills: every digest matches the served bytes", not wrong, ", ".join(wrong[:5]))

sys.exit(1 if failed else 0)
