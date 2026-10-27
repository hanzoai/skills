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
