#!/usr/bin/env python3
"""Build hanzoskills.com.

One corpus, two representations. `curl` gets markdown; a browser gets a page.
The markdown is the source of truth and the HTML is derived from it at build
time, so there is no renderer in the request path and nothing to drift.
"""
import os, re, io, json, html, shutil, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "skills")
HIPS = os.environ.get("HIPS_DIR", "/Users/z/work/hanzo/hips/HIPs")
OUT = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "public"))

CATEGORY_LABELS = {
    "standards-core": "Standards: Core AI & Models",
    "standards-infra": "Standards: Cloud & Native Infrastructure",
    "standards-interface": "Standards: Interfaces & Realtime Protocols",
    "standards-security": "Standards: Security, IAM & Cryptography",
    "standards-apps": "Standards: Platform & Applications",
}

# ---------------------------------------------------------------- markdown ---
def render(md):
    out, i, lines = [], 0, md.split('\n')
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            lang = ln[3:].strip()
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith('```'):
                buf.append(lines[i]); i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ''
            out.append(f'<pre><code{cls}>{html.escape(chr(10).join(buf))}</code></pre>')
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', ln)
        if m:
            lvl = len(m.group(1))
            out.append(f'<h{lvl}>{inline(m.group(2))}</h{lvl}>'); i += 1; continue
        if re.match(r'^\s*[-*]\s+', ln):
            items = []
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(inline(re.sub(r'^\s*[-*]\s+', '', lines[i]))); i += 1
            out.append('<ul>' + ''.join(f'<li>{x}</li>' for x in items) + '</ul>')
            continue
        if re.match(r'^\s*\d+\.\s+', ln):
            items = []
            while i < len(lines) and re.match(r'^\s*\d+\.\s+', lines[i]):
                items.append(inline(re.sub(r'^\s*\d+\.\s+', '', lines[i]))); i += 1
            out.append('<ol>' + ''.join(f'<li>{x}</li>' for x in items) + '</ol>')
            continue
        if re.match(r'^\s*>', ln):
            buf = []
            while i < len(lines) and re.match(r'^\s*>', lines[i]):
                buf.append(re.sub(r'^\s*>\s?', '', lines[i])); i += 1
            out.append(f'<blockquote>{render(chr(10).join(buf))}</blockquote>')
            continue
        if re.match(r'^\s*(---+|\*\*\*+)\s*$', ln):
            out.append('<hr>'); i += 1; continue
        if ln.strip() == '':
            i += 1; continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,6}\s|```|\s*[-*]\s|\s*\d+\.\s|\s*>)', lines[i]):
            buf.append(lines[i]); i += 1
        out.append(f'<p>{inline(" ".join(buf))}</p>')
    return '\n'.join(out)

def inline(s):
    s = html.escape(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![*\w])\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', s)
    return s

# ------------------------------------------------------------------ collect ---
def first_sentence(body, limit=240):
    for para in re.split(r'\n\s*\n', body):
        p = para.strip()
        if not p or p.startswith(('#', '|', '```', '---', '<!--', '*Note', '>', '**Category', '**Related')):
            continue
        p = re.sub(r'[*_`\[\]]', '', ' '.join(p.split()))
        m = re.match(r'(.{20,%d}?[.!?])(\s|$)' % limit, p)
        return (m.group(1) if m else p[:limit]).strip()
    return ''

BAD_DESC = re.compile(r'^(go |npm |bd |curl |cargo |pip |brew |git |docker )|[:,]$|^Skill for ', re.I)

def clean_desc(desc, title, body):
    if desc and not BAD_DESC.search(desc) and len(desc) > 24:
        return desc
    return first_sentence(body) or title

def map_hip_category(cat, n, title):
    c = cat.lower() if cat else ''
    num = int(n) if n.isdigit() else 9999
    
    # Core & AI
    if c in ('core', 'meta', 'general', 'governance') or num in (0, 1, 2, 3, 6, 7, 9, 39):
        if num in (4, 10, 11, 12, 15, 38):
            return 'standards-interface'
        return 'standards-core'
    
    # Security & Cryptography
    if c in ('security', 'cryptography') or num in (5, 26, 27, 54, 84, 85, 86, 87, 88, 111, 118, 143):
        return 'standards-security'
        
    # Infrastructure, Cloud & Host
    if c in ('infrastructure', 'bridge') or num in (14, 20, 24, 28, 29, 30, 33, 36, 101, 102, 106, 114, 117):
        return 'standards-infra'
        
    # Protocols & Interfaces
    if c in ('interface',):
        return 'standards-interface'
        
    # Applications, Operator & Platform
    if c in ('application', 'operator', 'platform') or num >= 400:
        return 'standards-apps'
        
    return 'standards-infra'

def collect_hips():
    out = []
    if not os.path.isdir(HIPS):
        return out
    for f in sorted(os.listdir(HIPS)):
        if not f.endswith('.md') or f.startswith('_'):
            continue
        t = io.open(os.path.join(HIPS, f), encoding='utf-8').read()
        m = re.match(r'^---\s*\n(.*?)^---\s*\n', t, re.S | re.M)
        if not m:
            continue
        fm, body = m.group(1), t[m.end():]
        num = re.search(r'^hip:\s*(\S+)', fm, re.M)
        ti = re.search(r'^title:\s*(.+)$', fm, re.M)
        st = re.search(r'^status:\s*(.+)$', fm, re.M)
        cat = re.search(r'^category:\s*(.+)$', fm, re.M)
        if not (num and ti):
            continue
        n = str(num.group(1)).strip().strip('"\'').lstrip('0') or '0'
        title = ti.group(1).strip().strip('"\'')
        status = (st.group(1).strip() if st else '').strip('"\'')
        category_raw = cat.group(1).strip().strip('"\'') if cat else ''
        name = f"hip-{int(n):04d}" if n.isdigit() else re.sub(r'[^a-z0-9-]+', '-', n.lower())
        desc = f"HIP-{n}: {title}." + (f" Status {status}." if status else "") + \
               " Hanzo architectural specification."
        mapped_cat = map_hip_category(category_raw, n, title)
        out.append({'name': name, 'desc': desc, 'cat': mapped_cat,
                    'body': body, 'raw': t, 'is_hip': True})
    return out

def collect_manifest():
    manifest_path = os.path.join(REPO_ROOT, "skill.md")
    if not os.path.isfile(manifest_path):
        return []
    t = io.open(manifest_path, encoding='utf-8').read()
    m = re.match(r'^---\s*\n(.*?)^---\s*\n', t, re.S | re.M)
    fm = m.group(1) if m else ''
    body = t[m.end():] if m else t
    ds = re.search(r'^description:\s*(.+)$', fm, re.M)
    desc = ' '.join(ds.group(1).strip().strip('"\'').split()) if ds else \
        "Onboarding manifest for an AI agent operating Hanzo — native code architecture (Go, Rust, C++), local-first cloud copy, backend migration with Hanzo Base realtime SSE, and working the AI cloud safely."
    return [{'name': 'skill', 'desc': desc, 'cat': 'hanzo', 'body': body, 'raw': t}]

def collect():
    docs = {}
    # 1. Manifest
    for d in collect_manifest():
        docs[d['name']] = d

    # 2. All HIPs organized across 5 architectural domains
    for d in collect_hips():
        docs[d['name']] = d

    # 3. All skills in SRC
    for dp, dn, fn in os.walk(SRC):
        for f in sorted(fn):
            if not f.endswith('.md') or f in ('INDEX.md', 'README.md', 'SECURITY.md', '_SKILL_TEMPLATE.md'):
                continue
            p = os.path.join(dp, f)
            try:
                t = io.open(p, encoding='utf-8').read()
            except Exception:
                continue

            m = re.match(r'^---\s*\n(.*?)^---\s*\n', t, re.S | re.M)
            if m:
                fm = m.group(1)
                nm = re.search(r'^name:\s*(.+)$', fm, re.M)
                ds = re.search(r'^description:\s*(.+)$', fm, re.M)
                name = re.sub(r'[^a-z0-9._-]+', '-', nm.group(1).strip().strip('"\'').lower()).strip('-') if nm else f[:-3].lower()
                desc = ' '.join(ds.group(1).strip().strip('"\'').split()) if ds else ''
                body = t[m.end():]
            else:
                name = f[:-3].lower()
                h1_m = re.search(r'^#\s+(.+)$', t, re.M)
                title = h1_m.group(1).strip() if h1_m else name
                desc = first_sentence(t) or title
                body = t

            rel = os.path.relpath(dp, SRC).split(os.sep)
            cat = rel[0] if rel and rel[0] != '.' else 'core'
            if cat == name:
                cat = 'core'
            if name in ('category-skill-name', '_skill_template'):
                continue

            docs.setdefault(name, {'name': name, 'desc': clean_desc(desc, name, body),
                                   'cat': cat, 'body': body, 'raw': t})
    return sorted(docs.values(), key=lambda d: (d['cat'], d['name']))

# --------------------------------------------------------------------- shell ---
CSS = """
/* Committed to one visual world: a dark instrument panel. No light variant —
   the dot field and the hairline rules only read on black, and a naive
   inversion would wash both out. Every colour is painted explicitly so the page
   never borrows a host background. */
@font-face{font-family:'Zen';src:url('/fonts/Zen-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen';src:url('/fonts/Zen-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Pixel Square';src:url('/fonts/ZenPixel-Square.woff2') format('woff2');font-style:normal;font-display:swap}

:root{
  --ink:#f4f4f2; --soft:#9c9c98; --faint:#6a6a66; --ground:#000; --panel:#0a0a0a;
  --line:#232323; --line-hi:#3a3a3a; --accent:#f4f4f2;
  --font-zen-sans:'Zen',-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --font-zen-mono:'Zen Mono',ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --font-zen-pixel-square:'Zen Pixel Square',monospace;
  --mono:var(--font-zen-mono);
  --sans:var(--font-zen-sans);
  /* Controls are pills and surfaces are soft, as on hanzo.ai. */
  --r:999px; --r-soft:14px;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:60rem;margin:0 auto;padding:0 1.75rem}

.hero{position:relative;border-bottom:1px solid var(--line)}
.hero::before{content:"";position:absolute;inset:0;
  background-image:radial-gradient(var(--line-hi) 1px, transparent 1px);
  background-size:7px 7px;
  -webkit-mask-image:linear-gradient(115deg,#000 0%,rgba(0,0,0,.55) 38%,transparent 72%);
  mask-image:linear-gradient(115deg,#000 0%,rgba(0,0,0,.55) 38%,transparent 72%);
  pointer-events:none}
.hero .wrap{position:relative;padding-top:5.5rem;padding-bottom:4rem}

nav{border-bottom:1px solid var(--line)}
nav .in{max-width:60rem;margin:0 auto;padding:1.1rem 1.75rem;display:flex;
  justify-content:space-between;align-items:center;gap:1rem}
.brand{font-family:var(--mono);font-size:.85rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink);text-decoration:none;
  display:inline-flex;align-items:center;gap:.6rem}
.brand i{font-style:normal;color:var(--faint)}
.mark{width:1.15em;height:1.15em;display:block;flex:0 0 auto}
nav a.up{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--soft);text-decoration:none}
nav a.up:hover{color:var(--ink)}

h1{font-family:var(--mono);font-size:clamp(2rem,5.4vw,3.4rem);line-height:1.12;
  letter-spacing:-.02em;margin:0 0 1.1rem;font-weight:500;text-wrap:balance}
h1 em{font-style:normal;color:var(--faint)}
.lede{color:var(--soft);font-size:1.05rem;max-width:52ch;margin:0 0 2.25rem}

.btns{display:flex;flex-wrap:wrap;gap:.7rem;margin:0 0 .5rem;align-items:center}
.btn{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);
  font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  padding:.8rem 1.4rem;border:1px solid var(--line-hi);border-radius:var(--r);
  color:var(--ink);text-decoration:none;background:transparent;cursor:pointer;
  line-height:1.2;font-weight:400}
.btn:hover{background:#141414;text-decoration:none;border-color:var(--soft)}
.btn.solid{background:var(--ink);color:#000;border-color:var(--ink)}
.btn.solid:hover{background:#d8d8d4}
.btn .chev{width:.7em;height:.7em;flex:0 0 auto;transition:transform .15s ease}
.btn[aria-expanded="true"] .chev{transform:rotate(180deg)}

.strip{border-bottom:1px solid var(--line);background:var(--panel)}
.strip .in{max-width:60rem;margin:0 auto;padding:.9rem 1.75rem;display:flex;
  align-items:center;gap:1.25rem;font-family:var(--mono);font-size:.72rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.strip b{color:var(--soft);font-weight:400;flex:0 0 auto;
  border-right:1px solid var(--line);padding-right:1.25rem;align-self:stretch;
  display:flex;align-items:center}

h2{font-family:var(--mono);font-size:1.1rem;letter-spacing:-.01em;margin:2.75rem 0 .75rem;font-weight:500}
h3{font-size:1rem;margin:2rem 0 .5rem;font-weight:600}
h4,h5,h6{font-size:.94rem;margin:1.5rem 0 .5rem;font-weight:600}
p{margin:0 0 1rem;max-width:70ch}
ul,ol{margin:0 0 1rem;padding-left:1.35rem;max-width:70ch}
li{margin:.22rem 0}
a{color:var(--ink);text-decoration:underline;text-underline-offset:3px;
  text-decoration-color:var(--line-hi)}
a:hover{text-decoration-color:var(--ink)}
a:focus-visible{outline:1px solid var(--ink);outline-offset:3px}
code{font-family:var(--mono);font-size:.85em;background:var(--panel);
  border:1px solid var(--line);border-radius:5px;padding:.1em .36em}
pre{background:var(--panel);border:1px solid var(--line);padding:1rem 1.1rem;
  border-radius:var(--r-soft);overflow-x:auto;margin:0 0 1.25rem}
pre code{background:none;border:0;padding:0;font-size:.82rem;line-height:1.7}
blockquote{margin:0 0 1rem;padding-left:1rem;border-left:1px solid var(--line-hi);color:var(--soft)}
hr{border:0;border-top:1px solid var(--line);margin:2.5rem 0}
table{border-collapse:collapse;width:100%;margin:0 0 1.25rem;font-size:.92rem;
  display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:.5rem .7rem;text-align:left}

.curl{display:block;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r-soft);padding:.95rem 1.1rem;font-family:var(--mono);
  font-size:.8rem;color:var(--soft);overflow-x:auto;margin:0 0 2.5rem;
  white-space:pre;line-height:1.9}
.curl b{color:var(--ink);font-weight:400}
.curl i{font-style:normal;color:var(--faint)}

/* index */
#q{width:100%;padding:.85rem 1.2rem;font-family:var(--mono);font-size:.85rem;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);border-radius:var(--r);margin:0 0 .6rem}
#q::placeholder{color:var(--faint);text-transform:uppercase;letter-spacing:.1em;font-size:.72rem}
#q:focus{outline:none;border-color:var(--soft)}
#count{color:var(--faint);font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;
  text-transform:uppercase;margin:0 0 2rem}
.cat{margin:2.5rem 0 .5rem;font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--faint);border-bottom:1px solid var(--line);
  padding-bottom:.45rem}
.row{display:block;padding:.6rem .9rem;margin:0 -.9rem;border-bottom:1px solid var(--line);
  border-radius:var(--r-soft);text-decoration:none}
.row:hover{background:#0a0a0a}
.nm{font-family:var(--mono);font-size:.84rem;color:var(--ink);display:block}
.ds{color:var(--soft);font-size:.9rem;display:block;margin-top:.12rem}
footer{margin-top:4rem;padding:2rem 0 4rem;border-top:1px solid var(--line);
  color:var(--faint);font-size:.85rem}
footer a{color:var(--soft)}
.fmap{display:grid;gap:1.75rem;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  margin:0 0 2rem;font-size:.85rem}
.fmap b{display:block;font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--faint);font-weight:400;margin:0 0 .6rem;
  border-bottom:1px solid var(--line);padding-bottom:.4rem}
.fmap a{display:inline-block;color:var(--ink);text-decoration:none;margin:.3rem .35rem 0 0}
.fmap a:hover{text-decoration:underline}
.fmap div{color:var(--faint);line-height:1.5}
.fend{border-top:1px solid var(--line);padding-top:1.25rem;font-size:.82rem}
"""

def shell(title, body, up=True, extra='', md=None, hero=None):
    alt = f'<link rel="alternate" type="text/markdown" href="{md}">' if md else ''
    note = (f'<!-- Reading this with a machine? Every page here is also plain markdown.\n'
            f'     This one: https://hanzoskills.com{md}\n'
            f'     Index:    https://hanzoskills.com/llms.txt\n'
            f'     Corpus:   https://hanzoskills.com/all.md\n'
            f'     Append .md to any path for the markdown twin. -->' if md else '')
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="color-scheme" content="dark">\n<style>{CSS}</style>
{alt}
</head>
{note}
<body>
<nav><div class="in">
  <a class="brand" href="/">HANZO<i>SKILLS</i></a>
  {'<a class="up" href="/">All skills</a>' if up else '<a class="up" href="https://hanzo.ai">hanzo.ai</a>'}
</div></nav>
{hero or ''}
<div class="wrap">
{body}
<footer>
<div class="fmap">
  <div><b>Build</b>
    <a href="https://hanzo.app">hanzo.app</a> app builder
    <a href="https://hanzo.codes">hanzo.codes</a> code with an agent
    <a href="https://hanzoskills.com">hanzoskills.com</a> what an agent reads</div>
  <div><b>Run</b>
    <a href="https://console.hanzo.ai">console.hanzo.ai</a> cloud console
    <a href="https://api.hanzo.ai">api.hanzo.ai</a> the API</div>
  <div><b>Account</b>
    <a href="https://hanzo.id">hanzo.id</a> sign in
    <a href="https://pay.hanzo.ai">pay.hanzo.ai</a> top up
    <a href="https://billing.hanzo.ai">billing.hanzo.ai</a> invoices and usage</div>
  <div><b>Company</b>
    <a href="https://hanzo.ai">hanzo.ai</a> canonical
    <a href="https://hanzo.ventures">hanzo.ventures</a> ventures</div>
</div>
<div class="fend">{'<a href="' + md + '">This page as plain text</a> · ' if md else ''}<a href="/llms.txt">Index for agents</a></div>
</footer>
</div>{extra}</body></html>"""

# ---------------------------------------------------------------------- emit ---
def main():
    docs = collect()
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    cats = {}
    for d in docs:
        cats.setdefault(d['cat'], []).append(d)

    # Category ordering:
    # 1. hanzo at the very top (Base, Cloud, Stack, Native DX, Migration)
    # 2. Main developer skill categories ordered by count
    # 3. Standards categories organized logically at the bottom
    standards_order = [
        "standards-core",
        "standards-infra",
        "standards-interface",
        "standards-security",
        "standards-apps",
    ]
    def sort_key(c):
        if c == 'hanzo':
            return (0, 0, '')
        if c.startswith('standards-'):
            idx = standards_order.index(c) if c in standards_order else 99
            return (2, idx, c)
        return (1, -len(cats[c]), c)

    order = sorted(cats, key=sort_key)

    # Emit one page + one markdown file per document
    print(f"Generating {len(docs)} individual pages (.html + .md)...")
    for d in docs:
        io.open(f"{OUT}/{d['name']}.md", 'w', encoding='utf-8').write(d['raw'])
        body = (f"<h1>{html.escape(d['name'])}</h1>"
                f"<p class=\"lede\">{html.escape(d['desc'])}</p>"
                + render(d['body']))
        io.open(f"{OUT}/{d['name']}.html", 'w', encoding='utf-8').write(
            shell(d['name'], body, md=f"/{d['name']}.md"))

    # index rows for index.html
    rows = []
    for c in order:
        cat_label = CATEGORY_LABELS.get(c, c)
        rows.append(f'<div class="cat" data-cat="{html.escape(c)}">{html.escape(cat_label)}</div>')
        for d in sorted(cats[c], key=lambda x: x['name']):
            rows.append(
                f'<a class="row" href="/{d["name"]}" data-k="{html.escape((d["name"]+" "+d["desc"]+" "+c+" "+cat_label).lower())}">'
                f'<span class="nm">{html.escape(d["name"])}</span>'
                f'<span class="ds">{html.escape(d["desc"])}</span></a>')

    js = ("<script>const q=document.getElementById('q'),n=document.getElementById('count'),"
          "R=[...document.querySelectorAll('.row')],C=[...document.querySelectorAll('.cat')];"
          "function f(){const v=q.value.trim().toLowerCase();let s=0;"
          "R.forEach(r=>{const m=!v||r.dataset.k.includes(v);r.style.display=m?'':'none';if(m)s++;});"
          "C.forEach(c=>{let e=c.nextElementSibling,any=false;"
          "while(e&&e.classList.contains('row')){if(e.style.display!=='none')any=true;e=e.nextElementSibling;}"
          "c.style.display=any?'':'none';});"
          "n.textContent=s+(s===1?' skill':' skills');}"
          "q.addEventListener('input',f);f();</script>")

    hero = ('<div class="hero"><div class="wrap">'
            '<h1>The native cloud in a single binary.<br>Go. Rust. C++. '
            '<em>Zero interpreter bloat.</em></h1>'
            '<p class="lede">Run a local copy of the complete cloud, extend with <code>/v1/*</code>, '
            'and migrate your backend to Hanzo Base with realtime native SSE, analytics, and insights. '
            'Built for native performance from local-first to the full Hanzo AI cloud.</p>'
            '<div class="btns">'
            '<a class="btn solid" href="/hanzo-base">Upgrade to Native Go</a>'
            '<a class="btn" href="/app-port">Port an app</a>'
            '<a class="btn" href="/hanzo-coursework-fullstack-go-cloud">Go Coursework</a>'
            '<a class="btn" href="/hanzo-tutorial-realtime-sse-backend">SSE Tutorial</a>'
            '<a class="btn" href="/hanzo-examples-native-cloud">Code Examples</a>'
            '<a class="btn" href="/skill">Read skill.md</a>'
            '<a class="btn" href="/llms.txt">Index for agents</a>'
            '</div></div></div>'
            '<div class="strip"><div class="in"><b>Drop into any agent</b>'
            '<span>curl hanzoskills.com/skill.md</span></div></div>')

    body = (f'<input id="q" type="search" placeholder="Filter {len(docs)} skills, guides & standards"'
            f' autocomplete="off" spellcheck="false">'
            f'<div id="count"></div>' + ''.join(rows))
    io.open(f'{OUT}/index.html', 'w', encoding='utf-8').write(
        shell('Hanzo Skills', body, up=False, extra=js, md='/index.md', hero=hero))

    # markdown index + llms.txt
    mi = [
        "# Hanzo Skills\n\n",
        "The native cloud in a single binary. Go, Rust, C++. Zero interpreter bloat.\n",
        "Run a local copy of the cloud, extend with /v1/*, and migrate your backend to Hanzo Base with realtime native SSE, analytics, and insights.\n\n",
        "    curl hanzoskills.com/llms.txt          # the index, for an agent\n",
        "    curl hanzoskills.com/<skill>.md        # one skill\n",
        "    curl hanzoskills.com/all.md            # the whole corpus, one file\n\n",
        "Every page is `text/markdown`. Paths work with or without the `.md`.\n\n",
        "## Start Here: Migrate to Native Go with Hanzo Base\n\n",
        "- [`hanzo-base`](hanzo-base.md) — Open-source single-binary Go backend with embedded SQLite/Postgres, auth, admin UI, realtime native SSE.\n",
        "- [`app-port`](app-port.md) — Bring an existing app onto this stack: backend to Go, transport to ZAP, runtime out of Python or JavaScript.\n",
        "- [`ui-port`](ui-port.md) — Make a rewritten interface match the one it replaces, by measurement rather than by eye.\n",
        "- [`hanzo-cloud`](hanzo-cloud.md) — Unified cloud platform: host + plugins over ZAP transport.\n",
        "- [`hanzo-stack`](hanzo-stack.md) — Full integrated local development environment (compose.yml).\n",
        "- [`go-sdk`](go-sdk.md) — Generated Go client library for the Hanzo AI API.\n\n",
        "## Everything\n"
    ]

    ll = [
        f"# Hanzo Skills\n\n",
        f"> {len(docs)} skills and architectural specifications for building apps, backends, agents and agentic companies on Hanzo.\n",
        "> Native compiled code only (Go, Rust, C++). Local-first cloud, realtime SSE, zero interpreter bloat. Plain markdown, curl-able, no auth.\n\n",
        "## Upgrade to Native Go with Hanzo Base\n\n",
        "- [hanzo-base](https://hanzoskills.com/hanzo-base.md): Open-source single-binary Go backend with embedded SQLite/Postgres, auth, admin UI, realtime native SSE.\n",
        "- [app-port](https://hanzoskills.com/app-port.md): Bring an existing app onto this stack — backend to Go, transport to ZAP, runtime out of Python or JavaScript.\n",
        "- [cloud-host](https://hanzoskills.com/cloud-host.md): Run the cloud on your own machine — the local API, ports, keys, and what already exists so you do not rebuild it.\n",
        "- [ui-port](https://hanzoskills.com/ui-port.md): Make a rewritten interface match the one it replaces, by measurement rather than by eye.\n",
        "- [hanzo-cloud](https://hanzoskills.com/hanzo-cloud.md): Unified cloud platform: host + plugins over ZAP transport.\n",
        "- [hanzo-stack](https://hanzoskills.com/hanzo-stack.md): Full integrated local development environment (compose.yml).\n\n",
        "## Skills & Standards\n\n"
    ]

    for c in order:
        cat_label = CATEGORY_LABELS.get(c, c)
        mi.append(f"\n### {cat_label}\n")
        ll.append(f"\n### {cat_label}\n")
        for d in sorted(cats[c], key=lambda x: x['name']):
            mi.append(f"- [`{d['name']}`]({d['name']}.md) — {d['desc']}\n")
            ll.append(f"- [{d['name']}](https://hanzoskills.com/{d['name']}.md): {d['desc']}\n")

    ll.append("\n## Optional\n\n- [Whole corpus in one file](https://hanzoskills.com/all.md)\n")
    io.open(f'{OUT}/index.md', 'w', encoding='utf-8').writelines(mi)
    io.open(f'{OUT}/llms.txt', 'w', encoding='utf-8').writelines(ll)

    allp = ["# Hanzo Skills — complete corpus\n\nEvery skill and architectural standard, concatenated.\n"]
    for d in docs:
        allp.append(f"\n\n---\n\n<!-- {d['name']} -->\n\n{d['raw']}")
    io.open(f'{OUT}/all.md', 'w', encoding='utf-8').writelines(allp)

    print(f"Generated index: total_files={len(docs)} categories={len(cats)}")

if __name__ == "__main__":
    main()
