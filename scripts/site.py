#!/usr/bin/env python3
"""Render the shared chrome of public/*.html: style, nav, hero, agent bar.

Every page carries one style block and one nav. Both are defined here and
written into every file, so the chrome has a single definition. Page bodies are
left untouched. Idempotent: running twice changes nothing, and CI runs it to
assert public/ matches this file.
"""
import glob
import hashlib
import html
import json
import os
import posixpath
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marks import MARK  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")

# The block-H, from @hanzo/logo (MARK_PATHS / MARK_VIEWBOX).
HANZO = ("0 0 67 67", "".join(
    '<path d="%s"/>' % d for d in (
        "M22.21 67V44.6369H0V67H22.21Z",
        "M66.7038 22.3184H22.2534L0.0878906 44.6367H44.4634L66.7038 22.3184Z",
        "M22.21 0H0V22.3184H22.21V0Z",
        "M66.7198 0H44.5098V22.3184H66.7198V0Z",
        "M66.7198 67V44.6369H44.5098V67H66.7198Z",
    )))
MARK["hanzo"] = HANZO

FAVICON = """<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect width="64" height="64" rx="8" fill="#000000"/>
  <g transform="translate(8, 8) scale(0.716)">
%s
  </g>
</svg>
""" % "\n".join('    <path d="%s" fill="#ffffff"/>' % d for d in re.findall(r'd="([^"]+)"', HANZO[1]))


def svg(name, cls="mark"):
    vb, body = MARK[name]
    return ('<svg class="%s" viewBox="%s" fill="currentColor" aria-hidden="true">%s</svg>'
            % (cls, vb, body))


# Every target gets its own prompt: the install path and file format differ per
# tool, so one shared string would be wrong everywhere but one place.
AGENTS = [
    ("claude-code", "Claude Code", "claude", """\
Set up the Hanzo skills in this Claude Code session.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Fetch https://hanzoskills.com/llms.txt for the index of every document.
3. Install a skill by writing https://hanzoskills.com/<name>.md to
   ~/.claude/skills/<name>/SKILL.md — one directory per skill.
4. Verify with: ls ~/.claude/skills

Everything is plain markdown over https and needs no auth."""),

    ("codex", "Codex", "codex", """\
Set up the Hanzo skills in this Codex session.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Fetch https://hanzoskills.com/llms.txt for the index of every document.
3. Codex reads AGENTS.md. Append the manifest to ./AGENTS.md for this repo,
   and to ~/.codex/AGENTS.md to load it in every project.
4. Pull single skills from https://hanzoskills.com/<name>.md and append the
   ones you need under a "Hanzo skills" heading.

Everything is plain markdown over https and needs no auth."""),

    ("cursor", "Cursor", "cursor", """\
Set up the Hanzo skills in this Cursor workspace.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Write .cursor/rules/hanzo.mdc containing the manifest under this front
   matter:
   ---
   description: Hanzo platform skills
   alwaysApply: true
   ---
3. Add one .cursor/rules/<name>.mdc per skill from
   https://hanzoskills.com/<name>.md, with alwaysApply: false so Cursor
   attaches it by description.

Everything is plain markdown over https and needs no auth."""),

    ("copilot", "Copilot", "copilot", """\
Set up the Hanzo skills for GitHub Copilot in this repo.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Copilot reads .github/copilot-instructions.md. Write the manifest there.
3. Add one .github/instructions/<name>.instructions.md per skill from
   https://hanzoskills.com/<name>.md, each with front matter:
   ---
   applyTo: "**"
   ---

Everything is plain markdown over https and needs no auth."""),

    ("gemini-cli", "Gemini CLI", "gemini", """\
Set up the Hanzo skills in this Gemini CLI session.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Gemini CLI reads GEMINI.md. Write the manifest to ./GEMINI.md for this
   repo, or ~/.gemini/GEMINI.md to load it everywhere.
3. Pull single skills from https://hanzoskills.com/<name>.md and add them as
   @-imports so context stays small.

Everything is plain markdown over https and needs no auth."""),

    ("qwen-code", "Qwen Code", "qwen", """\
Set up the Hanzo skills in this Qwen Code session.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Qwen Code reads QWEN.md. Write the manifest to ./QWEN.md for this repo,
   or ~/.qwen/QWEN.md to load it everywhere.
3. Pull single skills from https://hanzoskills.com/<name>.md as you need
   them rather than loading the whole corpus.

Everything is plain markdown over https and needs no auth."""),

    ("anthropic-sdk", "Anthropic SDK", "anthropic", """\
Wire the Hanzo skills into an Anthropic SDK agent.

1. Fetch https://hanzoskills.com/skill.md and put it in the system prompt —
   it is the onboarding manifest.
2. Fetch https://hanzoskills.com/llms.txt and expose it as a tool result so
   the model can choose a document by name.
3. Resolve a choice by fetching https://hanzoskills.com/<name>.md on demand.
   Load per request, not all at once.

Everything is plain markdown over https and needs no auth."""),

    ("openai-sdk", "OpenAI SDK", "openai", """\
Wire the Hanzo skills into an OpenAI SDK agent.

1. Fetch https://hanzoskills.com/skill.md and pass it as a system/developer
   message — it is the onboarding manifest.
2. Fetch https://hanzoskills.com/llms.txt and expose a function tool that
   returns it, so the model can pick a document by name.
3. Have the tool fetch https://hanzoskills.com/<name>.md on demand. Load per
   request, not all at once.

Everything is plain markdown over https and needs no auth."""),

    ("openclaw", "OpenClaw", "openclaw", """\
Set up the Hanzo skills in OpenClaw.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. OpenClaw reads, in order: <workspace>/skills, <workspace>/.agents/skills,
   then ~/.agents/skills. Write each skill to
   ~/.agents/skills/<name>/SKILL.md from https://hanzoskills.com/<name>.md
3. For a managed global copy instead, install with --global, which uses
   ~/.openclaw/skills.

Everything is plain markdown over https and needs no auth."""),

    ("hanzo-bot", "Hanzo Bot", "hanzo", """\
Set up the Hanzo skills in Hanzo Bot.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Write each skill to ~/.hanzo/bot/skills/<name>/SKILL.md from
   https://hanzoskills.com/<name>.md
3. A skills directory in the workspace takes precedence, so put project-only
   skills in <workspace>/skills instead.

Everything is plain markdown over https and needs no auth."""),

    ("hanzo-dev", "Hanzo Dev", "hanzo", """\
Set up the Hanzo skills in Hanzo Dev.

1. Read https://hanzoskills.com/skill.md — the onboarding manifest.
2. Write each skill to ~/.agents/skills/<name>/SKILL.md from
   https://hanzoskills.com/<name>.md
3. For skills that belong to one repo, use <repo>/.agents/skills/<name>/
   instead — dev walks up to the git root and prefers the repo copy.

Everything is plain markdown over https and needs no auth."""),
]

# The dropdown is the short path: the three harnesses people paste into most,
# then the manifest itself. The bar below carries the full set.
MENU = ["claude-code", "codex", "cursor"]

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

/* The dot field. A halftone made of one repeating radial-gradient, faded out
   along the diagonal by a mask so it reads as a wave rather than wallpaper. */
/* No overflow clip here: the dot field is inset to the box already, and a clip
   would cut off the copy-prompt menu. */
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

/* Copy-prompt selector. Each row copies the prompt for its own target. */
.copy{position:relative;display:inline-flex}
.menu{position:absolute;top:calc(100% + .5rem);left:0;z-index:20;min-width:21rem;
  background:var(--panel);border:1px solid var(--line-hi);border-radius:var(--r-soft);
  padding:.4rem;display:flex;flex-direction:column;
  box-shadow:0 12px 32px rgba(0,0,0,.75)}
.menu[hidden]{display:none}
.mi{display:flex;align-items:center;gap:.7rem;width:100%;text-align:left;
  padding:.62rem .8rem;border:0;border-radius:calc(var(--r-soft) - .35rem);
  background:transparent;color:var(--ink);font-family:var(--mono);font-size:.76rem;
  letter-spacing:.04em;text-decoration:none;cursor:pointer;line-height:1.3;
  white-space:nowrap}
.mi:hover,.mi:focus-visible{background:#181818;outline:none}
.mi .mark{width:1.05rem;height:1.05rem;color:var(--soft)}
.mi:hover .mark{color:var(--ink)}
.mi u{margin-left:auto;color:var(--faint);text-decoration:none;font-size:.68rem;
  letter-spacing:.1em;text-transform:uppercase}

/* Works-with bar: one scrolling row, name and mark per agent. */
.strip{border-bottom:1px solid var(--line);background:var(--panel)}
.strip .in{max-width:60rem;margin:0 auto;padding:.9rem 1.75rem;display:flex;
  align-items:center;gap:1.25rem;font-family:var(--mono);font-size:.72rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.strip b{color:var(--soft);font-weight:400;flex:0 0 auto;
  border-right:1px solid var(--line);padding-right:1.25rem;align-self:stretch;
  display:flex;align-items:center}
.bar{display:flex;align-items:center;gap:.5rem;flex:1 1 auto;min-width:0;
  overflow-x:auto;scrollbar-width:none;scroll-snap-type:x proximity;
  -webkit-overflow-scrolling:touch;
  -webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 3rem),transparent);
  mask-image:linear-gradient(90deg,#000 calc(100% - 3rem),transparent)}
.bar::-webkit-scrollbar{display:none}
.agent{display:inline-flex;align-items:center;gap:.45rem;flex:0 0 auto;
  padding:.4rem .85rem;border:1px solid var(--line);border-radius:var(--r);
  color:var(--soft);white-space:nowrap;scroll-snap-align:start;
  text-decoration:none;background:var(--ground)}
.agent:hover{color:var(--ink);border-color:var(--line-hi);text-decoration:none}
.agent .mark{width:.95rem;height:.95rem}

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

@media (max-width:640px){
  .hero .wrap{padding-top:3.25rem;padding-bottom:2.75rem}
  .btns{gap:.5rem}
  .btn{padding:.75rem 1.15rem;font-size:.68rem}
  .copy,.copy .btn{width:100%;justify-content:center}
  .menu{left:0;right:0;min-width:0}
  .strip .in{flex-direction:column;align-items:flex-start;gap:.6rem;padding:.9rem 1.75rem}
  .strip b{border-right:0;padding-right:0}
  .bar{width:100%}
}
"""

CHEV = ('<svg class="chev" viewBox="0 0 12 12" fill="none" stroke="currentColor" '
        'stroke-width="1.6" aria-hidden="true"><path d="M2.5 4.5 6 8l3.5-3.5"/></svg>')


def nav(up_href, up_text):
    return ('<nav><div class="in">\n'
            '  <a class="brand" href="/">%s HANZO<i>SKILLS</i></a>\n'
            '  <a class="up" href="%s">%s</a>\n'
            '</div></nav>' % (svg("hanzo"), up_href, up_text))


def btns():
    rows = []
    for aid, name, mark, _ in AGENTS:
        if aid in MENU:
            rows.append('<button class="mi" role="menuitem" type="button" data-a="%s">'
                        '%s Open in %s<u>copy</u></button>' % (aid, svg(mark), html.escape(name)))
    rows.append('<a class="mi" role="menuitem" href="/skill">%s View skill.md</a>' % svg("hanzo"))
    prompts = {aid: p for aid, _, _, p in AGENTS}
    return (
        '<div class="btns">'
        '<div class="copy">'
        '<button class="btn solid" id="copybtn" type="button" aria-haspopup="menu" '
        'aria-expanded="false" aria-controls="copymenu">Copy setup prompt %s</button>'
        '<div class="menu" id="copymenu" role="menu" aria-labelledby="copybtn" hidden>%s</div>'
        '</div>'
        '<a class="btn" href="/hanzo-base">Upgrade to Native Go</a>'
        '<a class="btn" href="/hanzo-migrate-backend-to-go">Migration Guide</a>'
        '<a class="btn" href="/hanzo-coursework-fullstack-go-cloud">Go Coursework</a>'
        '<a class="btn" href="/hanzo-tutorial-realtime-sse-backend">SSE Tutorial</a>'
        '<a class="btn" href="/hanzo-examples-native-cloud">Code Examples</a>'
        '<a class="btn" href="/llms.txt">Index for agents</a>'
        '<a class="btn" href="https://hanzo.codes">Code with an agent</a>'
        '</div>'
        '<script type="application/json" id="prompts">%s</script>'
        % (CHEV, "".join(rows),
           json.dumps(prompts, ensure_ascii=False).replace("</", "<\\/")))


def strip():
    chips = "".join(
        '<span class="agent">%s %s</span>' % (svg(mark), html.escape(name))
        for _, name, mark, _ in AGENTS)
    return ('<div class="strip"><div class="in"><b>Drop into any agent</b>'
            '<div class="bar">%s</div></div></div>' % chips)


COPY_JS = """
<script id="copyjs">
(function(){
var b=document.getElementById('copybtn'),m=document.getElementById('copymenu'),
    P=JSON.parse(document.getElementById('prompts').textContent);
function open(v){m.hidden=!v;b.setAttribute('aria-expanded',v?'true':'false');}
b.addEventListener('click',function(e){e.stopPropagation();open(m.hidden);});
document.addEventListener('click',function(e){if(!m.contains(e.target))open(false);});
document.addEventListener('keydown',function(e){if(e.key==='Escape')open(false);});
m.addEventListener('click',function(e){
  var t=e.target.closest('.mi[data-a]');if(!t)return;
  var p=P[t.dataset.a],u=t.querySelector('u');
  navigator.clipboard.writeText(p).then(function(){
    u.textContent='copied';setTimeout(function(){u.textContent='copy';},1400);
  });
  open(false);
});
})();
</script>"""


# The install command per OS, in tab order; the page opens on the visitor's OS.
# Windows is one row here, once hanzo.sh/install.ps1 serves a script:
#   ("win", "Windows", "irm hanzo.sh/install.ps1 | iex"),
INSTALL = [
    ("mac", "macOS", "curl -fsSL hanzo.sh | sh"),
    ("linux", "Linux", "curl -fsSL hanzo.sh | sh"),
    ("wsl", "WSL", "curl -fsSL hanzo.sh | sh"),
]
PASTE = "Set up Hanzo on this machine: follow https://hanzo.sh/setup.md"
PASTE_TO = "Claude Code, Codex, Cursor, Antigravity"
NEXT = ("Then run <code>hanzo</code>. It signs you in through the browser, links this machine "
        "to Hanzo Tabs, asks once which coding agent is your favourite, adds Hanzo's tools to it, "
        "and launches it accelerated.")
AFTER = ("After that, <code>hanzo</code> launches your favourite, "
         "<code>hanzo dev --claude|--codex|--cursor|--agy</code> switches it, and "
         "<code>hanzo -- -c</code> continues the last conversation.")

GET_CSS = """
.get{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:1rem;margin:0 0 2.25rem}
.gbox{background:var(--panel);border:1px solid var(--line-hi);border-radius:var(--r-soft);
  padding:1rem 1.1rem;min-width:0}
.ghead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:.5rem .75rem;margin:0 0 .8rem;min-height:1.9rem}
.ghead b{font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;
  font-weight:500;color:var(--soft)}
.ghead span{font-family:var(--mono);font-size:.68rem;letter-spacing:.04em;color:var(--faint)}
.tabs{display:inline-flex;gap:.2rem;border:1px solid var(--line);border-radius:var(--r);padding:.18rem}
.tabs[hidden]{display:none}
.tab{border:0;background:transparent;color:var(--soft);font-family:var(--mono);font-size:.7rem;
  letter-spacing:.06em;padding:.32rem .8rem;border-radius:var(--r);cursor:pointer}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{background:var(--ink);color:#000}
.tab:focus-visible,.cp:focus-visible{outline:1px solid var(--ink);outline-offset:2px}
.cmd{display:flex;align-items:center;gap:.75rem;background:var(--ground);border:1px solid var(--line);
  border-radius:10px;padding:.55rem .55rem .55rem .9rem;margin:0 0 .5rem}
.cmd[hidden]{display:none}
.cmd code{flex:1 1 auto;min-width:0;background:none;border:0;padding:0;font-size:.84rem;
  color:var(--ink);white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}
.os{flex:0 0 3.4rem;font-family:var(--mono);font-size:.6rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}
.get.js .os{display:none}
.cp{display:none;flex:0 0 auto;border:1px solid var(--line-hi);background:transparent;color:var(--ink);
  font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;
  padding:.42rem .85rem;border-radius:var(--r);cursor:pointer}
.get.js .cp{display:inline-flex}
.cp:hover{background:#141414;border-color:var(--soft)}
.next{color:var(--soft);font-size:.86rem;line-height:1.6;margin:.7rem 0 0}
.next code{overflow-wrap:anywhere}
@media (max-width:860px){.get{grid-template-columns:minmax(0,1fr)}}
"""

GET_JS = """
<script id="getjs">
(function(){
var g=document.querySelector('.get'),L=g.querySelector('.tabs'),
    T=[].slice.call(L.querySelectorAll('.tab'));
g.classList.add('js');
function pick(os){T.forEach(function(t){var on=t.dataset.os===os;
  t.setAttribute('aria-selected',on?'true':'false');
  document.getElementById(t.getAttribute('aria-controls')).hidden=!on;});}
var u=navigator.userAgentData,p=((u&&u.platform)||navigator.platform||'')+' '+navigator.userAgent,
    os=/windows|win32|win64/i.test(p)?'win':/mac|iphone|ipad/i.test(p)?'mac':/linux|x11|cros/i.test(p)?'linux':'';
if(os==='win'&&!L.querySelector('[data-os="win"]'))os='wsl';
var f=L.querySelector('[data-os="'+os+'"]')||T[0];
L.insertBefore(f,L.firstChild);L.hidden=false;pick(f.dataset.os);
L.addEventListener('click',function(e){var t=e.target.closest('.tab');if(t)pick(t.dataset.os);});
g.addEventListener('click',function(e){var b=e.target.closest('.cp');if(!b)return;
  navigator.clipboard.writeText(b.previousElementSibling.textContent).then(function(){
    b.textContent='Copied';setTimeout(function(){b.textContent='Copy';},1400);});});
})();
</script>"""


def get():
    tabs = "".join(
        '<button class="tab" role="tab" type="button" id="tab-%s" data-os="%s" '
        'aria-controls="os-%s" aria-selected="false">%s</button>' % (o, o, o, html.escape(n))
        for o, n, _ in INSTALL)
    cmds = "".join(
        '<div class="cmd" id="os-%s" role="tabpanel" aria-labelledby="tab-%s">'
        '<span class="os">%s</span><code>%s</code>'
        '<button class="cp" type="button" aria-label="Copy the %s install command">Copy</button></div>'
        % (o, o, html.escape(n), html.escape(c), html.escape(n))
        for o, n, c in INSTALL)
    return (
        '<section class="get" aria-label="Get started"><style>%s</style>'
        '<div class="gbox"><div class="ghead"><b>Install</b>'
        '<div class="tabs" role="tablist" aria-label="Operating system" hidden>%s</div></div>'
        '%s<p class="next">%s</p><p class="next">%s</p></div>'
        '<div class="gbox"><div class="ghead"><b>Or paste into your agent</b><span>%s</span></div>'
        '<div class="cmd"><code>%s</code>'
        '<button class="cp" type="button" aria-label="Copy the setup line for your agent">Copy</button>'
        '</div></div>%s</section>'
        % (GET_CSS, tabs, cmds, NEXT, AFTER, html.escape(PASTE_TO), html.escape(PASTE), GET_JS))


def get_md():
    rows = {}
    for _, n, c in INSTALL:
        rows.setdefault(c, []).append(n)
    width = max(len(c) for c in rows)
    plain = lambda s: re.sub(r"</?code>", "`", s)
    return ("## Install\n\n"
            + "".join("    %s   # %s\n" % (c.ljust(width), ", ".join(ns)) for c, ns in rows.items())
            + "\n%s %s\n\nOr paste this into your agent (%s):\n\n    %s\n\n"
              "The skills, over plain https:\n\n"
            % (plain(NEXT), plain(AFTER), PASTE_TO, PASTE))


BEACON = re.compile(r'\s*<script[^>]*cloudflareinsights\.com[^>]*>\s*</script>', re.S)


def patch(path):
    src = open(path, encoding="utf-8").read()
    # Cloudflare adds its analytics beacon at the edge; it is not ours to ship.
    out = BEACON.sub("", src)
    out = re.sub(r"<style>.*?</style>", lambda _: "<style>%s</style>" % CSS, out, count=1, flags=re.S)

    up = re.search(r'<a class="up" href="([^"]+)">([^<]+)</a>', out)
    out = re.sub(r"<nav>.*?</nav>", lambda _: nav(up.group(1), up.group(2)), out, count=1, flags=re.S)

    name = os.path.basename(path)[:-5]
    slug = "" if name == "index" else name
    # Markdown is a twin file, not content negotiation: /<path>.md.
    out = out.replace(
        "     Send no Accept: text/html and any path returns markdown. -->",
        "     Append .md to any path for the markdown twin. -->")
    # Every page names its twin, and loads the WebMCP tools (scripts/webmcp.ts).
    twin = os.path.exists(path[:-5] + ".md")
    head = ('<link rel="icon" href="/favicon.svg">\n'
            '<link rel="canonical" href="https://hanzoskills.com/%s">\n' % slug
            + ('<link rel="alternate" type="text/markdown" href="/%s.md">\n' % name if twin else "")
            + '<script type="module" src="/webmcp.js"></script>\n')
    out = re.sub(r'\s*<link rel="(?:icon|canonical|alternate)"[^>]*>', "", out)
    out = re.sub(r'\s*<script type="module" src="/webmcp\.js"></script>', "", out)
    out = re.sub(r"\s*</head>", lambda _: "\n" + head + "</head>", out, count=1)

    if os.path.basename(path) == "index.html":
        # Anchored on markup outside each region so a rewritten page re-matches
        # exactly the same span.
        hero_h1_lede = (
            '<h1>The native cloud in a single binary.<br>Go. Rust. C++. '
            '<em>Zero interpreter bloat.</em></h1>'
            '<p class="lede">Run the open-source parts on your own machine, extend the cloud with <code>/v1/*</code>, '
            'and move your backend to Hanzo Base. <a href="/skill">skill.md</a> gives the command that starts '
            'each part and says what needs the hosted cloud.</p>'
        )
        out = re.sub(r'<h1>.*?(?=<div class="btns">)', lambda _: hero_h1_lede + get(), out,
                     count=1, flags=re.S)
        out = re.sub(r'<div class="btns">.*?(?=</div></div><div class="strip">)',
                     lambda _: btns(), out, count=1, flags=re.S)
        out = re.sub(r'<div class="strip">.*?(?=\n<div class="wrap">)',
                     lambda _: strip(), out, count=1, flags=re.S)
        out = re.sub(r'\s*<script id="copyjs">.*?</script>', "", out, flags=re.S)
        out = re.sub(r"\s*</body>", COPY_JS + "</body>", out, count=1)

    if out != src:
        write(path, out)
        return True
    return False


def write(path, text):
    """Replace a file's contents, never truncate them in place.

    `open(path, "w")` empties the file and then fills it, so anything reading it
    in that window gets zero bytes — and this generator rewrites each page after
    writing it, with the publish reading the same directory. hanzoskills.com went
    out with a 0-byte index.html that way: every other object uploaded intact,
    and the one file this pass rewrites twice did not.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sitemap(files):
    urls = []
    for f in files:
        n = os.path.basename(f)[:-5]
        urls.append("https://hanzoskills.com/" + ("" if n == "index" else n))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "".join("<url><loc>%s</loc></url>\n" % html.escape(u) for u in sorted(urls))
            + "</urlset>\n")


# Agent Skills Discovery v0.2.0 (github.com/cloudflare/agent-skills-discovery-rfc).
# The index lists the skills this site serves whole: a skills/**/SKILL.md whose
# front matter names it <name> (the Agent Skills name rule, a description of at most
# 1,024 characters), served byte for byte as /<name>.md, and every file it points at
# by a relative path published beside it. A skill whose references/, scripts/ or
# templates/ are not published is left out until they are: type skill-md delivers
# SKILL.md alone. Each entry carries the SHA-256 of the bytes served, and the index
# is derived here, beside the sitemap, so CI fails when it disagrees with public/.
SKILLS_SCHEMA = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
BLOCK = re.compile(r"^[|>][+-]?$")
SOURCE = os.path.join(os.path.dirname(ROOT), "skills")
# A markdown link's target, and a bare path into a skill's own directories.
LINK = re.compile(r"\]\(([^)\s]+)")
PATH = re.compile(r"(?<![\w/.:@~-])((?:\.{1,2}/)*(?:references?|scripts|templates|assets|examples|resources|tools|skills)/[\w.@+-]+(?:/[\w.@+-]+)*)")


def front(text):
    """The top-level scalar fields of a file's YAML front matter, as strings.

    Plain, single- and double-quoted scalars and `|`/`>` block scalars: the forms
    the skills here use. A key whose value is a list or mapping maps to "". A plain
    scalar YAML refuses (": " inside it, a trailing ":", a leading indicator) raises
    ValueError, as a YAML loader would refuse the file."""
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 3)
    if end < 0:
        return {}
    lines = text[4:end].split("\n")
    out, i = {}, 0
    while i < len(lines):
        m = re.match(r"^([A-Za-z0-9_-]+):(?:\s+(.*))?$", lines[i])
        i += 1
        if not m:
            continue
        key, val = m.group(1), (m.group(2) or "").strip()
        body = []
        while i < len(lines) and (lines[i][:1] in (" ", "\t") or lines[i] == ""):
            body.append(lines[i])
            i += 1
        while body and body[-1] == "":
            body.pop()
        if BLOCK.match(val):
            ind = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
            rows = [b[ind:] for b in body]
            if val[0] == "|":
                val = "\n".join(rows)
            else:
                val = re.sub(r"(?<!\n)\n(?!\n)", " ", "\n".join(rows)).replace("\n\n", "\n")
        elif val[:1] == '"' and val.endswith('"') and len(val) > 1:
            val = json.loads(val)
        elif val[:1] == "'" and val.endswith("'") and len(val) > 1:
            val = val[1:-1].replace("''", "'")
        else:
            if ": " in val or val.endswith(":") or val[:1] in tuple("*&!%@`"):
                raise ValueError("front matter %s: not a plain YAML scalar" % key)
            if body and val:
                val = " ".join([val] + [b.strip() for b in body])
        out[key] = val
    return out


def refs(text):
    """The relative paths a skill points at, resolved from its own URL, /<name>.md."""
    out = set()
    # A link names a file when it has a directory or an extension; `(URL)` in prose
    # is a placeholder, not a file.
    targets = [m.group(1) for m in LINK.finditer(text) if re.search(r"/|\.\w{1,8}$", m.group(1))]
    targets += [m.group(1) for m in PATH.finditer(text)]
    for target in targets:
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target or target.startswith("/") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue
        out.add(posixpath.normpath("/" + target).lstrip("/"))
    return out


def skills_index(root, source=SOURCE):
    sources = {}
    for path in sorted(glob.glob(os.path.join(source, "**", "SKILL.md"), recursive=True)):
        raw = open(path, "rb").read()
        try:
            name = front(raw.decode("utf-8")).get("name")
        except ValueError:
            continue
        sources.setdefault(name, set()).add(raw)
    skills = []
    for name in sorted(n for n in sources if isinstance(n, str)):
        served = os.path.join(root, name + ".md")
        if not SKILL_NAME.match(name) or len(name) > 64 or not os.path.isfile(served):
            continue
        raw = open(served, "rb").read()
        if raw not in sources[name]:
            continue
        text = raw.decode("utf-8")
        desc = " ".join(front(text).get("description", "").split())
        if not desc or len(desc) > 1024:
            continue
        if any(not os.path.isfile(os.path.join(root, r)) for r in refs(text)):
            continue
        skills.append({
            "name": name,
            "type": "skill-md",
            "description": desc,
            "url": "/%s.md" % name,
            "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        })
    return json.dumps({"$schema": SKILLS_SCHEMA, "skills": skills}, ensure_ascii=False, indent=1) + "\n"


def main():
    write(os.path.join(ROOT, "favicon.svg"), FAVICON)
    files = sorted(glob.glob(os.path.join(ROOT, "*.html")))
    write(os.path.join(ROOT, "sitemap.xml"), sitemap(files))
    changed = sum(patch(f) for f in files)
    md = os.path.join(ROOT, "index.md")
    src = open(md, encoding="utf-8").read()
    out = re.sub(r"(?:## Install\n.*?)?(?=    curl hanzoskills\.com/llms\.txt)",
                 lambda _: get_md(), src, count=1, flags=re.S)
    if out != src:
        write(md, out)
    index = os.path.join(ROOT, ".well-known", "agent-skills", "index.json")
    os.makedirs(os.path.dirname(index), exist_ok=True)
    write(index, skills_index(ROOT))
    print("%d/%d pages rewritten" % (changed, len(files)))


if __name__ == "__main__":
    main()
