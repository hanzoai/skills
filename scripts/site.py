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
from catalog import BROWSER_JS, emit_progressive, render_index_body  # noqa: E402
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
Set up the complete Hanzo platform in this Claude Code session using native Claude config and progressive disclosure:

1. Progressive instructions: fetch https://hanzoskills.com/CLAUDE.md and append to ./CLAUDE.md (and ~/.claude/CLAUDE.md for every repo).
2. Native Hanzo MCP: run `claude mcp add --scope user hanzo -- npx -y hanzo mcp` (or hosted https://api.hanzo.ai/v1/mcp).
3. Native CLI, Local Lifecycle, Hanzo Dev & Hanzo Bot: install `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo`), sign in with `hanzo auth login` (or headless wallet signup via https://hanzoskills.com/hanzo.md). Local stack lifecycle: `hanzo start`/`stop`/`restart` (runtime: VM, k3s), `hanzo up`/`down` (Cloud & apps; `down` keeps k3s warm), `hanzo ps`. Run `hanzo dev` or `npm i -g @hanzo/bot && hanzo-bot onboard --install-daemon` (cloud: https://hanzo.bot). Build custom apps with `@hanzo/mod` (https://docs.hanzo.ai/docs/apps/mods).
4. Browser Extension (Chrome/Firefox/Safari/Edge): install from https://hanzo.ai/extension (or https://github.com/hanzoai/extension/releases/latest) so MCP browser/computer tools bridge live tabs.
5. Progressive skills (never load /all.md): read https://hanzoskills.com/llms.txt -> /categories/<slug>.md -> save ONLY needed skills to ~/.claude/skills/<name>/SKILL.md.
6. Forge OSS repos: browse https://hanzoskills.com/oss/index.md -> /oss/<org>.md -> clone from https://forge.hanzo.ai/<org>/<repo>.git.

Docs: https://docs.hanzo.ai/docs/skills — everything is plain markdown over https and needs no auth."""),

    ("codex", "Codex", "codex", """\
Set up the complete Hanzo platform in this Codex session using native Codex config and progressive disclosure:

1. Progressive instructions: fetch https://hanzoskills.com/AGENTS.md and append to ./AGENTS.md (and ~/.codex/AGENTS.md for every project).
2. Native Hanzo MCP: run `codex mcp add hanzo -- npx -y hanzo mcp` (or hosted https://api.hanzo.ai/v1/mcp).
3. Native CLI, Local Lifecycle, Hanzo Dev & Hanzo Bot: install `curl -fsSL https://hanzo.sh | sh` (or `npm i -g hanzo`), `hanzo auth login`, `hanzo start`/`stop` (runtime), `hanzo up`/`down`/`ps` (Cloud & apps), `hanzo dev`, and `npm i -g @hanzo/bot` (cloud: https://hanzo.bot). App SDK: `@hanzo/mod` (https://docs.hanzo.ai/docs/apps/mods).
4. Browser Extension: install from https://hanzo.ai/extension (or https://github.com/hanzoai/extension/releases/latest) for live MCP browser/CDP automation.
5. Progressive skills: fetch https://hanzoskills.com/llms.txt -> /categories/<slug>.md -> pull single skills on demand from https://hanzoskills.com/skills/<name>/SKILL.md. Browse Forge OSS via https://hanzoskills.com/oss/index.md.

Docs: https://docs.hanzo.ai/docs/skills — everything is plain markdown over https and needs no auth."""),

    ("cursor", "Cursor", "cursor", """\
Set up the complete Hanzo platform in this Cursor workspace using native Cursor rules, MCP, and progressive disclosure:

1. Progressive instructions: fetch https://hanzoskills.com/CLAUDE.md and write .cursor/rules/hanzo.mdc with front matter:
   ---
   description: Hanzo platform, CLI, MCP, Dev, Bot & progressive skills
   alwaysApply: true
   ---
2. Native Hanzo MCP: add {"mcpServers":{"hanzo":{"command":"npx","args":["-y","hanzo","mcp"]}}} to .cursor/mcp.json (or use hosted https://api.hanzo.ai/v1/mcp).
3. CLI, Hanzo Dev, Hanzo Bot & Browser Extension: `curl -fsSL https://hanzo.sh | sh` (`hanzo auth login`, `hanzo dev`), `npm i -g @hanzo/bot` (https://hanzo.bot), and browser extension from https://hanzo.ai/extension.
4. Progressive skills: browse https://hanzoskills.com/llms.txt -> /categories/<slug>.md (and /oss/index.md for Forge OSS) and add one .cursor/rules/<name>.mdc per needed skill from https://hanzoskills.com/skills/<name>/SKILL.md with alwaysApply: false.

Docs: https://docs.hanzo.ai/docs/skills — everything is plain markdown over https and needs no auth."""),

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
/* Matches hanzo.ai's monochrome editorial aesthetic: light-first white canvas
   with automatic light/dark color-scheme support, Zen sans display headings,
   edge halftone dot framing, pill controls, and crisp dark terminal cards. */
@font-face{font-family:'Zen';src:url('/fonts/Zen-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen';src:url('/fonts/Zen-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Pixel Square';src:url('/fonts/ZenPixel-Square.woff2') format('woff2');font-style:normal;font-display:swap}

:root{
  color-scheme:light dark;
  --ground-light:#ffffff; --ground-dark:#000000;
  --panel-light:#fafafa; --panel-dark:#0a0a0a;
  --panel-2-light:#f5f5f7; --panel-2-dark:#141414;
  --ink-light:#0a0a0a; --ink-dark:#f5f5f7;
  --soft-light:#6e6e73; --soft-dark:#a1a1a6;
  --faint-light:#86868b; --faint-dark:#737378;
  --line-light:#e5e5e5; --line-dark:rgba(255,255,255,0.08);
  --line-hi-light:#d4d4d4; --line-hi-dark:rgba(255,255,255,0.16);
  --solid-bg-light:#0a0a0a; --solid-bg-dark:#ffffff;
  --solid-ink-light:#ffffff; --solid-ink-dark:#0a0a0a;
  --solid-hover-light:#262626; --solid-hover-dark:#e5e5e5;
  --dot-light:#d4d4d4; --dot-dark:#262626;

  --ground:var(--ground-light); --panel:var(--panel-light); --panel-2:var(--panel-2-light);
  --ink:var(--ink-light); --soft:var(--soft-light); --faint:var(--faint-light);
  --line:var(--line-light); --line-hi:var(--line-hi-light);
  --solid-bg:var(--solid-bg-light); --solid-ink:var(--solid-ink-light);
  --solid-hover:var(--solid-hover-light); --dot:var(--dot-light);
  --term-bg:#000000; --term-ink:#f5f5f7; --term-soft:#a1a1a6; --term-faint:#737378; --term-line:rgba(255,255,255,0.08);
  --font-zen-sans:'Zen',-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --font-zen-mono:'Zen Mono',ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --font-zen-pixel-square:'Zen Pixel Square',monospace;
  --mono:var(--font-zen-mono);
  --sans:var(--font-zen-sans);
  --r:999px; --r-soft:16px;
}
@media (prefers-color-scheme: dark){
  :root{
    --ground:var(--ground-dark); --panel:var(--panel-dark); --panel-2:var(--panel-2-dark);
    --ink:var(--ink-dark); --soft:var(--soft-dark); --faint:var(--faint-dark);
    --line:var(--line-dark); --line-hi:var(--line-hi-dark);
    --solid-bg:var(--solid-bg-dark); --solid-ink:var(--solid-ink-dark);
    --solid-hover:var(--solid-hover-dark); --dot:var(--dot-dark);
  }
}
@supports (color: light-dark(white, black)){
  :root{
    --ground:light-dark(var(--ground-light),var(--ground-dark));
    --panel:light-dark(var(--panel-light),var(--panel-dark));
    --panel-2:light-dark(var(--panel-2-light),var(--panel-2-dark));
    --ink:light-dark(var(--ink-light),var(--ink-dark));
    --soft:light-dark(var(--soft-light),var(--soft-dark));
    --faint:light-dark(var(--faint-light),var(--faint-dark));
    --line:light-dark(var(--line-light),var(--line-dark));
    --line-hi:light-dark(var(--line-hi-light),var(--line-hi-dark));
    --solid-bg:light-dark(var(--solid-bg-light),var(--solid-bg-dark));
    --solid-ink:light-dark(var(--solid-ink-light),var(--solid-ink-dark));
    --solid-hover:light-dark(var(--solid-hover-light),var(--solid-hover-dark));
    --dot:light-dark(var(--dot-light),var(--dot-dark));
  }
}
*,*::before,*::after{box-sizing:border-box}
html{background:var(--ground);-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:1rem;line-height:1.58;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility}
.wrap{max-width:68rem;margin:0 auto;padding:0 1.75rem}

/* Organic 3D topological dot matrix behind the hero, matching hanzo.ai. */
.hero{position:relative;overflow:hidden;border-bottom:1px solid var(--line)}
.hz-dots{position:absolute;inset:0;width:100%;height:100%;color:var(--ink);pointer-events:none;z-index:0}
.hero .wrap{position:relative;z-index:1;padding-top:3.25rem;padding-bottom:2.75rem}

.side-mark{display:none}
@media (min-width:1240px){
  .side-mark{position:fixed;left:1.25rem;bottom:2rem;z-index:30;width:2.25rem;height:2.25rem;
    display:flex;align-items:center;justify-content:center;border-radius:var(--r);
    border:1px solid var(--line);background:var(--ground);
    background:color-mix(in srgb,var(--ground) 86%,transparent);backdrop-filter:blur(10px);
    color:var(--ink);opacity:.6;text-decoration:none;transition:opacity .15s ease,border-color .15s ease}
  .side-mark:hover{opacity:1;border-color:var(--line-hi);text-decoration:none}
  .side-mark .mark{width:1rem;height:1rem}
}

nav{position:sticky;top:0;z-index:40;border-bottom:1px solid var(--line);
  background:var(--ground);background:color-mix(in srgb,var(--ground) 90%,transparent);
  backdrop-filter:blur(16px)}
nav .in{max-width:68rem;margin:0 auto;padding:.85rem 1.75rem;display:flex;
  justify-content:space-between;align-items:center;gap:1rem}
.brand{font-family:var(--sans);font-size:.95rem;font-weight:600;letter-spacing:-.02em;
  color:var(--ink);text-decoration:none;display:inline-flex;align-items:center;gap:.55rem}
.brand i{font-style:normal;font-weight:400;color:var(--soft)}
.mark{width:1.25em;height:1.25em;display:block;flex:0 0 auto}
.nlinks{display:flex;align-items:center;gap:1.15rem}
.nlinks a{font-family:var(--sans);font-size:.875rem;font-weight:500;color:var(--soft);text-decoration:none}
.nlinks a:hover{color:var(--ink);text-decoration:none}
nav a.up{font-family:var(--sans);font-size:.875rem;font-weight:500;color:var(--soft);text-decoration:none}
nav a.up:hover{color:var(--ink);text-decoration:none}
.ncta{display:inline-flex;align-items:center;justify-content:center;padding:.45rem 1rem;
  border-radius:var(--r);background:var(--solid-bg);color:var(--solid-ink);font-size:.84rem;
  font-weight:500;text-decoration:none;white-space:nowrap}
.ncta:hover{background:var(--solid-hover);color:var(--solid-ink);text-decoration:none}

h1{font-family:var(--sans);font-size:clamp(2.15rem,4.8vw,3.5rem);line-height:1.06;
  letter-spacing:-.038em;margin:0 0 1.15rem;font-weight:600;text-wrap:balance;max-width:20ch}
h1 em{font-style:normal;color:var(--soft);font-weight:500}
.lede{color:var(--soft);font-size:1.0625rem;line-height:1.55;max-width:58ch;margin:0 0 2rem;text-wrap:pretty}

.btns{display:flex;flex-wrap:wrap;gap:.55rem;margin:0 0 .5rem;align-items:center}
.btn{display:inline-flex;align-items:center;gap:.45rem;font-family:var(--sans);
  font-size:.875rem;font-weight:500;letter-spacing:-.005em;
  min-height:2.5rem;padding:.55rem 1.2rem;border:1px solid var(--line-hi);border-radius:var(--r);
  color:var(--ink);text-decoration:none;background:var(--ground);cursor:pointer;
  line-height:1.2;transition:background .15s ease,border-color .15s ease}
.btn:hover{background:var(--panel-2);text-decoration:none;border-color:var(--ink)}
.btn.solid{background:var(--solid-bg);color:var(--solid-ink);border-color:var(--solid-bg)}
.btn.solid:hover{background:var(--solid-hover);border-color:var(--solid-hover)}
.btn .chev{width:.75em;height:.75em;flex:0 0 auto;transition:transform .15s ease}
.btn[aria-expanded="true"] .chev{transform:rotate(180deg)}

/* Copy-prompt selector. Each row copies the prompt for its own target. */
.copy{position:relative;display:inline-flex}
.menu{position:absolute;top:calc(100% + .5rem);left:0;z-index:20;min-width:21rem;
  background:var(--ground);border:1px solid var(--line-hi);border-radius:var(--r-soft);
  padding:.4rem;display:flex;flex-direction:column;
  box-shadow:0 16px 40px rgba(0,0,0,.18)}
.menu[hidden]{display:none}
.mi{display:flex;align-items:center;gap:.7rem;width:100%;text-align:left;
  padding:.62rem .8rem;border:0;border-radius:calc(var(--r-soft) - .4rem);
  background:transparent;color:var(--ink);font-family:var(--sans);font-size:.86rem;
  font-weight:500;text-decoration:none;cursor:pointer;line-height:1.3;white-space:nowrap}
.mi:hover,.mi:focus-visible{background:var(--panel-2);outline:none;text-decoration:none}
.mi .mark{width:1.05rem;height:1.05rem;color:var(--soft)}
.mi:hover .mark{color:var(--ink)}
.mi u{margin-left:auto;color:var(--faint);text-decoration:none;font-family:var(--mono);font-size:.68rem;
  letter-spacing:.08em;text-transform:uppercase}

/* Works-with bar: one scrolling row, name and mark per agent. */
.strip{border-bottom:1px solid var(--line);background:var(--panel);overflow-x:clip}
.strip .in{max-width:68rem;min-width:0;margin:0 auto;padding:.9rem 1.75rem;display:flex;
  align-items:center;gap:1.25rem;font-family:var(--sans);font-size:.8rem;color:var(--faint)}
.strip b{color:var(--soft);font-weight:500;flex:0 0 auto;font-family:var(--mono);font-size:.7rem;
  letter-spacing:.1em;text-transform:uppercase;
  border-right:1px solid var(--line);padding-right:1.25rem;align-self:stretch;
  display:flex;align-items:center}
.bar{display:flex;align-items:center;gap:.5rem;flex:1 1 auto;min-width:0;
  overflow-x:auto;scrollbar-width:none;scroll-snap-type:x proximity;
  -webkit-overflow-scrolling:touch;
  -webkit-mask-image:linear-gradient(90deg,#000000 calc(100% - 3rem),transparent);
  mask-image:linear-gradient(90deg,#000000 calc(100% - 3rem),transparent)}
.bar::-webkit-scrollbar{display:none}
.agent{display:inline-flex;align-items:center;gap:.45rem;flex:0 0 auto;
  padding:.4rem .9rem;border:1px solid var(--line);border-radius:var(--r);
  color:var(--soft);white-space:nowrap;scroll-snap-align:start;font-size:.82rem;font-weight:500;
  text-decoration:none;background:var(--ground)}
.agent:hover{color:var(--ink);border-color:var(--line-hi);text-decoration:none}
.agent .mark{width:.95rem;height:.95rem}

h2{font-family:var(--sans);font-size:1.35rem;letter-spacing:-.025em;margin:2.75rem 0 .75rem;font-weight:600;text-wrap:balance}
h3{font-size:1.05rem;margin:2rem 0 .5rem;font-weight:600;letter-spacing:-.015em}
h4,h5,h6{font-size:.95rem;margin:1.5rem 0 .5rem;font-weight:600}
p{margin:0 0 1rem;max-width:70ch;text-wrap:pretty}
ul,ol{margin:0 0 1rem;padding-left:1.35rem;max-width:70ch}
li{margin:.25rem 0}
a{color:var(--ink);text-decoration:underline;text-underline-offset:3px;
  text-decoration-color:var(--line-hi)}
a:hover{text-decoration-color:var(--ink)}
a:focus-visible,button:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
code{font-family:var(--mono);font-size:.84em;background:var(--panel-2);
  border:1px solid var(--line);border-radius:6px;padding:.12em .4em;overflow-wrap:anywhere}
pre{color-scheme:dark;background:var(--term-bg);color:var(--term-ink);border:1px solid var(--term-line);
  padding:1.05rem 1.2rem;border-radius:var(--r-soft);overflow-x:auto;margin:0 0 1.25rem}
pre code{background:none;border:0;padding:0;font-size:.84rem;line-height:1.68;color:var(--term-ink)}
blockquote{margin:0 0 1rem;padding-left:1rem;border-left:2px solid var(--line-hi);color:var(--soft)}
hr{border:0;border-top:1px solid var(--line);margin:2.5rem 0}
table{border-collapse:collapse;width:100%;margin:0 0 1.25rem;font-size:.92rem;
  display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:.55rem .75rem;text-align:left}

.curl{color-scheme:dark;display:block;background:var(--term-bg);border:1px solid var(--term-line);
  border-radius:var(--r-soft);padding:1rem 1.2rem;font-family:var(--mono);
  font-size:.82rem;color:var(--term-soft);overflow-x:auto;margin:0 0 2.5rem;
  white-space:pre;line-height:1.85}
.curl b{color:var(--term-ink);font-weight:500}
.curl i{font-style:normal;color:var(--term-faint)}

/* progressive disclosure index */
.tiers{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.85rem;margin:2rem 0 0}
.tier{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-soft);
  padding:1rem 1.15rem;display:flex;flex-direction:column;gap:.45rem}
.tmeta{display:flex;align-items:center;justify-content:space-between;gap:.5rem;
  font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.tmeta b{font-weight:500;color:var(--soft)}
.tier h3{margin:0;font-size:.98rem;font-weight:600;letter-spacing:-.015em;color:var(--ink)}
.tier p{margin:0;font-size:.84rem;line-height:1.48;color:var(--soft)}
.tlinks{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:auto;padding-top:.35rem}
.tlinks a{font-family:var(--mono);font-size:.7rem;color:var(--soft);text-decoration:none;
  padding:.2rem .55rem;border:1px solid var(--line);border-radius:var(--r);background:var(--ground)}
.tlinks a:hover{color:var(--ink);border-color:var(--line-hi);text-decoration:none}
.searchbar{margin:1.75rem 0 1.25rem}
#q{width:100%;padding:.85rem 1.25rem;font-family:var(--sans);font-size:.95rem;color:var(--ink);
  background:var(--panel);border:1px solid var(--line-hi);border-radius:var(--r);margin:0 0 .65rem}
#q::placeholder{color:var(--faint)}
#q:focus{outline:none;border-color:var(--ink);background:var(--ground)}
.statusrow{display:flex;align-items:center;justify-content:space-between;gap:1rem;min-height:1.6rem}
#count{color:var(--faint);font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;
  text-transform:uppercase;margin:0}
.clearbtn{border:1px solid var(--line-hi);background:transparent;color:var(--ink);
  font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;
  padding:.3rem .8rem;border-radius:var(--r);cursor:pointer}
.clearbtn:hover{background:var(--panel-2);border-color:var(--ink)}
.clearbtn[hidden]{display:none}
.catgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(19rem,1fr));gap:.85rem;margin:0 0 1.5rem}
.catgrid[hidden]{display:none}
.catcard{text-align:left;background:var(--panel);border:1px solid var(--line);border-radius:var(--r-soft);
  padding:1.1rem 1.2rem;display:flex;flex-direction:column;gap:.55rem;cursor:pointer;color:var(--ink);
  font-family:var(--sans);transition:border-color .15s ease,background .15s ease}
.catcard:hover{border-color:var(--line-hi);background:var(--panel-2)}
.cathead{display:flex;align-items:center;justify-content:space-between;gap:.75rem;width:100%}
.catname{font-size:.96rem;font-weight:600;letter-spacing:-.015em;color:var(--ink)}
.catcnt{font-family:var(--mono);font-size:.72rem;color:var(--soft);background:var(--ground);
  border:1px solid var(--line);border-radius:var(--r);padding:.14rem .55rem;flex:0 0 auto}
.catdesc{margin:0;font-size:.84rem;line-height:1.45;color:var(--soft)}
.chips{display:flex;flex-wrap:wrap;gap:.32rem;margin-top:.15rem}
.chip{font-family:var(--mono);font-size:.68rem;color:var(--faint);background:var(--ground);
  border:1px solid var(--line);border-radius:6px;padding:.12rem .42rem}
.catfoot{display:flex;align-items:center;justify-content:space-between;width:100%;margin-top:auto;
  padding-top:.45rem;border-top:1px solid var(--line);font-family:var(--mono);font-size:.68rem;
  letter-spacing:.08em;text-transform:uppercase;color:var(--soft)}
.catfmt{color:var(--faint)}
.browser[hidden]{display:none}
.bhead{display:flex;flex-wrap:wrap;align-items:flex-start;justify-content:space-between;gap:1rem;
  margin:0 0 1.15rem;padding-bottom:.9rem;border-bottom:1px solid var(--line)}
.bhead h2{margin:0 0 .25rem;font-size:1.25rem}
.bsub{margin:0;font-size:.9rem;color:var(--soft)}
.blinks{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center}
.slist{display:flex;flex-direction:column;border-top:1px solid var(--line)}
.srow{display:flex;flex-wrap:wrap;align-items:flex-start;justify-content:space-between;gap:.75rem 1.25rem;
  padding:.9rem 0;border-bottom:1px solid var(--line)}
.smain{flex:1 1 24rem;min-width:0}
.shead{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
.nm{font-family:var(--mono);font-size:.88rem;font-weight:600;color:var(--ink);text-decoration:none}
.nm:hover{text-decoration:underline}
.stag{font-family:var(--mono);font-size:.66rem;color:var(--faint);border:1px solid var(--line);
  border-radius:var(--r);padding:.08rem .5rem}
.ds{color:var(--soft);font-size:.9rem;margin:.25rem 0 0;line-height:1.48}
.tags{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.4rem}
.tag{font-family:var(--mono);font-size:.66rem;color:var(--faint);background:var(--panel);
  border:1px solid var(--line);border-radius:6px;padding:.08rem .4rem}
.sact{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;flex:0 0 auto}
.sbtn{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--mono);font-size:.7rem;
  letter-spacing:.04em;color:var(--ink);background:var(--panel);border:1px solid var(--line-hi);
  border-radius:var(--r);padding:.36rem .8rem;text-decoration:none;cursor:pointer;line-height:1.2}
.sbtn:hover{background:var(--panel-2);border-color:var(--ink);text-decoration:none}
.drawer{flex:1 0 100%;margin-top:.45rem;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r-soft);padding:.85rem 1rem;min-width:0}
.drawer[hidden]{display:none}
.dbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.6rem;margin:0 0 .75rem}
.dbar code{font-size:.76rem}
.dacts{display:flex;flex-wrap:wrap;gap:.35rem}
.dpre{margin:0;max-height:24rem;overflow:auto;font-size:.8rem}
.morewrap{display:flex;justify-content:center;margin:1.5rem 0 0}
.loading,.empty{color:var(--soft);font-family:var(--mono);font-size:.82rem;padding:1.5rem 0;margin:0}
.cat{margin:2.5rem 0 .5rem;font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);border-bottom:1px solid var(--line);
  padding-bottom:.45rem}
.row{display:block;padding:.75rem 1rem;margin:0 -1rem;border-bottom:1px solid var(--line);
  border-radius:var(--r-soft);text-decoration:none;transition:background .12s ease;overflow-wrap:anywhere}
.row:hover{background:var(--panel-2);text-decoration:none}
footer{margin-top:4rem;padding:2.25rem 0 4rem;border-top:1px solid var(--line);
  color:var(--faint);font-size:.85rem}
footer a{color:var(--soft);text-decoration:none}
footer a:hover{color:var(--ink);text-decoration:underline}
.fmap{display:grid;gap:1.75rem;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  margin:0 0 2rem;font-size:.85rem}
.fmap b{display:block;font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);font-weight:500;margin:0 0 .6rem;
  border-bottom:1px solid var(--line);padding-bottom:.4rem}
.fmap a{display:inline-block;color:var(--ink);text-decoration:none;margin:.3rem .35rem 0 0}
.fmap a:hover{text-decoration:underline}
.fmap div{color:var(--faint);line-height:1.5}
.fend{border-top:1px solid var(--line);padding-top:1.35rem;font-size:.82rem;
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:1rem}
.fcopy{order:1;color:var(--soft);justify-self:start;white-space:nowrap}
.flegal{order:2;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:.5rem 1.2rem;justify-self:center}
.flegal a{color:var(--soft);text-decoration:none}
.flegal a:hover{color:var(--ink);text-decoration:underline}
.fmark{order:3;display:inline-flex;align-items:center;color:var(--ink);justify-self:end;text-decoration:none}
.fmark:hover{opacity:.75;text-decoration:none}
.fmark .mark{width:1.25rem;height:1.25rem}
.fbar{display:contents}

@media (max-width:860px){.tiers{grid-template-columns:minmax(0,1fr)}}
@media (max-width:640px){
  .hero .wrap{padding-top:2.5rem;padding-bottom:2rem}
  .nlinks .nhide{display:none}
  .btns{gap:.5rem}
  .btn{padding:.55rem 1rem;font-size:.84rem}
  .copy,.copy .btn{width:100%;justify-content:center}
  .menu{left:0;right:0;min-width:0}
  .strip .in{flex-direction:column;align-items:flex-start;gap:.6rem;padding:.9rem 1.75rem}
  .strip b{border-right:0;padding-right:0}
  .bar{width:100%}
  .catgrid{grid-template-columns:minmax(0,1fr)}
  .fend{display:flex;flex-direction:column;align-items:stretch;gap:1rem}
  .flegal{order:1;justify-content:center}
  .fbar{order:2;display:flex;align-items:center;justify-content:space-between;width:100%}
}
"""

CHEV = ('<svg class="chev" viewBox="0 0 12 12" fill="none" stroke="currentColor" '
        'stroke-width="1.6" aria-hidden="true"><path d="M2.5 4.5 6 8l3.5-3.5"/></svg>')


def nav(up_href, up_text):
    return ('<nav><div class="in">\n'
            '  <a class="brand" href="/">Hanzo <i>Skills</i></a>\n'
            '  <div class="nlinks">'
            '<a class="nhide" href="https://hanzo.sh">hanzo.sh</a>'
            '<a class="nhide" href="https://docs.hanzo.ai">Docs</a>'
            '<a class="nhide" href="https://hanzo.ai/security">Security</a>'
            '<a class="up" href="%s">%s</a>'
            '<a class="ncta" href="https://hanzo.ai/signup">Try Hanzo</a>'
            '</div>\n'
            '</div></nav>' % (up_href, up_text))


def side_mark():
    return '<a class="side-mark" href="https://hanzo.ai" aria-label="Hanzo AI">%s</a>' % svg("hanzo")


def fend(md_href):
    return (
        '<div class="fend">'
        '<nav class="flegal" aria-label="Legal and machine links">'
        '<a href="https://hanzo.ai/terms">Terms</a>'
        '<a href="https://hanzo.ai/privacy">Privacy</a>'
        '<a href="https://hanzo.ai/security">Security</a>'
        '<a href="%s">Plain text</a>'
        '<a href="/CLAUDE.md">CLAUDE.md</a>'
        '<a href="/llms.txt">llms.txt</a>'
        '<a href="/catalog.json">catalog.json</a>'
        '<a href="/oss/index.md">OSS Index</a>'
        '<a href="https://forge.hanzo.ai">Forge</a>'
        '<a href="https://docs.hanzo.ai/docs/skills">Docs</a>'
        '<a href="https://hips.hanzo.ai/llms.txt">HIPs</a>'
        '</nav>'
        '<div class="fbar">'
        '<span class="fcopy">© 2026 Hanzo AI, Inc.</span>'
        '<a class="fmark" href="https://hanzo.ai" aria-label="Hanzo AI">%s</a>'
        '</div>'
        '</div>'
        % (html.escape(md_href), svg("hanzo"))
    )


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
        '<a class="btn" href="/CLAUDE.md">CLAUDE.md</a>'
        '<a class="btn" href="/skill">skill.md</a>'
        '<a class="btn" href="/llms.txt">llms.txt</a>'
        '<a class="btn" href="/oss/index.md">OSS Index</a>'
        '<a class="btn" href="https://docs.hanzo.ai/docs/skills">Docs</a>'
        '<a class="btn" href="https://hanzo.sh">hanzo.sh</a>'
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


DOTS_JS = """
<script id="dotsjs">
(function(){
var hero=document.querySelector('.hero');if(!hero)return;
var cv=hero.querySelector('canvas.hz-dots');if(!cv)return;
var cx=cv.getContext('2d');if(!cx)return;
var G=10,R=255,ZS=68,LZ=108;
var mqM=matchMedia('(prefers-reduced-motion: reduce)'),mqD=matchMedia('(prefers-color-scheme: dark)');
var calm=mqM.matches,anim=!calm,t0=performance.now();
var W=0,H=0,cols=0,rows=0,raf=0,vis=true;
var tx=-1e4,ty=-1e4,lx=-1e4,ly=-1e4,inside=false,energy=0,boost=0;
var rgb='10,10,10',aA=[0.095,0.175,0.265],gA=[0.35,0.47,0.59,0.71,0.83,0.94],aS=[],gS=[];
function tone(){
  var m=getComputedStyle(cv).color.match(/[\\d.]+/g)||[10,10,10];
  rgb=m[0]+','+m[1]+','+m[2];
  aS=aA.map(function(a){return 'rgba('+rgb+','+a+')';});
  gS=gA.map(function(a){return 'rgba('+rgb+','+a+')';});
}
tone();
var amb=[[],[],[]],glm=[[],[],[],[],[],[]];
function draw(now){
  var t=anim?(now-t0)*0.00055:0;
  cx.clearRect(0,0,W,H);
  var b,j,i,k;
  for(b=0;b<3;b++)amb[b].length=0;
  for(b=0;b<6;b++)glm[b].length=0;
  var gx1=W*(0.18+0.07*Math.sin(t*0.52)),gy1=H*(0.48+0.14*Math.cos(t*0.41));
  var gx2=W*(0.82+0.07*Math.cos(t*0.47)),gy2=H*(0.52+0.14*Math.sin(t*0.37));
  var invH=1/Math.max(1,H),wL=Math.max(1,W*0.3),wR=Math.max(1,W*0.34);
  var lit=energy>0.01&&lx>-5e3,r2Max=R*R;
  for(j=0;j<rows;j++){
    var y=j*G,rawEy=0.3+y*invH*0.95,ey=rawEy<1?rawEy:1,dey=rawEy<1?invH*0.95:0;
    for(i=0;i<cols;i++){
      var x=i*G,lR=1-x/wL,rR=(x-W*0.66)/wR,ex=0.14,dex=0;
      if(lR>=rR&&lR>0.14){ex=lR;dex=-1/wL;}else if(rR>0.14){ex=rR;dex=1/wR;}
      var p1=x*0.012+y*0.006-t*1.25,p2=y*0.014-x*0.005+t*0.9,p3=(x+y)*0.007+t*0.65;
      var w1=Math.sin(p1),c1=Math.cos(p1),w2=Math.cos(p2),s2=Math.sin(p2),w3=Math.sin(p3),c3=Math.cos(p3);
      var wave=w1*w2*0.65+w3*0.35;
      var dwx=0.65*(0.012*c1*w2+w1*0.005*s2)+0.35*0.007*c3;
      var dwy=0.65*(0.006*c1*w2-w1*0.014*s2)+0.35*0.007*c3;
      var n=0.56+0.44*wave,dnx=0.44*dwx,dny=0.44*dwy;
      var f=ex*ey*n*1.3,dfx=(dex*ey*n+ex*ey*dnx)*1.3,dfy=(ex*dey*n+ex*ey*dny)*1.3;
      var dx1=x-gx1,dy1=y-gy1,d1Sq=dx1*dx1+dy1*dy1;
      if(d1Sq<57600){var d1=Math.sqrt(d1Sq)||1,k1=1-d1/240,gs1=(-0.56*k1)/(240*d1);f+=k1*k1*0.28;dfx+=gs1*dx1;dfy+=gs1*dy1;}
      var dx2=x-gx2,dy2=y-gy2,d2Sq=dx2*dx2+dy2*dy2;
      if(d2Sq<57600){var d2=Math.sqrt(d2Sq)||1,k2=1-d2/240,gs2=(-0.56*k2)/(240*d2);f+=k2*k2*0.28;dfx+=gs2*dx2;dfy+=gs2*dy2;}
      var z=f<0?0:f>1?1:f;if(z<0.05)continue;
      var sx=-dfx*ZS,sy=-dfy*ZS,invN=1/Math.sqrt(sx*sx+sy*sy+1);
      var nx=sx*invN,ny=sy*invN,nz=invN;
      var drawX=x+nx*z*2.6,drawY=y+ny*z*2.6,r=z*G*0.39,gb=-1;
      if(lit){
        var vx=lx-x,vy=ly-y,dSq=vx*vx+vy*vy;
        if(dSq<r2Max){
          var dist2=Math.sqrt(dSq),fall=1-dist2/R,atten=fall*fall*(3-2*fall);
          var vz=LZ-z*44,invL=1/Math.sqrt(dSq+vz*vz);
          var ldx=vx*invL,ldy=vy*invL,ldz=vz*invL;
          var ndotl=Math.max(0,nx*ldx+ny*ldy+nz*ldz),hzf=ldz+1;
          var invHlf=1/Math.sqrt(ldx*ldx+ldy*ldy+hzf*hzf);
          var ndoth=Math.max(0,(nx*ldx+ny*ldy+nz*hzf)*invHlf);
          var nh2=ndoth*ndoth,nh4=nh2*nh2,spec=nh4*nh4*nh4;
          var crest=Math.max(0.06,Math.min(1,(wave+0.72)*0.68))*(0.2+0.8*z);
          var gleam=atten*energy*crest*(0.42*ndotl+1.05*spec);
          var lift=gleam*(0.35+0.65*z);
          r=Math.min(G*0.49,r+lift*G*0.18);
          drawX-=(vx/R)*z*2.4*atten*energy;drawY-=(vy/R)*z*2.4*atten*energy;
          if(gleam>0.07)gb=Math.min(5,Math.floor((gleam-0.07)*7.2));
        }
      }
      if(gb>=0){glm[gb].push(drawX,drawY,r);}
      else{amb[z<0.28?0:z<0.56?1:2].push(drawX,drawY,r);}
    }
  }
  for(b=0;b<3;b++){var arr=amb[b];if(!arr.length)continue;cx.fillStyle=aS[b];cx.beginPath();
    for(k=0;k<arr.length;k+=3){cx.moveTo(arr[k]+arr[k+2],arr[k+1]);cx.arc(arr[k],arr[k+1],arr[k+2],0,6.2832);}cx.fill();}
  for(b=0;b<6;b++){var g=glm[b];if(!g.length)continue;cx.fillStyle=gS[b];cx.beginPath();
    for(k=0;k<g.length;k+=3){cx.moveTo(g[k]+g[k+2],g[k+1]);cx.arc(g[k],g[k+1],g[k+2],0,6.2832);}cx.fill();}
}
function tick(now){
  boost*=0.9;var target=inside?Math.min(1,0.72+boost*0.28):0;
  energy+=(target-energy)*(inside?0.18:0.08);
  if(lx<-5e3&&tx>-5e3){lx=tx;ly=ty;}else if(tx>-5e3){lx+=(tx-lx)*0.28;ly+=(ty-ly)*0.28;}
  draw(now);
  raf=(anim||energy>0.01||inside)&&vis?requestAnimationFrame(tick):0;
}
function start(){if(!raf&&vis&&(anim||energy>0.01||inside))raf=requestAnimationFrame(tick);}
function size(){
  var r=hero.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
  W=r.width;H=r.height;cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  cx.setTransform(dpr,0,0,dpr,0,0);cols=Math.ceil(W/G)+1;rows=Math.ceil(H/G)+1;draw(performance.now());
}
function move(e){
  if(e.pointerType!=='mouse'&&e.pointerType!=='pen')return;
  var r=cv.getBoundingClientRect(),nx=e.clientX-r.left,ny=e.clientY-r.top;
  if(tx>-5e3){boost=Math.min(1,boost+Math.hypot(nx-tx,ny-ty)*0.025);}else{lx=nx;ly=ny;boost=0.6;}
  tx=nx;ty=ny;inside=true;
  if(calm){energy=0.85;lx=nx;ly=ny;draw(performance.now());}else start();
}
function leave(){inside=false;if(calm){energy=0;draw(performance.now());}else start();}
size();start();
if(typeof IntersectionObserver!=='undefined'){
  new IntersectionObserver(function(es){vis=es[0].isIntersecting;if(vis)start();else if(raf){cancelAnimationFrame(raf);raf=0;}}).observe(cv);
}
if(typeof ResizeObserver!=='undefined'){new ResizeObserver(size).observe(hero);}else{addEventListener('resize',size);}
hero.addEventListener('pointerenter',move);
hero.addEventListener('pointermove',move);
hero.addEventListener('pointerleave',leave);
mqD.addEventListener('change',function(){tone();draw(performance.now());});
})();
</script>"""


# The install command per OS, in tab order; the page opens on the visitor's OS.
# Windows is one row here, once hanzo.sh/install.ps1 serves a script:
#   ("win", "Windows", "irm hanzo.sh/install.ps1 | iex"),
INSTALL = [
    ("mac", "macOS", "curl -fsSL https://hanzo.sh | sh"),
    ("linux", "Linux", "curl -fsSL https://hanzo.sh | sh"),
    ("wsl", "WSL", "curl -fsSL https://hanzo.sh | sh"),
]
PASTE = "Set up Hanzo on this machine: follow https://hanzo.sh/setup.md"
PASTE_TO = "Claude Code · Codex · Cursor · Antigravity"
NEXT = "Run <code>hanzo</code> — signs in, wires tools, launches your agent."
AFTER = ("<code>hanzo dev --claude|--codex|--cursor|--agy</code> switches agents; "
         "<code>hanzo -- -c</code> resumes.")

GET_CSS = """
.get{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:1rem;margin:0 0 2rem}
.gbox{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-soft);
  padding:1.15rem 1.25rem;min-width:0}
.ghead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:.5rem .75rem;margin:0 0 .85rem;min-height:1.9rem}
.ghead b{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;
  font-weight:500;color:var(--soft)}
.ghead span{font-family:var(--mono);font-size:.7rem;letter-spacing:.02em;color:var(--faint)}
.tabs{display:inline-flex;gap:.2rem;background:var(--panel-2);border:1px solid var(--line);
  border-radius:var(--r);padding:.18rem}
.tabs[hidden]{display:none}
.tab{border:0;background:transparent;color:var(--soft);font-family:var(--sans);font-size:.75rem;
  font-weight:500;padding:.3rem .8rem;border-radius:var(--r);cursor:pointer}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{background:var(--solid-bg);color:var(--solid-ink)}
.tab:focus-visible,.cp:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.cmd{color-scheme:dark;display:flex;align-items:center;gap:.75rem;background:var(--term-bg);
  border:1px solid var(--term-line);border-radius:12px;padding:.7rem .7rem .7rem 1rem;margin:0 0 .5rem}
.cmd[hidden]{display:none}
.cmd code{flex:1 1 auto;min-width:0;background:none;border:0;padding:0;font-size:.86rem;
  color:var(--term-ink);white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}
.os{flex:0 0 3.4rem;font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--term-faint)}
.get.js .os{display:none}
.cp{display:none;flex:0 0 auto;border:1px solid #3a3a3a;background:transparent;color:var(--term-ink);
  font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;
  padding:.4rem .85rem;border-radius:var(--r);cursor:pointer}
.get.js .cp{display:inline-flex}
.cp:hover{background:#262626;border-color:#a1a1a6}
.next{color:var(--soft);font-size:.875rem;line-height:1.58;margin:.7rem 0 0}
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
        '%s<p class="next">%s</p></div>'
        '<div class="gbox"><div class="ghead"><b>Or paste into your agent</b><span>%s</span></div>'
        '<div class="cmd"><code>%s</code>'
        '<button class="cp" type="button" aria-label="Copy the setup line for your agent">Copy</button>'
        '</div></div>%s</section>'
        % (GET_CSS, tabs, cmds, NEXT, html.escape(PASTE_TO), html.escape(PASTE), GET_JS))


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


def patch(path, catalog_data=None):
    src = open(path, encoding="utf-8").read()
    # Cloudflare adds its analytics beacon at the edge; it is not ours to ship.
    out = BEACON.sub("", src)
    out = re.sub(r"<style>.*?</style>", lambda _: "<style>%s</style>" % CSS, out, count=1, flags=re.S)

    up = re.search(r'<a class="up" href="([^"]+)">([^<]+)</a>', out)
    out = re.sub(r"<nav>.*?</nav>", lambda _: nav(up.group(1), up.group(2)), out, count=1, flags=re.S)
    # Remove any prior side-mark before inserting cleanly after </nav>
    out = re.sub(r'\s*<a class="side-mark"[^>]*>.*?</a>', "", out, flags=re.S)
    out = re.sub(r"</nav>", lambda _: "</nav>\n" + side_mark(), out, count=1)

    name = os.path.basename(path)[:-5]
    slug = "" if name == "index" else name
    md_href = "/%s.md" % name
    # Align footer across all pages: copyright left, legal centered, H mark flush right opposite copyright.
    out = re.sub(r'<div class="fend">.*?</div>(?=\s*</footer>)',
                 lambda _: fend(md_href), out, count=1, flags=re.S)
    # Markdown is a twin file, not content negotiation: /<path>.md.
    out = out.replace(
        "     Send no Accept: text/html and any path returns markdown. -->",
        "     Append .md to any path for the markdown twin. -->")
    # Every page declares color-scheme, names its twin, and loads the WebMCP tools (scripts/webmcp.ts).
    twin = os.path.exists(path[:-5] + ".md")
    head = ('<meta name="color-scheme" content="light dark">\n'
            '<link rel="icon" href="/favicon.svg">\n'
            '<link rel="canonical" href="https://hanzoskills.com/%s">\n' % slug
            + ('<link rel="alternate" type="text/markdown" href="/%s.md">\n' % name if twin else "")
            + '<script type="module" src="/webmcp.js"></script>\n')
    out = re.sub(r'\s*<meta name="color-scheme"[^>]*>', "", out)
    out = re.sub(r'\s*<link rel="(?:icon|canonical|alternate)"[^>]*>', "", out)
    out = re.sub(r'\s*<script type="module" src="/webmcp\.js"></script>', "", out)
    out = re.sub(r"\s*</head>", lambda _: "\n" + head + "</head>", out, count=1)

    if os.path.basename(path) == "index.html":
        # Anchored on markup outside each region so a rewritten page re-matches
        # exactly the same span.
        out = re.sub(r'<canvas class="hz-dots"[^>]*></canvas>', "", out)
        out = re.sub(r'<div class="hero">', '<div class="hero"><canvas class="hz-dots" aria-hidden="true"></canvas>', out, count=1)
        hero_h1_lede = (
            '<h1>Native cloud skills.<br><em>Drop into any agent.</em></h1>'
            '<p class="lede">Progressive disclosure over plain HTTPS. Load <code>/llms.txt</code> (~4 KB), pick a category, fetch one <code>SKILL.md</code> on demand — zero context bloat.</p>'
        )
        out = re.sub(r'<h1>.*?(?=<div class="btns">)', lambda _: hero_h1_lede + get(), out,
                     count=1, flags=re.S)
        out = re.sub(r'<div class="btns">.*?(?=</div></div><div class="strip">)',
                     lambda _: btns(), out, count=1, flags=re.S)
        out = re.sub(r'<div class="strip">.*?(?=\n<div class="wrap">)',
                     lambda _: strip(), out, count=1, flags=re.S)
        if catalog_data is not None:
            out = re.sub(
                r'(<div class="strip">.*?</div></div>\n<div class="wrap">).*?(?=\n<footer>)',
                lambda m: m.group(1) + "\n" + render_index_body(catalog_data),
                out,
                count=1,
                flags=re.S,
            )
        # Strip legacy eager filter script and re-attach modular landing scripts
        out = re.sub(r"</div><script>const q=document\.getElementById\('q'\).*?</script>", "</div>", out, count=1, flags=re.S)
        out = re.sub(r'\s*<script id="copyjs">.*?</script>', "", out, flags=re.S)
        out = re.sub(r'\s*<script id="dotsjs">.*?</script>', "", out, flags=re.S)
        out = re.sub(r'\s*<script id="browserjs">.*?</script>', "", out, flags=re.S)
        out = re.sub(r"\s*</body>", lambda _: COPY_JS + DOTS_JS + BROWSER_JS + "</body>", out, count=1)

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
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
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
    text = text.replace("\r\n", "\n")
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
    catalog_data = emit_progressive(ROOT, SOURCE, get_md())
    files = sorted(glob.glob(os.path.join(ROOT, "*.html")))
    write(os.path.join(ROOT, "sitemap.xml"), sitemap(files))
    changed = sum(patch(f, catalog_data=catalog_data) for f in files)
    index = os.path.join(ROOT, ".well-known", "agent-skills", "index.json")
    os.makedirs(os.path.dirname(index), exist_ok=True)
    write(index, skills_index(ROOT))
    print("%d/%d pages rewritten (%d skills in %d categories)" % (
        changed, len(files), catalog_data["total"], len(catalog_data["categories"])))


if __name__ == "__main__":
    main()
