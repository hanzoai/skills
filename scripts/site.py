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

# The kit, copied verbatim from ~/Code/hanzo-b/kit/hz-kit.css (Hanzo version B).
# Left out because this site uses neither: section 6 (ledger) and section 9 (stage).
KIT = r"""
/* hz-kit.css · Hanzo version B · product-true, symmetric, still.
   One file, no build step, no dependencies. How to use it: KIT.md, beside this file.
   Put class="hz" on the page container. To follow a host theme, alias the --hz-* colour roles in a
   rule on .hz: the kit declares them at zero specificity, so the host wins whatever the order. */

/* ---------- 1. tokens ---------- */
:where(.hz) {
  /* colour roles, light then dark. light-dark() follows color-scheme: set that on <html>. */
  --hz-page: light-dark(#ffffff, #0a0a0a);
  --hz-ink: light-dark(#0a0a0a, #f5f5f7);
  --hz-soft: light-dark(#525252, #a1a1a6);
  --hz-faint: light-dark(#6e6e73, #8e8e93);
  --hz-card: light-dark(#f5f5f7, #0f0f0f);
  --hz-raised: light-dark(#ffffff, #171717);
  --hz-line: light-dark(rgb(0 0 0 / .1), rgb(255 255 255 / .12));
  --hz-line-strong: light-dark(rgb(0 0 0 / .16), rgb(255 255 255 / .2));
  --hz-on-ink: light-dark(#ffffff, #0a0a0a);

  /* type: one `font` value per step (weight size/line-height family), phone to desktop; tracking beside it */
  --hz-sans: Zen, 'Zen Fallback', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --hz-mono: 'Zen Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --hz-t-display: 560 clamp(40px, 7.2vw, 72px)/1.04 var(--hz-sans); --hz-k-display: -.045em;
  --hz-t-h2: 560 clamp(28px, 4.4vw, 48px)/1.08 var(--hz-sans); --hz-k-h2: -.035em;
  --hz-t-h3: 560 clamp(18px, 2.4vw, 20px)/1.25 var(--hz-sans); --hz-k-h3: -.02em;
  --hz-t-lede: 400 clamp(17px, 1.6vw, 20px)/1.5 var(--hz-sans);
  --hz-t-body: 400 clamp(15px, 4vw, 16px)/1.55 var(--hz-sans);
  --hz-t-small: 400 14px/1.45 var(--hz-sans);
  --hz-t-label: 500 12px/1.2 var(--hz-mono); --hz-k-label: .08em;
  --hz-t-mono: 400 clamp(12px, 3.4vw, 13px)/1.7 var(--hz-mono);

  /* the one spacing scale */
  --hz-s1: 4px; --hz-s2: 8px; --hz-s3: 12px; --hz-s4: 16px; --hz-s5: 24px; --hz-s6: 32px; --hz-s7: 48px; --hz-s8: 64px;
  --hz-section: clamp(72px, 10vw, 144px); /* the air between two sections */
  --hz-wrap: 1200px; --hz-gutter: clamp(16px, 5vw, 24px);

  --hz-r8: 8px; --hz-r12: 12px; --hz-r16: 16px; --hz-r24: 24px; --hz-pill: 999px;
  --hz-ease: cubic-bezier(.16, 1, .3, 1);
}

/* ---------- 2. the scope: a small reset that holds inside any host ---------- */
.hz { background: var(--hz-page); color: var(--hz-ink); font: var(--hz-t-body); -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.hz *, .hz ::before, .hz ::after { box-sizing: border-box; }
.hz, .hz * { font-feature-settings: 'ss01' !important; } /* Zen's house alternates. Pinned, because every `font` shorthand resets it */
.hz :where(h1, h2, h3, p, ul, ol, dl, dd, pre, figure) { margin: 0; padding: 0; }
.hz :where(ul, ol) { list-style: none; }
.hz :where(a), .hz :where(a):hover { color: inherit; text-decoration: none; }
.hz :where(code, pre, kbd) { padding: 0; border-radius: 0; background: none; font-family: var(--hz-mono); font-size: inherit; }
.hz :where(svg) { flex: none; fill: currentColor; }
.hz :focus-visible { outline: 2px solid var(--hz-ink); outline-offset: 3px; }
.hz-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

/* ---------- 3. page wrap, section rhythm, section head ---------- */
.hz-wrap { width: min(100% - 2 * var(--hz-gutter), var(--hz-wrap)); margin-inline: auto; }
.hz-section { padding-block: calc(var(--hz-section) / 2); } /* two neighbours make one full gap */
.hz-head { display: grid; justify-items: center; gap: var(--hz-s4); margin-bottom: clamp(32px, 4vw, 48px); text-align: center; }
.hz-head > :where(h1, h2) { max-width: 22ch; font: var(--hz-t-h2); letter-spacing: var(--hz-k-h2); text-wrap: balance; }
.hz-head > :where(p) { max-width: 56ch; color: var(--hz-soft); font: var(--hz-t-lede); text-wrap: pretty; }
.hz-head-start { justify-items: start; text-align: start; }

/* type steps as classes, for anything outside a section head */
.hz-display { font: var(--hz-t-display); letter-spacing: var(--hz-k-display); }
.hz-h3 { font: var(--hz-t-h3); letter-spacing: var(--hz-k-h3); }
.hz-small { color: var(--hz-soft); font: var(--hz-t-small); }
.hz-label { color: var(--hz-faint); font: var(--hz-t-label); letter-spacing: var(--hz-k-label); text-transform: uppercase; }
.hz-mono { font: var(--hz-t-mono); }

/* ---------- 4. pill buttons: secondary (hairline) is the base; primary is ink; quiet is text ---------- */
.hz-btn { display: inline-flex; align-items: center; justify-content: center; gap: var(--hz-s2); min-height: 44px; padding: 0 22px; border: 1px solid var(--hz-line-strong); border-radius: var(--hz-pill); background: none; color: var(--hz-ink); font: 500 15px/1 var(--hz-sans); white-space: nowrap; cursor: pointer; transition: background-color .2s, border-color .2s, color .2s, scale .15s var(--hz-ease); }
.hz-btn:hover { border-color: var(--hz-ink); color: var(--hz-ink); }
.hz-btn:active { scale: .97; }
.hz-btn-primary, .hz-btn-primary:hover { border-color: transparent; background: var(--hz-ink); color: var(--hz-on-ink); }
.hz-btn-primary:hover { background: color-mix(in srgb, var(--hz-ink) 84%, var(--hz-page)); }
.hz-btn-quiet, .hz-btn-quiet:hover { padding-inline: var(--hz-s3); border-color: transparent; color: var(--hz-soft); }
.hz-btn-quiet:hover { color: var(--hz-ink); text-decoration: underline; text-underline-offset: 4px; }
.hz-btn-sm { min-height: 34px; padding-inline: 14px; font-size: 13px; }
.hz-btn:disabled, .hz-btn[aria-disabled='true'] { opacity: .4; pointer-events: none; }
.hz-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: var(--hz-s3); }
@media (max-width: 479px) { .hz-actions > .hz-btn:not(.hz-btn-quiet) { flex: 1 1 100%; min-height: 48px; } } /* on a phone the actions fill the row, primary on top */

/* ---------- 5. card, and the equal grid that never leaves an orphan ---------- */
.hz-card { display: flex; flex-direction: column; gap: var(--hz-s3); min-width: 0; padding: var(--hz-s5); border: 1px solid var(--hz-line); border-radius: var(--hz-r24); background: var(--hz-card); color: var(--hz-ink); }
.hz-card > :where(h3) { font: var(--hz-t-h3); letter-spacing: var(--hz-k-h3); }
.hz-card > :where(p) { color: var(--hz-soft); }
.hz-card > .hz-tags { margin-top: auto; padding-top: var(--hz-s2); } /* tags sit on the floor of equal-height cards */
a.hz-card { transition: border-color .2s; }
a.hz-card:hover { border-color: var(--hz-ink); color: var(--hz-ink); }
/* Name the grid after the number of items: 2 -> hz-grid-2, 3 or 9 -> hz-grid-3, 4 or 8 -> hz-grid-4, 6 or 12 -> hz-grid-6.
   Each class only ever uses column counts that divide its item count, so the last row is always full. */
.hz-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--hz-s4); }
@media (min-width: 600px) { .hz-grid-2, .hz-grid-4, .hz-grid-6 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (min-width: 768px) { .hz-grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (min-width: 960px) { .hz-grid-6 { grid-template-columns: repeat(3, minmax(0, 1fr)); } .hz-grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); } }

/* ---------- 7. code line, code block, tag ---------- */
.hz-code { display: block; padding: var(--hz-s4) var(--hz-s5); border: 1px solid var(--hz-line); border-radius: var(--hz-r16); background: var(--hz-raised); color: var(--hz-ink); font: var(--hz-t-mono); text-align: start; white-space: pre-wrap; overflow-wrap: anywhere; tab-size: 2; }
.hz-code > :where(span) { display: block; } /* one span per line, so lines can step in */
.hz-code :where(i) { color: var(--hz-faint); font-style: normal; } /* <i> is the quiet voice: a prompt, a comment, output */
.hz-codeline { display: flex; align-items: center; gap: var(--hz-s3); min-height: 48px; padding: 0 6px 0 18px; border: 1px solid var(--hz-line); border-radius: var(--hz-pill); background: var(--hz-raised); font: var(--hz-t-mono); }
.hz-codeline > :where(code) { flex: 1; min-width: 0; overflow-x: auto; white-space: nowrap; scrollbar-width: none; } /* a long command scrolls inside its line: nothing is cut off */
.hz-codeblock { position: relative; }
.hz-codeblock > .hz-code { padding-right: 96px; } /* room for the copy slot */
.hz-codeblock > .hz-copy { position: absolute; top: var(--hz-s2); right: var(--hz-s2); } /* the copy slot: any .hz-btn.hz-btn-sm.hz-copy; the host wires the click */
.hz-tags { display: flex; flex-wrap: wrap; gap: var(--hz-s2); }
.hz-tag { display: inline-flex; align-items: center; gap: 6px; min-height: 24px; padding: 0 10px; border: 1px solid var(--hz-line); border-radius: var(--hz-pill); color: var(--hz-soft); font: 500 12px/1 var(--hz-mono); white-space: nowrap; }
a.hz-tag:hover { border-color: var(--hz-ink); color: var(--hz-ink); }

/* ---------- 8. field: a pill input with an icon slot ---------- */
.hz-field { display: flex; align-items: center; gap: 10px; min-height: 48px; padding: 0 18px; border: 1px solid var(--hz-line-strong); border-radius: var(--hz-pill); background: var(--hz-card); color: var(--hz-faint); cursor: text; }
.hz-field:focus-within { outline: 2px solid var(--hz-ink); outline-offset: 3px; color: var(--hz-ink); }
.hz-field > input, .hz-field > input:focus { flex: 1; min-width: 0; height: 46px; padding: 0; border: 0; outline: 0; background: none; color: var(--hz-ink); font: 400 16px/1 var(--hz-sans); } /* 16px: iOS does not zoom the page */
.hz-field > input::placeholder { color: var(--hz-faint); opacity: 1; }

/* ---------- 10. motion: one gleam, one play, and small responses to what a visitor does ---------- */
/* hz-gleam: a soft light crosses the primary object once, on hover or press. Never loops. */
.hz-gleam { position: relative; overflow: clip; }
.hz-gleam::after { content: ''; position: absolute; inset: 0; pointer-events: none; background: linear-gradient(100deg, transparent 34%, color-mix(in srgb, var(--hz-gleam, var(--hz-ink)) 8%, transparent) 50%, transparent 66%); transform: translateX(-100%); }
.hz-btn-primary.hz-gleam { --hz-gleam: var(--hz-on-ink); }
@keyframes hz-gleam { to { transform: translateX(100%); } }
/* hz-seg: segmented control. The host sets --hz-seg-i (active tab) and --hz-seg-n (tab count) on the track: one thumb, no measuring. */
.hz-seg { position: relative; isolation: isolate; display: grid; grid-template-columns: repeat(var(--hz-seg-n, 2), 1fr); padding: 4px; border: 1px solid var(--hz-line); border-radius: var(--hz-pill); background: var(--hz-card); }
.hz-seg::before { content: ''; position: absolute; z-index: -1; inset: 4px auto 4px 4px; width: calc((100% - 8px) / var(--hz-seg-n, 2)); border-radius: var(--hz-pill); background: var(--hz-ink); transform: translateX(calc(var(--hz-seg-i, 0) * 100%)); }
.hz-seg > * { min-height: 36px; border: 0; border-radius: var(--hz-pill); background: none; color: var(--hz-soft); font: 500 14px/1 var(--hz-sans); cursor: pointer; }
.hz-seg > :is([aria-selected='true'], [aria-pressed='true']) { color: var(--hz-on-ink); }
/* hz-fold: a native <details>; the summary is the control, a CSS chevron replaces the triangle. */
.hz-fold > summary { display: flex; align-items: center; justify-content: space-between; gap: var(--hz-s3); list-style: none; cursor: pointer; }
.hz-fold > summary::-webkit-details-marker { display: none; }
.hz-fold > summary::after { content: ''; width: 8px; height: 8px; border: solid currentColor; border-width: 0 1.5px 1.5px 0; rotate: 45deg; }
.hz-fold[open] > summary::after { rotate: 225deg; }
.hz-arrow { display: inline-block; } /* translate does not apply to a plain inline box */
@media (pointer: coarse) { .hz-seg > *, .hz-fold > summary { min-height: 44px; } }
/* hz-play: the hero object plays its real example once; each child steps in, in order, then the page is still. */
@keyframes hz-step { from { opacity: 0; transform: translateY(6px); } }
@keyframes hz-rise { from { opacity: 0; transform: translateY(8px); } }
@keyframes hz-pop { from { transform: scale(.8); } }
@media (prefers-reduced-motion: no-preference) {
  .hz-gleam:hover::after, .hz-gleam:active::after { animation: hz-gleam .9s var(--hz-ease); }
  .hz-play > * { animation: hz-step .5s var(--hz-ease) backwards; animation-delay: calc(var(--hz-play-at, .2s) + var(--hz-i, 0) * .16s); }
  .hz-play > :nth-child(2) { --hz-i: 1; } .hz-play > :nth-child(3) { --hz-i: 2; } .hz-play > :nth-child(4) { --hz-i: 3; }
  .hz-play > :nth-child(5) { --hz-i: 4; } .hz-play > :nth-child(6) { --hz-i: 5; }
  .hz-play > .hz-label { animation: none; } /* a label is furniture: it is there from the start */
  /* hz-arrive: on the hero container; the first six children rise once on load. Backwards fill: with animation off the HTML is the end state. */
  .hz-arrive > :nth-child(-n+6) { animation: hz-rise .5s var(--hz-ease) backwards; animation-delay: calc(var(--hz-i, 0) * 60ms); }
  :is(.hz-arrive, .hz-fold) > :nth-child(2) { --hz-i: 1; } :is(.hz-arrive, .hz-fold) > :nth-child(3) { --hz-i: 2; } :is(.hz-arrive, .hz-fold) > :nth-child(4) { --hz-i: 3; }
  :is(.hz-arrive, .hz-fold) > :nth-child(5) { --hz-i: 4; } :is(.hz-arrive, .hz-fold) > :nth-child(6) { --hz-i: 5; } :is(.hz-arrive, .hz-fold) > :nth-child(7) { --hz-i: 6; }
  .hz-seg::before { transition: transform .3s var(--hz-ease); } .hz-seg > * { transition: color .3s var(--hz-ease); }
  .hz-swap:is([data-hz-swap], .is-in) { animation: hz-step .22s var(--hz-ease) backwards; } /* plays when the node is inserted or the attribute is set again */
  .hz-fold { interpolate-size: allow-keywords; } .hz-fold > summary::after { transition: rotate .3s var(--hz-ease); }
  .hz-fold::details-content { height: 0; overflow: clip; }
  /* only opening eases; closing is instant, because a tile that leaves its open layout mid-ease flashes */
  .hz-fold[open]::details-content { height: auto; transition: height .3s var(--hz-ease), content-visibility .3s; transition-behavior: allow-discrete; } /* unsupported: opens instantly */
  .hz-fold[open] > :nth-child(n+2):nth-child(-n+7) { animation: hz-step .3s var(--hz-ease) backwards; animation-delay: calc((var(--hz-i, 1) - 1) * 40ms); }
  :is(.is-copied, [data-copied='true']) > :where(svg, .hz-ico) { animation: hz-pop .2s var(--hz-ease); } /* the host swaps the icon to a tick, holds 1.6s, removes the state */
}
/* touch: hover and keyboard focus, fine pointers only */
@media (hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference) {
  a.hz-card, .hz-card.hz-lift, .hz-arrow { transition: translate .2s var(--hz-ease), border-color .2s; }
  :is(a.hz-card, .hz-card.hz-lift):is(:hover, :focus-visible) { translate: 0 -2px; border-color: var(--hz-ink); }
  :is(a, button):is(:hover, :focus-visible) .hz-arrow { translate: 3px 0; }
}

/* ---------- 11. touch: every control is at least 44 x 44 ---------- */
@media (pointer: coarse) { .hz-btn, a.hz-tag, a.hz-item, .hz-copy { min-width: 44px; min-height: 44px; } }

/* ---------- 12. reduced motion: every state still changes, nothing moves (also covers transitions declared outside section 10) ---------- */
@media (prefers-reduced-motion: reduce) { .hz, .hz *, .hz ::before, .hz ::after { transition: none !important; animation: none !important; scroll-behavior: auto !important; } }
"""

CSS = """
/* Light and dark follow the device. Colour comes only from the --hz-* roles. */
@font-face{font-family:'Zen';src:url('/fonts/Zen-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen';src:url('/fonts/Zen-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Variable.woff2') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Zen Mono';src:url('/fonts/ZenMono-Italic-Variable.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Zen Pixel Square';src:url('/fonts/ZenPixel-Square.woff2') format('woff2');font-style:normal;font-display:swap}
""" + KIT + r"""
/* Version B. The kit above is the system; this layer only composes it for the
   site's own markup, with --hz-* tokens and nothing else. */
html{color-scheme:light dark;-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh}
.hz a{overflow-wrap:anywhere}
.hz :not(pre,.hz-code)>code{box-decoration-break:clone;-webkit-box-decoration-break:clone;font-size:.88em;background:var(--hz-raised);border:1px solid var(--hz-line);border-radius:6px;padding:.08em .36em;overflow-wrap:anywhere}
.mark{width:1.15em;height:1.15em;display:block}

/* nav */
nav{border-bottom:1px solid var(--hz-line)}
nav .in{display:flex;justify-content:space-between;align-items:center;gap:var(--hz-s4);min-height:64px}
.brand{display:inline-flex;align-items:center;gap:var(--hz-s3);min-height:44px;font:560 15px/1 var(--hz-sans);letter-spacing:.06em}
.brand i{font-style:normal;color:var(--hz-faint)}

/* hero: centred, symmetric, still */
.hero .wrap{display:grid;justify-items:center;padding-block:clamp(48px,8vw,96px) clamp(40px,6vw,72px);text-align:center}
.hero h1{margin:0 0 var(--hz-s4);font:var(--hz-t-h2);letter-spacing:var(--hz-k-h2);text-wrap:balance}
.hero h1 em{font-style:normal;color:var(--hz-faint)}
.lede{max-width:56ch;color:var(--hz-soft);font:var(--hz-t-lede);text-wrap:pretty}
.hero .lede{margin:0 0 var(--hz-s7)}
.lede a{text-decoration:underline;text-underline-offset:3px;text-decoration-color:var(--hz-line-strong)}
.lede a:hover{text-decoration-color:var(--hz-ink)}
.get{width:100%;margin:0 0 var(--hz-s6);text-align:start}
.ghead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:var(--hz-s2) var(--hz-s3);min-height:34px}
.ghead b{font:var(--hz-t-h3);letter-spacing:var(--hz-k-h3)}
.ghead span{color:var(--hz-soft);font:var(--hz-t-small)}
.tabs[hidden]{display:none}
.gbox .hz-seg{background:var(--hz-page)}
.get .cmd{display:flex;flex-direction:column;justify-content:center;min-height:64px;padding-right:6.5rem}
.get .cmd[id] code{white-space:nowrap}
@media (max-width:479px){.get .cmd{padding:0 6rem 0 var(--hz-s4)}.hz-field{padding-inline:14px;gap:8px}}
@media (max-width:399px){.hz-field>svg{display:none}}
.get .cmd[hidden]{display:none}
.cmd code{display:block;overflow-wrap:anywhere}
.get.js .os{display:none}
.get .cp{display:none;top:50%;translate:0 -50%}
.get.js .cp{display:inline-flex}
.next{font:var(--hz-t-small)}
.hz-gleam{overflow-clip-margin:6px}
.btns{display:grid;justify-items:center;gap:var(--hz-s4);width:100%}
.nexts{max-width:72ch;margin:var(--hz-s5) auto var(--hz-s6);display:grid;gap:var(--hz-s3)}
.nexts p{color:var(--hz-soft)}
.copy{position:relative;display:inline-flex}
.btn .chev{width:.7em;height:.7em;fill:none;transition:transform .15s}
.btn[aria-expanded="true"] .chev{transform:rotate(180deg)}
.menu{position:absolute;top:calc(100% + var(--hz-s2));left:0;z-index:20;min-width:21rem;display:flex;flex-direction:column;padding:var(--hz-s2);border:1px solid var(--hz-line-strong);border-radius:var(--hz-r16);background:var(--hz-raised);box-shadow:0 12px 32px rgb(0 0 0/.18);text-align:start}
.menu[hidden]{display:none}
.mi{display:flex;align-items:center;gap:var(--hz-s3);width:100%;min-height:44px;padding:0 var(--hz-s3);border:0;border-radius:var(--hz-r12);background:none;color:var(--hz-ink);font:var(--hz-t-small);text-align:start;white-space:nowrap;cursor:pointer}
.mi:hover,.mi:focus-visible{background:var(--hz-card)}
.mi .mark{width:16px;height:16px;color:var(--hz-soft)}
.mi u{margin-left:auto;color:var(--hz-faint);text-decoration:none}
.strip{border-top:1px solid var(--hz-line)}
.strip .in{display:grid;justify-items:center;gap:var(--hz-s4);padding-block:var(--hz-s6)}
.strip b{color:var(--hz-faint);font:var(--hz-t-label);letter-spacing:var(--hz-k-label);text-transform:uppercase}
.bar{display:flex;flex-wrap:wrap;justify-content:center;gap:var(--hz-s2);max-width:820px}
.agent{min-height:32px;padding-inline:12px;transition:border-color .2s}
@media (hover:hover) and (pointer:fine){.agent:hover{border-color:var(--hz-ink);color:var(--hz-ink)}}
.get .cp{min-width:5rem}.cp .hz-ico{display:inline-block}
.agent .mark{width:14px;height:14px}
@media (max-width:479px){
  .hero .wrap{padding-top:var(--hz-s6)}
  .btns,.copy{width:100%}.copy .btn{width:100%}
  .more{display:block;width:100%;border-bottom:1px solid var(--hz-line);text-align:start}
  .more>a.hz-btn{display:flex;justify-content:space-between;width:100%;min-height:48px;padding:0;border:0;border-top:1px solid var(--hz-line);border-radius:0;font-size:15px;color:var(--hz-ink)}
  .more>a.hz-btn::after{content:'';width:7px;height:7px;margin-right:4px;border:solid var(--hz-soft);border-width:1.5px 1.5px 0 0;rotate:45deg}
  .more>a.hz-btn:active{scale:1}
  .menu{right:0;min-width:0}
}

/* catalogue: find, index, sections */
.find{position:sticky;top:0;z-index:10;display:grid;gap:var(--hz-s2);margin-block:var(--hz-s7) var(--hz-s5);padding-block:var(--hz-s2);background:var(--hz-page)}
.find .hz-field{flex:1;min-width:0}
.hz-field svg{width:16px;height:16px;fill:none}
#count{justify-self:end;color:var(--hz-faint);font:var(--hz-t-label);letter-spacing:var(--hz-k-label);text-transform:uppercase}
.catgrid{align-items:start;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 var(--hz-s6)}
@media (min-width:960px){.catgrid{grid-template-columns:repeat(3,minmax(0,1fr))}}
.catstd{grid-template-columns:minmax(0,1fr);margin-top:var(--hz-s7)}
.sec{min-width:0;border-bottom:1px solid var(--hz-line);scroll-margin-top:88px}
.sec[hidden]{display:none}
.sec[open]{grid-column:1/-1;border:1px solid var(--hz-line);border-radius:var(--hz-r16);background:var(--hz-card);margin-block:var(--hz-s3)}
.sec>summary{display:flex;align-items:center;justify-content:space-between;gap:var(--hz-s3);min-height:48px;padding:0 var(--hz-s2);list-style:none;cursor:pointer}
.sec[open]>summary{padding-inline:var(--hz-s4)}
@media (max-width:599px){.sec>summary{min-height:56px}}
.sec::details-content{display:grid;grid-template-columns:minmax(0,1fr);gap:var(--hz-s4);padding:0 var(--hz-s4) var(--hz-s4)}
@media (min-width:600px){.sec::details-content{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (min-width:960px){.sec::details-content{grid-template-columns:repeat(3,minmax(0,1fr))}}
@supports not selector(::details-content){.sec[open]{display:grid;gap:var(--hz-s4);padding-bottom:var(--hz-s4);grid-template-columns:repeat(auto-fill,minmax(16rem,1fr))}.sec[open]>summary{grid-column:1/-1}.sec[open]>.row{margin-inline:var(--hz-s4)}}
.filtering .sec::details-content,.filtering .sec>summary::after{transition:none}
/* kit transitions (.hz-btn, a.hz-card) are not gated by the kit; here they are, until it is synced */
@media (prefers-reduced-motion:reduce){.hz-btn,.hz-card,a.hz-card,.btn .chev,.agent{transition:none!important}}.filtering .sec>*{animation:none!important}
.cat{margin:0;font:var(--hz-t-body);font-weight:560;overflow-wrap:break-word;hyphens:manual}
.row.hz-card{gap:var(--hz-s2);padding:var(--hz-s4) var(--hz-s5);border-radius:var(--hz-r16);background:var(--hz-page)}
.nm{font:var(--hz-t-mono);font-weight:560;overflow-wrap:anywhere}
.ds{color:var(--hz-soft);font:var(--hz-t-small)}

/* content pages: Zen title, a 62ch measure, kit code blocks */
.doc{--hz-wrap:46rem;padding-block:var(--hz-s7) 0}
.hz .doc h1{margin:0 0 var(--hz-s4);font:var(--hz-t-h2);letter-spacing:var(--hz-k-h2);text-wrap:balance;overflow-wrap:anywhere}
.hz .doc h2{margin:var(--hz-s7) 0 var(--hz-s3);font:var(--hz-t-h3);letter-spacing:var(--hz-k-h3)}
.hz .doc :where(h3,h4,h5,h6){margin:var(--hz-s5) 0 var(--hz-s2);font:var(--hz-t-body);font-weight:560}
.hz .doc :where(p,ul,ol,blockquote){max-width:62ch;margin-bottom:var(--hz-s4)}
.hz .doc ul{list-style:disc;padding-left:1.35em}
.hz .doc ol{list-style:decimal;padding-left:1.35em}
.doc li{margin:.25em 0}
.hz .doc :where(p,li,td,blockquote) a{text-decoration:underline;text-underline-offset:3px;text-decoration-color:var(--hz-line-strong)}
.hz .doc :where(p,li,td,blockquote) a:hover{text-decoration-color:var(--hz-ink)}
.doc .lede{margin-bottom:var(--hz-s5)}
.doc pre.hz-code{margin-bottom:var(--hz-s5)}
.doc pre code{font-size:inherit}
.doc blockquote{padding-left:var(--hz-s4);border-left:1px solid var(--hz-line-strong);color:var(--hz-soft)}
.doc hr{border:0;border-top:1px solid var(--hz-line);margin:var(--hz-s7) 0}
.doc table{display:block;overflow-x:auto;width:100%;margin-bottom:var(--hz-s5);border-collapse:collapse;font:var(--hz-t-small)}
.doc :where(th,td){padding:var(--hz-s2) var(--hz-s3);border:1px solid var(--hz-line);text-align:start}

footer{margin-top:var(--hz-section);padding-block:var(--hz-s7) var(--hz-s8);border-top:1px solid var(--hz-line);color:var(--hz-faint);font:var(--hz-t-small)}
.fmap{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:var(--hz-s6);margin-bottom:var(--hz-s7)}
.fmap>div{display:grid;grid-template-columns:auto 1fr;gap:var(--hz-s1) var(--hz-s3);align-content:start}
.fmap b{grid-column:1/-1;margin-bottom:var(--hz-s2);color:var(--hz-ink);font-weight:560}
.fmap a{color:var(--hz-ink)}
.fmap a:hover,.fend a:hover{text-decoration:underline;text-underline-offset:3px}
.fend{padding-top:var(--hz-s5);border-top:1px solid var(--hz-line)}
.fend a{color:var(--hz-soft)}
@media (max-width:640px){.fmap a,.fend a{display:inline-flex;align-items:center;min-height:44px}}
"""

CHEV = ('<svg class="chev" viewBox="0 0 12 12" fill="none" stroke="currentColor" '
        'stroke-width="1.6" aria-hidden="true"><path d="M2.5 4.5 6 8l3.5-3.5"/></svg>')


def nav(up_href, up_text):
    return ('<nav><div class="in hz-wrap">\n'
            '  <a class="brand" href="/">%s HANZO<i>SKILLS</i></a>\n'
            '  <a class="up hz-btn hz-btn-quiet" href="%s">%s</a>\n'
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
        '<button class="btn solid hz-btn hz-btn-primary" id="copybtn" type="button" aria-haspopup="menu" '
        'aria-expanded="false" aria-controls="copymenu">Copy setup prompt %s</button>'
        '<div class="menu" id="copymenu" role="menu" aria-labelledby="copybtn" hidden>%s</div>'
        '</div>'
        '<div class="more hz-actions">'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/hanzo-base">Upgrade to Native Go</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/hanzo-migrate-backend-to-go">Migration Guide</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/hanzo-coursework-fullstack-go-cloud">Go Coursework</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/hanzo-tutorial-realtime-sse-backend">SSE Tutorial</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/hanzo-examples-native-cloud">Code Examples</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="/llms.txt">Index for agents</a>'
        '<a class=\"btn hz-btn hz-btn-sm\" href="https://hanzo.codes">Code with an agent</a>'
        '</div></div>'
        '<script type="application/json" id="prompts">%s</script>'
        % (CHEV, "".join(rows),
           json.dumps(prompts, ensure_ascii=False).replace("</", "<\\/")))


def strip():
    chips = "".join(
        '<span class="agent hz-tag">%s %s</span>' % (svg(mark), html.escape(name))
        for _, name, mark, _ in AGENTS)
    return ('<div class="strip"><div class="in hz-wrap"><b>Drop into any agent</b>'
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

GET_JS = """
<script id="getjs">
(function(){
var g=document.querySelector('.get'),L=g.querySelector('.tabs'),
    T=[].slice.call(L.querySelectorAll('.tab'));
g.classList.add('js');
function pick(os,anim){T.forEach(function(t,i){var on=t.dataset.os===os,p=document.getElementById(t.getAttribute('aria-controls'));
  t.setAttribute('aria-selected',on?'true':'false');
  if(on){L.style.setProperty('--hz-seg-i',i);p.hidden=false;if(anim)p.setAttribute('data-hz-swap','');}else p.hidden=true;});}
var u=navigator.userAgentData,p=((u&&u.platform)||navigator.platform||'')+' '+navigator.userAgent,
    os=/windows|win32|win64/i.test(p)?'win':/mac|iphone|ipad/i.test(p)?'mac':/linux|x11|cros/i.test(p)?'linux':'';
if(os==='win'&&!L.querySelector('[data-os="win"]'))os='wsl';
var f=L.querySelector('[data-os="'+os+'"]')||T[0];
L.hidden=false;pick(f.dataset.os);
L.addEventListener('click',function(e){var t=e.target.closest('.tab');if(t)pick(t.dataset.os,true);});
g.addEventListener('click',function(e){var b=e.target.closest('.cp');if(!b)return;
  navigator.clipboard.writeText(b.previousElementSibling.textContent).then(function(){
    b.innerHTML='<span class="hz-ico">Copied</span>';b.dataset.copied='true';
    setTimeout(function(){b.textContent='Copy';delete b.dataset.copied;},1400);});});
})();
</script>"""


def get():
    tabs = "".join(
        '<button class="tab" role="tab" type="button" id="tab-%s" data-os="%s" '
        'aria-controls="os-%s" aria-selected="false">%s</button>' % (o, o, o, html.escape(n))
        for o, n, _ in INSTALL)
    cmds = "".join(
        '<div class="cmd hz-code hz-codeblock hz-swap" id="os-%s" role="tabpanel" aria-labelledby="tab-%s">'
        '<span class="os hz-label">%s</span><code>%s</code>'
        '<button class="cp hz-btn hz-btn-sm hz-copy" type="button" aria-label="Copy the %s install command">Copy</button></div>'
        % (o, o, html.escape(n), html.escape(c), html.escape(n))
        for o, n, c in INSTALL)
    return (
        '<section class="get hz-grid hz-grid-2" aria-label="Get started">'
        '<div class="gbox hz-card hz-gleam"><div class="ghead"><b>Install</b>'
        '<div class="tabs hz-seg" style="--hz-seg-n:3" role="tablist" aria-label="Operating system" hidden>%s</div></div>'
        '%s</div>'
        '<div class="gbox hz-card"><div class="ghead"><b>Or paste into your agent</b><span>%s</span></div>'
        '<div class="cmd hz-code hz-codeblock"><code>%s</code>'
        '<button class="cp hz-btn hz-btn-sm hz-copy" type="button" aria-label="Copy the setup line for your agent">Copy</button>'
        '</div></div>%s</section>'
        '<div class="nexts hz-small"><p>%s</p><p>%s</p></div>'
        % (tabs, cmds, html.escape(PASTE_TO), html.escape(PASTE), GET_JS, NEXT, AFTER))


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


SEARCH = ('<svg viewBox="0 0 16 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
          'aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3 3"/></svg>')

FILTER_JS = (
    '<script id="filterjs">const q=document.getElementById(\'q\'),n=document.getElementById(\'count\'),'
    'R=[...document.querySelectorAll(\'.row\')],S=[...document.querySelectorAll(\'.sec\')];let was=false;'
    'function f(){const v=q.value.trim().toLowerCase();let s=0;'
    'R.forEach(r=>{const m=!v||r.dataset.k.includes(v);r.style.display=m?\'\':\'none\';if(m)s++;});'
    'document.documentElement.classList.toggle(\'filtering\',!!v||was);S.forEach(c=>{const any=R.some(r=>c.contains(r)&&r.style.display!==\'none\');c.hidden=!any;'
    'if(v)c.open=any;else if(was)c.open=false;});if(was&&!v)requestAnimationFrame(()=>requestAnimationFrame(()=>{if(!q.value.trim())document.documentElement.classList.remove(\'filtering\')}));was=!!v;'
    'n.textContent=s+(s===1?\' skill\':\' skills\');}'
    'function h(){const d=document.getElementById(location.hash.slice(1));if(d&&d.matches(\'.sec\')){d.open=true;d.scrollIntoView();}}'
    # toggle and re-anchor in the same task, before the browser paints: a frame later shows the jump
    'document.addEventListener(\'click\',e=>{const m=e.target.closest(\'.sec>summary\');if(!m)return;e.preventDefault();'
    'const t=m.getBoundingClientRect().top;m.parentElement.open=!m.parentElement.open;scrollBy(0,m.getBoundingClientRect().top-t);});'
    'q.addEventListener(\'input\',f);addEventListener(\'hashchange\',h);f();h();</script>')


def catalogue(out, wrap_open):
    """Rebuild the catalogue block: sticky filter, an index of the categories, one
    section per category. Built from the page's own categories and rows, in either
    the old flat form or this one, so a second run changes nothing."""
    m = re.search(r'<div class="wrap[^"]*">\n(.*?)(?=\n<footer>)', out, re.S)
    box = m.group(1)
    inp = re.search(r'<input id="q"[^>]*>', box).group(0)
    secs = []
    for t in re.finditer(r'<(?:div|h2) class="cat" data-cat="([^"]*)">([^<]*)</(?:div|h2)>'
                         r'|(<a class="row[^"]*" href="[^"]*" data-k="[^"]*">.*?</a>)', box, re.S):
        if t.group(3):
            secs[-1][2].append(re.sub(r'^<a class="row[^"]*"', '<a class="row hz-card"', t.group(3)))
        else:
            secs.append((t.group(1), t.group(2), []))
    ids = ["c-" + re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-") for c, _, _ in secs]
    assert len(set(ids)) == len(ids)
    cut = lambda std: "".join(
        '<details class="sec hz-fold" id="%s"><summary><h2 class="cat" data-cat="%s">%s</h2></summary>'
        '%s</details>' % (i, c, l, "".join(r))
        for i, (c, l, r) in zip(ids, secs) if c.startswith("standards-") == std)
    body, std = cut(False), cut(True)
    new = ('<div class="find"><label class="hz-field">%s%s</label><div id="count"></div></div>'
           '<div class="catgrid">%s</div><div class="catgrid catstd">%s</div>' % (SEARCH, inp, body, std))
    out = out[:m.start()] + wrap_open + "\n" + new + out[m.end(1):]
    return re.sub(r'<script(?: id="filterjs")?>const q=.*?</script>', lambda _: FILTER_JS, out, count=1, flags=re.S)


BEACON = re.compile(r'\s*<script[^>]*cloudflareinsights\.com[^>]*>\s*</script>', re.S)


def patch(path):
    src = open(path, encoding="utf-8").read()
    # Cloudflare adds its analytics beacon at the edge; it is not ours to ship.
    out = BEACON.sub("", src)
    out = re.sub(r"<style>.*?</style>", lambda _: "<style>%s</style>" % CSS, out, count=1, flags=re.S)
    # Light and dark follow the device; the kit's tokens live on .hz.
    out = re.sub(r'<meta name="color-scheme" content="[^"]*">', '<meta name="color-scheme" content="light dark">', out, count=1)
    out = re.sub(r"<body[^>]*>", '<body class="hz">', out, count=1)
    out = re.sub(r"<pre(?: class=\"[^\"]*\")?>", '<pre class="hz-code">', out)
    wrap = '<div class="wrap hz-wrap%s">'

    up = re.search(r'<a class="up[^"]*" href="([^"]+)">([^<]+)</a>', out)
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
        out = re.sub(r'<h1>.*?(?=<div class="btns[^"]*">)', lambda _: hero_h1_lede + get(), out,
                     count=1, flags=re.S)
        out = re.sub(r'<div class="btns[^"]*">.*?(?=</div></div><div class="strip">)',
                     lambda _: btns(), out, count=1, flags=re.S)
        out = re.sub(r'<div class="strip">.*?(?=\n<div class="wrap[^"]*">)',
                     lambda _: strip(), out, count=1, flags=re.S)
        out = re.sub(r'<div class="hero"><div class="wrap[^"]*">', '<div class="hero">' + wrap % " hz-arrive", out, count=1)
        out = catalogue(out, wrap % " cats")
        out = re.sub(r'\s*<script id="copyjs">.*?</script>', "", out, flags=re.S)
        out = re.sub(r"\s*</body>", COPY_JS + "</body>", out, count=1)

    else:
        out = re.sub(r'<div class="wrap[^"]*">', wrap % " doc", out, count=1)

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
