---
name: showcase-tech-terminal
description: Tech Terminal showcase — animated CLI/terminal output for developer-focused videos. Typewriter commands, line-by-line outputs, progress bars, and GlassmorphismCard highlights. Produces a 15–45s 16:9 MP4 via the TechTerminal composition.
source: melgarafael/growthOS
---

# Showcase: Tech Terminal

Specialist skill for the **Tech Terminal** showcase. Activated by the `remotion-pro` orchestrator when the user selects showcase `2` (`terminal`, `cli`, `tech`, `código`, `code`). Receives a context object with `video_profile`, `brand_voice`, `remotion`, and `briefing`.

---

## Overview

The Tech Terminal showcase produces developer-centric videos featuring a realistic terminal emulator window. Commands appear with typewriter animation, outputs render line-by-line, and progress bars animate for install or build sequences. After the core terminal sequence, the 2–3 most impressive results are featured in glassmorphic highlight cards.

**Best for:** CLI tools, developer libraries, DevOps products, open-source projects, API demos, build pipelines, npm packages, Homebrew formulae.

**Composition ID:** `TechTerminal`
**Aspect ratio:** 16:9 (1920×1080)
**FPS:** 30
**Duration range:** 15–45 seconds (450–1350 frames)

---

## Components Used

| Component | File | Purpose in video |
|---|---|---|
| `TerminalEmulator` | `components/TerminalEmulator.tsx` | Realistic terminal window with configurable theme, prompt, title, font size |
| `AnimatedText` | `components/AnimatedText.tsx` | Typewriter effect for individual command and output lines |
| `ProgressBar` | `components/ProgressBar.tsx` | Animated install/build progress bar inside terminal lines |
| `GlassmorphismCard` | `components/GlassmorphismCard.tsx` | Glassmorphic highlight cards for the top 2–3 result moments |
| `BackgroundGradient` | `components/BackgroundGradient.tsx` | Dark branded gradient background |
| `BrandWatermark` | `components/BrandWatermark.tsx` | Handle overlay at watermark position |

The `TerminalEmulator` accepts a `commands: TerminalCommand[]` array. Each `TerminalCommand` has:
```typescript
{
  text: string;
  type: "input" | "output" | "spinner" | "success" | "error" | "blank";
  color?: string;
  delay?: number;    // frames to wait before this line appears
  typeSpeed?: number; // chars per frame for "input" type
}
```

---

## Scene Flow Template

Default sequence for a medium-pacing, 4-command demo:

```
Scene 1 — Hook           (0s–2.5s,   frames 0–74)
  AnimatedText: product/tool name fades in
  Subtext: one-line pitch, animation: fade-in delay 12

Scene 2 — Terminal Opens  (2.5s–4s,  frames 75–119)
  TerminalEmulator window fades up (slide-up animation)
  Blinking cursor appears at prompt
  Title bar shows: "[tool-name] — zsh — 80×24"

Scene 3 — Commands Type  (4s–[dynamic]s, frames 120–[calculated])
  All TerminalCommand[] lines render sequentially
  input lines: typewriter at typeSpeed chars/frame
  output lines: appear immediately after input finishes
  spinner lines: animate for spinner_duration frames then resolve
  success lines: appear in green
  error lines: appear in red
  blank lines: add breathing room between command groups

Scene 4 — Output Highlight (after terminal, 4s)
  Terminal blurs slightly (opacity 0.4)
  2–3 GlassmorphismCard overlays slide in with key results
  Each card: metric label + value, animation: slide-up

Scene 5 — CTA            (3.5s, last frames)
  Cards fade out
  AnimatedText: CTA headline (install command or URL)
  Accent-colored pill with package name / URL
```

**Pacing multipliers:**
- `fast`: typeSpeed × 1.5 (types faster), shorter delays
- `medium`: typeSpeed = 2 chars/frame (default)
- `slow`: typeSpeed × 0.7, longer pauses between commands

---

## Storyboard Generation Rules

### Step 1 — Parse the briefing

```
tool_name     → explicit name in briefing OR brand-voice.yaml → brand.name
description   → one-line pitch from briefing OR brand-voice.yaml → brand.tagline
commands      → explicit command list in briefing (ordered, code blocks preferred)
vague_trigger → briefing is vague ("show ", "demo my CLI") → use default command set
highlights    → pick 2–3 output lines with numbers, percentages, or success indicators
cta_text      → explicit in briefing OR brand-voice.yaml → cta.default
               OR auto-generate: "npm install [tool_name]"
```

### Step 2 — Build TerminalCommand list

**If briefing provides explicit commands:**

Parse code blocks and command-output pairs from the briefing. For each item:

```
1. If line starts with $ or > → type: "input"
2. If line contains [===] or (N%) → type: "spinner" + embed ProgressBar
3. If line starts with ✓ or ✅ or "Success" → type: "success"
4. If line starts with ✗ or ✘ or "Error" / "Failed" → type: "error"
5. Empty line → type: "blank"
6. Everything else → type: "output"
```

**If briefing is vague (no explicit commands):**

Use this default impressive command set tailored to the tool_name:

```typescript
const defaultCommands: TerminalCommand[] = [
  { type: "input",   text: `$ [tool_name] --version`,   typeSpeed: 3 },
  { type: "output",  text: `[tool_name] v2.1.0`,         color: "#A3E635" },
  { type: "blank",   text: "" },
  { type: "input",   text: `$ [tool_name] init my-project`, typeSpeed: 2 },
  { type: "spinner", text: `Initializing project...`,    delay: 8 },
  { type: "success", text: `✓ Project created in 0.8s`, color: "#22C55E" },
  { type: "blank",   text: "" },
  { type: "input",   text: `$ [tool_name] build --prod`, typeSpeed: 2 },
  { type: "spinner", text: `Bundling 847 modules...`,    delay: 30 },
  { type: "output",  text: `  dist/main.js  42.3 kB`,   color: "#60A5FA" },
  { type: "output",  text: `  dist/styles.css  8.1 kB`, color: "#60A5FA" },
  { type: "success", text: `✓ Built in 1.2s`,           color: "#22C55E" },
  { type: "blank",   text: "" },
  { type: "input",   text: `$ [tool_name] deploy`,       typeSpeed: 2 },
  { type: "spinner", text: `Deploying to production...`, delay: 45 },
  { type: "success", text: `✓ Live at https://[tool].app`, color: "#22C55E" },
];
```

Substitute `[tool_name]` with the actual tool name from the briefing or brand-voice.yaml.

### Step 3 — Pick output highlights (2–3 GlassmorphismCards)

Scan the TerminalCommand list for the most impressive moments to feature:

**Priority order for highlight selection:**

1. Lines with time metrics: "1.2s", "0.8s", "847ms" → "Built in 1.2s"
2. Lines with size metrics: "42.3 kB", "12 MB" → "Bundle: 42.3 kB"
3. Lines with counts: "847 modules", "12 tests passed" → "847 Modules"
4. Lines with URLs or deploy confirmation → "Live at [URL]"
5. Success lines (✓) → first 2–3 success lines

If fewer than 2 candidates: use first and last success lines.

Format each highlight as a GlassmorphismCard:
```yaml
- label: "[short label, 1-3 words]"
  value: "[metric or key phrase]"
  icon: "[relevant emoji]"
```

### Step 4 — Calculate duration

```
hook_duration        = 2.5s (always)
terminal_open        = 1.5s (always)
per_input_line       = ceil(text.length / typeSpeed) / 30 seconds
per_output_line      = 0.3s per line
per_spinner_line     = delay / 30 seconds (min 1s)
per_blank_line       = 0.3s
output_highlight     = 4s (always, covers all cards overlapping)
cta_duration         = 3.5s (always)

terminal_section_s   = sum of per-line durations
total_s              = hook + terminal_open + terminal_section + output_highlight + cta

IF total_s < 15: add additional default commands to pad
IF total_s > 45: trim output lines to first 3 per command, remove blank lines
```

### Step 5 — Output storyboard YAML

```yaml
# storyboard.yaml — Tech Terminal
showcase: TechTerminal
tool: "[tool_name]"
description: "[one-line pitch]"
total_duration_s: [calculated]
pacing: "[fast|medium|slow]"
terminal:
  theme: dark
  prompt: "$ "
  title: "[tool_name] — zsh — 80x24"
  font_size: 14

scenes:
  - id: hook
    type: hook
    duration_s: 2.5
    text: "[tool_name]"
    subtext: "[description]"
    animation: fade-in

  - id: terminal-open
    type: terminal
    duration_s: 1.5
    animation: slide-up

  - id: commands
    type: terminal
    duration_s: [calculated]
    commands:
      - { type: input,   text: "$ [command]", typeSpeed: 2 }
      - { type: output,  text: "[output line]" }
      - { type: spinner, text: "[progress text]", delay: 30 }
      - { type: success, text: "✓ [result]", color: "#22C55E" }
      # ... full command list

  - id: output-highlight
    type: output-highlight
    duration_s: 4
    cards:
      - { label: "[label]", value: "[value]", icon: "[emoji]" }
      - { label: "[label]", value: "[value]", icon: "[emoji]" }

  - id: cta
    type: cta
    duration_s: 3.5
    text: "[cta headline]"
    subtext: "[install command or URL]"
    animation: bounce
```

---

## Composition Mapping

Convert storyboard to Remotion props for the `TechTerminal` composition:

### Frame calculation

```
scene.duration_s × 30 = scene.durationFrames
startFrame[N] = sum of durationFrames[0..N-1]
```

### Props object structure

```typescript
// TechTerminalProps (from types.ts)
{
  brand: {
    name: "[tool_name]",
    handle: brand_voice.brand.handle,
    colors: {
      primary:    design_tokens.primary_color,
      secondary:  design_tokens.secondary_color,
      background: "#0D1117",     // dark terminal default
      text:       "#F0F6FC",
      accent:     design_tokens.accent_color,
    },
    watermark_position: "bottom-right",
  },
  terminalTitle: "[tool_name] — zsh — 80x24",
  terminalTheme: "dark",
  showProgressBar: true,
  showWatermark: true,
  scenes: [
    {
      type: "hook",
      startFrame: 0,
      durationFrames: 75,
      text: "[tool_name]",
      subtext: "[description]",
      animation: "fade-in",
    },
    {
      type: "terminal",
      startFrame: 75,
      durationFrames: 45,
      animation: "slide-up",
      commands: [],     // empty — just opens terminal
    },
    {
      type: "terminal",
      startFrame: 120,
      durationFrames: [calculated],
      commands: [
        { text: "$ [command]", type: "input", typeSpeed: 2 },
        { text: "[output]",    type: "output" },
        { text: "[progress]",  type: "spinner", delay: 30 },
        { text: "✓ [result]",  type: "success", color: "#22C55E" },
        // ...
      ],
    },
    {
      type: "output-highlight",
      startFrame: [after terminal],
      durationFrames: 120,
      // GlassmorphismCards defined here
      text: "[top metric label]",
      subtext: "[top metric value]",
      animation: "slide-up",
    },
    {
      type: "cta",
      startFrame: [after highlight],
      durationFrames: 105,
      text: "[cta headline]",
      subtext: "[install command or URL]",
      animation: "bounce",
    },
  ],
}
```

### Terminal theme color mapping

| Theme | Background | Text | Prompt color |
|---|---|---|---|
| `dark` (default) | `#0D1117` | `#F0F6FC` | `#22C55E` |
| `dracula` | `#282A36` | `#F8F8F2` | `#50FA7B` |
| `solarized` | `#002B36` | `#839496` | `#859900` |

The terminal window border and title bar use `brand.colors.primary` at 30% opacity.

---

## Config Reading

### From video-profile.yaml

| Field | Used for |
|---|---|
| `design_tokens.primary_color` | `brand.colors.primary`, terminal border |
| `design_tokens.accent_color` | ProgressBar color, CTA pill |
| `style.pacing` | typeSpeed multiplier |
| `style.theme` | terminal theme selection |

### From brand-voice.yaml

| Field | Used for |
|---|---|
| `brand.name` | tool_name fallback |
| `brand.handle` | `brand.handle` watermark |
| `brand.tagline` | hook subtext fallback |
| `cta.default` | CTA text fallback |
| `automation.autonomy` | output bundle level |

### Briefing overrides (highest priority)

Any explicit value in the user's briefing (tool name, commands, CTA text) takes precedence over both YAML files.

---

## Render Command

After generating the composition props, return to the orchestrator with:

```yaml
composition_id: TechTerminal
props: { ...full TechTerminalProps JSON... }
```

The orchestrator renders using the discovered Remotion CLI:

```bash
cd "${REMOTION_PROJECT}" && npx remotion render src/index.ts TechTerminal "${OUTPUT_DIR}/video.mp4" \
  --props='{"brand":{...},"scenes":[...],"terminalTitle":"..."}' \
  --fps=30
```

---

## Briefing Collection UI

When the orchestrator routes here, display this briefing prompt:

```
TECH TERMINAL SHOWCASE
─────────────────────────────────────────────────────────
I'll create a developer terminal video with animated
commands, typewriter output, and highlighted results.

  Tool / project name:
  > [e.g. " CLI" or "my-npm-package"]

  What to demonstrate (paste commands + outputs, or describe):
  > [e.g. "show npm install, then init, then build with output"
    or paste actual terminal session]

  Terminal theme: [dark / dracula / solarized]  (default: dark)
  >

  CTA text (optional):
  > [e.g. "npm install " or "docs.brand.app"]

  Pacing: [fast / medium / slow]  (default: medium)
  >
─────────────────────────────────────────────────────────
```

If user pastes a raw terminal session (no structure), parse it line-by-line using the Step 2 rules above. Lines starting with `$` or `>` are input; everything else is output.

---

## Error Handling

| Error condition | Response |
|---|---|
| Briefing too vague with no tool name | Ask: "What's the name of the CLI tool or package?" |
| Command list results in duration > 45s | Trim: keep first input + output block per command group, remove blanks |
| Command list results in duration < 15s | Expand with 2–3 additional default commands (version, help, deploy) |
| No highlight candidates found | Use first and last success/output lines as highlights |
| User pastes non-terminal content | Gently note format, ask to paste actual terminal session or describe commands |

---

## Output

Return to orchestrator:

```yaml
status: ready
composition_id: TechTerminal
storyboard_path: "[OUTPUT_DIR]/storyboard.md"
props: { ...TechTerminalProps JSON... }
estimated_duration_s: [N]
command_count: [N]
highlight_count: [N]
```

The orchestrator handles the render call, thumbnail, captions, and output bundle.
