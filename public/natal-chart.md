---
name: natal-chart
description: "Compute an astrology natal/birth chart: planetary positions, Ascendant, MC, houses, and aspects with orbs. Use when someone asks for their birth chart, natal chart, horoscope, rising sign, moon sign, 'what's my chart', or gives a birth date + time + place."
source: hanzo (verified 2026-09-30, pyswisseph 2.10.3)
---

# Natal Chart

Deterministic computation from (date, local time, UTC offset, lat, lon). Never
invent positions — always run the script.

## Inputs you MUST have

1. **Date** (YYYY-MM-DD), **time** (HH:MM local), **place**
2. **Timezone offset** — confirm it, don't guess. DST is the #1 error source:
   a 1-hour mistake shifts the Ascendant ~15° and rehouses every planet.
   If the user gives only a place and date, look up whether DST was in effect
   (e.g. Topeka KS, June 30 1983 → CDT = UTC−5).
3. **Coordinates** — geocode the place:
   `curl -s "https://nominatim.openstreetmap.org/search?q=Topeka,KS&format=json&limit=1"`
   → `lat`, `lon` (west is negative).

## Setup — run via `/v1/exec`, the session IS the cache

One-off Python runs against `POST /v1/exec` (hanzoai/cloud `apps/exec`). The
design fact that makes caching trivial: **a session is a sandbox** — it has a
lease and files that persist across calls. `session_id` continues an existing
sandbox, so an install done in call 1 is still there in call 2. No stamp-file
hack needed; the session id IS the cache key.

**Auth: IAM only.** Send the end user's validated IAM bearer (the chat server
forwards it as `codeAuthHeaders`). Do NOT use an `hrun_` run key and do NOT
use `X-API-Key`/`CODE_EXEC_API_KEY` — both doors are deliberately closed; a
caller IAM can't resolve an identity for cannot run code.

Request body (`CodeRun`):
```json
{
  "lang": "py",
  "code": "<whole program, written to one file>",
  "args": ["--date", "1983-06-30", "--time", "16:20", "--tz", "-5",
           "--lat", "39.0473", "--lon", "-95.6752"],
  "session_id": "<omit on first call; echo it back to reuse the sandbox>"
}
```

Install-once pattern — first call in a fresh session installs, every later
call on the same `session_id` finds it already there:
```python
import importlib.util, subprocess, sys
if importlib.util.find_spec("swisseph") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyswisseph"])
# ...then the chart computation
```

Policy:
- **Reuse the `session_id`** across a day's requests → one ~5s install serves
  every chart on that warm sandbox. The reaper ends the lease on TTL / ~1h
  idle; the next call re-leases fresh and reinstalls once. That is the
  "≤1 install per day" behavior, for free.
- **On explicit request** — start a NEW session (drop the id) or `pip install
  -U pyswisseph` when a user says "update the astrology lib." Never silent.
- **Shared across users** — sessions are per-principal (the org IAM resolves).
  One warm sandbox serves that org's members; it never crosses tenants.

The core chart needs **no ephemeris files** — pyswisseph's built-in Moshier
fallback covers Sun–Pluto + Node to arcsecond precision for 1800–2050.

**Chiron does NOT work out of the box**: it requires `seas_18.se1`, and the
public mirrors (kset.org, astro.com/ftp) return HTML error pages. Do not
fabricate a Chiron position. Either skip it and say so, or vendor the file
into the sandbox image from a known-good copy.

## Image input — CPU-only OCR fallback (verified 2026-09-30)

When someone drops a **chart image** (a screenshot of an astro.com wheel, a
scanned birth card) and the vision-LLM is unavailable, OCR it with
**docTR** — pure PyTorch, runs CPU-only, no GPU needed. Extract the
placements, then **recompute with this skill** to verify/correct them: OCR
misreads degrees, the script doesn't.

**Recipe (all via `/v1/exec`, same session as the chart):**

```python
# 1. CPU-only venv — Python 3.12 (3.14 has no torch wheels yet)
subprocess.check_call(["uv", "venv", "--python", "3.12", "/mnt/data/ocrvenv"])
# 2. torch from the CPU index (default pulls a ~2.5GB CUDA build)
subprocess.check_call(["uv", "pip", "install", "--python", "/mnt/data/ocrvenv/bin/python",
                      "--index-url", "https://download.pytorch.org/whl/cpu", "torch"])
# 3. doctr + pillow
subprocess.check_call(["uv", "pip", "install", "--python", "/mnt/data/ocrvenv/bin/python",
                      "python-doctr", "pillow"])
```

**The one gotcha that bites:** `python-doctr` pulls `torchvision` from PyPI,
which is built against a *different* torch ABI than the `+cpu` build →
`RuntimeError: operator torchvision::nms does not exist`. Fix by forcing the
matching CPU wheel:

```
uv pip install --python /mnt/data/ocrvenv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torchvision==<ver>+cpu
```

**Background the install.** The gateway 502s any exec call past ~20–30s; a
torch install runs longer. Launch it detached and poll a marker:

```python
subprocess.Popen(["bash", "/mnt/data/install.sh"], stdout=open(log,"w"),
                 stderr=subprocess.STDOUT, start_new_session=True)
# poll: os.path.exists("/mnt/data/install.done")
```

**Run OCR through the venv python, not the exec `code`** — `code` runs under
the sandbox's system Python 3.14, which has no torch. Subprocess into
`/mnt/data/ocrvenv/bin/python`:

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn",
                      pretrained=True, assume_straight_pages=True)
res = model(DocumentFile.from_images("/mnt/data/chart.png"))
for line in (w for p in res.export()["pages"] for b in p["blocks"]
             for l in b["lines"] for w in [l]):
    print(" ".join(x["value"] for x in line["words"]))
```

**Timings (measured):** install ~35s once per session; pretrained weights
(~100MB) download once to `~/.cache/doctr` and persist with the session;
warm predict **~11s on CPU**. Counts as an `/v1/exec` run against the daily
cap like any other.

**Fallback order:** vision-LLM first (reads a wheel's geometry directly);
docTR CPU when it's unavailable or you need text off a scanned card. Either
way, **recompute the chart from the birth data** — never trust OCR'd degrees
as final.

## Run

`POST /v1/exec` with `lang: "py"`, the chart program in `code`, the birth
args in `args`, and the warm `session_id` from the install call. The program
is `scripts/natal_chart.py` (same CLI, just shipped as the `code` field):

```
--date 1983-06-30 --time 16:20 --tz -5 --lat 39.0473 --lon -95.6752
```

Returns: Ascendant, MC, all planets with sign/degree/house/retrograde, and
all aspects with orbs. Default Placidus (`--houses K` Koch, `W` Whole-sign).
Files the run writes (e.g. a rendered wheel PNG) stay in the session sandbox —
read them back with `GET /v1/exec/download/{sid}/{fileId}`.

## Correctness traps (all hit and fixed in the script)

- **Retrograde = `r[0][3]` (dlon/day), NOT `r[0][4]`** — index 4 is dlat;
  using it falsely flags the Moon retrograde and misses real retrogrades.
- **The Moon is never retrograde.** If your output says so, the bug is yours.
- **Sun/Moon trine ≠ full/new moon** — aspect geometry is independent of
  phase; don't "correct" one with the other.
- Birth time ±4 min moves the ASC ~1°. State the assumption in your answer.

## Output shape

Lead with the Big Three (Sun / Moon / Ascendant), then planets-by-house,
then tight aspects (orb < 2°) as the interesting ones. Note timezone used.

## Caching & usage policy (decided 2026-09-30)

The chart is a **pure function** of `(date|time|tz|lat|lon|house_system)`.

- **Result cache: SHARED across all accounts.** Key =
  `sha1(date|time|tz|lat|lon|system)`. Same birth data → one compute,
  served to everyone. (Charts aren't secret — the inputs are the key.)
- **Environment cache:** reuse the `session_id` so the sandbox keeps its
  installed `pyswisseph`; one ~5s install per warm session, reaper recycles
  on idle. Worst case one reinstall per cold-start, not per request.
- **Every astrology query counts against the daily cap** — cache hit or
  fresh compute alike. The cap is a *fairness* limit, not a cost limit.

### Free-plan limits (grounded in existing metering)

`/v1/exec` is already metered at 1¢/run (`CODE_EXEC_FEE_CENTS`, per-org
gate + debit in `apps/exec/meter.go`); per-scope rate limiting lives in
`middleware_ratelimit.go`. On top of that, for the free astrology surface:

| Cap | Value |
|---|---|
| Astrology queries / account / day | **50** (every query counts) |
| LLM calls per agentic turn | **5** |
| `/v1/exec` tool calls per turn | **10** |
| Hard `/v1/exec` runs / account / day | **100** |

A cache hit burns 1 of the 50 but ~0 compute. The per-turn caps stop a
runaway agentic loop from spending past the daily ceiling.

## Verified worked example (accuracy proof)

April 11 1985, 2:05 AM, Royal Victoria Hospital Montréal (45.5017, −73.5673),
EST = **UTC−5** (NOT EDT — the chat AI's "correction" to UTC−4 was backwards):

```
Asc 27°Capricorn  MC 24°Scorpio
Sun 21°Aries  Moon 10°Capricorn  Mercury 8°Aries R  Venus 9°Aries R
Mars 19°Taurus  Jupiter 12°Aquarius  Saturn 27°Scorpio R
Uranus 17°Sagittarius R  Neptune 3°Capricorn R  Pluto 3°Scorpio R
```

This matches astro.com's computed chart to the arcminute. The chat AI, with no
compute tool, guessed ~6 of 10 placements wrong and missed every retrograde.
**Always run the script; never guess positions.**
