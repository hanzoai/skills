/**
 * What is actually on the screen.
 *
 * A layout box is not visibility. Three ways a node keeps a box and shows
 * nothing, all of which counted as visible until they were measured:
 *
 *   a screen-reader wrapper clips it to one pixel — the child still reports a
 *   box, and a heading in a 1px box wraps one letter to a line and returns a
 *   width of twenty-one and a height of a hundred and eighteen;
 *
 *   a closed <details> keeps its contents in the tree, laid out, addressable,
 *   and draws none of them but its own summary, which is in plain sight;
 *
 *   an overflow-hidden ancestor cuts the child's box away from the part of the
 *   screen the ancestor occupies.
 *
 * So the question is asked of the text, not the element: a Range over the node
 * returns the rectangles the glyphs were painted into, and each one is tested
 * against every clipping ancestor up to the root. Nothing survives that it
 * would not survive on the retina.
 *
 * `scope` says where the question is being asked. The three answers are
 * different facts about the same page and must never be compared to each
 * other: `view` is the reference screenshot's contract — the first frame, the
 * thing a person sees before deciding to scroll. `page` is the whole document.
 * `region` is one named part of it. An assertion records which it used.
 *
 * This is one half of the answer. It counts semantic material and cannot see
 * black text on a black ground, which passes every test here. The screenshot
 * is the other half and is the final authority.
 */

/** Runs in the page. `scope` is 'view' | 'page' | a CSS selector for a region. */
export function census(scope = 'view') {
  // The document, for both of the whole-page scopes. `view` is a fact about
  // the viewport and `page` a fact about the document; neither is a fact about
  // a landmark, and rooting them at `main ?? body` made them one — this app has
  // a `main` and the reference has none, so the same word named two different
  // questions and every cross-build row was the shell counted on one side only.
  // A screen on its own is a region and is asked for by name.
  const root =
    scope === 'view' || scope === 'page' ? document.body : document.querySelector(scope)
  if (!root) return { missing: scope }

  const fold = scope === 'view' ? innerHeight : Infinity
  const frame = scope === 'view' || scope === 'page'
    ? { top: 0, left: 0, bottom: fold, right: innerWidth }
    : root.getBoundingClientRect()

  const meets = (a, b) =>
    a.bottom > b.top + 0.5 && a.top < b.bottom - 0.5 && a.right > b.left + 0.5 && a.left < b.right - 0.5

  /** Every ancestor that cuts, and every ancestor that hides outright. */
  const clips = (e) => {
    const out = []
    for (let n = e.parentElement, from = e; n; from = n, n = n.parentElement) {
      const s = getComputedStyle(n)
      if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') return null
      if (s.opacity !== '' && Number.parseFloat(s.opacity) < 0.01) return null
      // A closed `details` draws one thing: its own `summary`, which is its
      // direct child. So what arrived from there survives and everything else
      // does not. Skipping the summary and carrying on up was not the same
      // rule — the walk reached the closed details on the next step and
      // returned null anyway, and every line that opens an explanation counted
      // as unpainted while sitting in plain sight on nine screens.
      if (n.tagName === 'DETAILS' && !n.open && from.tagName !== 'SUMMARY') return null
      if (s.overflow !== 'visible' || s.clipPath !== 'none') out.push(n.getBoundingClientRect())
    }
    return out
  }

  /** Where the glyphs actually landed, not where the element claims to be. */
  const painted = (e) => {
    const cut = clips(e)
    if (!cut) return []
    const s = getComputedStyle(e)
    if (s.display === 'none' || s.visibility === 'hidden' || Number.parseFloat(s.opacity || '1') < 0.01) return []
    const r = document.createRange()
    r.selectNodeContents(e)
    return [...r.getClientRects()].filter(
      (b) => b.width > 1 && b.height > 1 && meets(b, frame) && cut.every((c) => meets(b, c)),
    )
  }

  const shown = (e) => painted(e).length > 0
  const leaves = [...root.querySelectorAll('*')].filter((e) => !e.children.length && (e.textContent || '').trim())
  const words = (e) => e.textContent.trim().split(/\s+/).length

  /** A box that reads as its own object: it is drawn apart from its ground. */
  const carded = (e) => {
    const s = getComputedStyle(e)
    if (s.boxShadow === 'none' && !(Number.parseFloat(s.borderTopWidth) > 0)) return false
    const b = e.getBoundingClientRect()
    if (b.height <= 60) return false
    const cut = clips(e)
    return cut !== null && meets(b, frame) && cut.every((c) => meets(b, c))
  }

  const see = leaves.filter(shown)
  return {
    scope,
    h1: [...root.querySelectorAll('h1')].filter(shown).length,
    heads: [...root.querySelectorAll('h1,h2,h3')].filter(shown).length,
    words: see.reduce((n, e) => n + words(e), 0),
    prose: see.filter((e) => e.textContent.trim().length > 55).length,
    controls: [...root.querySelectorAll('button,input,select,textarea,a[href],[role="button"],[role="tab"]')]
      .filter((e) => { const b = e.getBoundingClientRect(); const c = clips(e)
        return b.width > 1 && b.height > 1 && c !== null && meets(b, frame) && c.every((x) => meets(b, x)) }).length,
    cards: [...root.querySelectorAll('*')].filter(carded).length,
    tall: Math.round(root.getBoundingClientRect().height),
    bg: getComputedStyle(document.body).backgroundColor,
  }
}

/** The y a named line sits on, for matching one composition's rhythm to another. */
export function lines(marks) {
  const out = {}
  for (const [name, sel] of Object.entries(marks)) {
    const e = document.querySelector(sel)
    if (!e) { out[name] = null; continue }
    const b = e.getBoundingClientRect()
    out[name] = { top: Math.round(b.top), bottom: Math.round(b.bottom), left: Math.round(b.left), w: Math.round(b.width), h: Math.round(b.height) }
  }
  return out
}

/**
 * The primitives underneath every screen: the type ramp, the spacing grid and
 * the control geometry. A screen measured on a wrong ramp reports geometry
 * errors that are really one type error, so this is asked first and the screen
 * gates mean nothing until it agrees.
 *
 * Sub-4px vertical padding is excluded from the grid. It is not layout — it is
 * the inset an engine gives an inline control so a tap lands, and counting it
 * beside a 16px gutter said 44% of our spacing was off-grid when every actual
 * layout step was on it. A gate that measures the wrong quantity fails a thing
 * that is right, which costs more than not having the gate.
 */
export function primitives() {
  const root = document.body
  const px = (v) => Math.round(Number.parseFloat(v) * 100) / 100
  const bump = (m, k) => m.set(k, (m.get(k) ?? 0) + 1)
  const top = (m) => [...m.entries()].sort((a, z) => z[1] - a[1]).slice(0, 8)
    .map(([k, n]) => `${k}:${n}`).join(' ')

  const size = new Map(), lead = new Map(), step = new Map(), inset = new Map()
  const high = new Map(), round = new Map(), heft = new Map(), face = new Map()
  for (const e of root.querySelectorAll('*')) {
    const r = e.getBoundingClientRect()
    if (r.width < 1 || r.height < 1 || r.top > innerHeight) continue
    const s = getComputedStyle(e)
    if ([...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) {
      bump(size, px(s.fontSize))
      bump(lead, px(s.lineHeight))
      // Weight and size together, because a ramp can agree on every rung and
      // still read as another typeface's page when the same rung is set in a
      // different weight.
      bump(heft, s.fontWeight)
      bump(face, `${px(s.fontSize)}/${s.fontWeight}`)
    }
    const inline = s.display.startsWith('inline') || /^(A|BUTTON|SUMMARY|SPAN|LABEL)$/.test(e.tagName)
    for (const side of ['paddingTop', 'paddingBottom', 'paddingLeft', 'paddingRight']) {
      const v = px(s[side])
      if (v <= 0) continue
      if (v < 4 && inline) bump(inset, v)
      else bump(step, v)
    }
    if (/^(BUTTON|INPUT|SELECT)$/.test(e.tagName) || e.getAttribute('role') === 'button') {
      bump(high, Math.round(r.height))
      bump(round, px(s.borderRadius))
    }
  }
  const steps = [...step.keys()]
  const on = (n) => `${Math.round((steps.filter((v) => v % n === 0).length / (steps.length || 1)) * 100)}%`
  return {
    size: top(size), lead: top(lead), step: top(step), inset: top(inset),
    heft: top(heft), face: top(face),
    grid: `4px ${on(4)}  8px ${on(8)}`, height: top(high), radius: top(round),
    body: `${getComputedStyle(document.body).fontSize} ${getComputedStyle(document.body).fontFamily.split(',')[0]}`,
  }
}

/**
 * What a control is made of, so the reason it is the wrong height is visible
 * rather than guessed at.
 *
 * Ours render at 40 and 52 where the reference has a tier at 27. The first
 * instinct is to reach for the type ramp, and the ramp is where it belongs —
 * the two builds now agree on 16/14/18. So the height difference is assembled
 * further down, out of a line-height, a padding pair, a border and a
 * min-height, and only a decomposition says which.
 *
 * The stop condition is the rendered TIER, not the mechanism. Basel at 16px
 * with leading 1.0 and 5px of padding lands on 27; Zen has different metrics
 * and will need a different combination to land on the same number. Matching
 * the combination rather than the number would be copying the arithmetic and
 * missing the answer.
 */
export function controls() {
  const root = document.body
  const px = (v) => Math.round(Number.parseFloat(v) * 100) / 100

  /** The same semantic primitive in two DOMs that share no class names. */
  const kind = (e) => {
    const tag = e.tagName
    const role = e.getAttribute('role')
    const text = (e.textContent || '').trim()
    const icon = e.querySelector('svg, img')
    if (role === 'tab' || e.closest('[role="tablist"]')) return 'Tab'
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
      const hint = `${e.getAttribute('placeholder') ?? ''} ${e.getAttribute('aria-label') ?? ''}`
      return e.type === 'search' || /search|find/i.test(hint) ? 'Search' : 'Input'
    }
    if (role === 'combobox' || e.getAttribute('aria-haspopup')) return 'Select'
    if (tag === 'BUTTON' || role === 'button') {
      if (!text && icon) return 'IconButton'
      return 'Button'
    }
    return null
  }

  const seen = new Map()
  for (const e of root.querySelectorAll('button,input,select,textarea,[role="button"],[role="tab"],[role="combobox"]')) {
    const b = e.getBoundingClientRect()
    if (b.width < 1 || b.height < 1 || b.top > innerHeight) continue
    const k = kind(e)
    if (!k) continue
    const s = getComputedStyle(e)
    const r = document.createRange()
    r.selectNodeContents(e)
    const glyph = [...r.getClientRects()].filter((x) => x.height > 1)
    const svg = e.querySelector('svg, img')
    const box = svg?.getBoundingClientRect()
    const row = {
      outer: Math.round(b.height),
      content: px(s.height),
      size: px(s.fontSize),
      lead: px(s.lineHeight),
      glyph: glyph.length ? Math.round(Math.max(...glyph.map((x) => x.height))) : 0,
      pad: `${px(s.paddingTop)}/${px(s.paddingBottom)}`,
      border: `${px(s.borderTopWidth)}/${px(s.borderBottomWidth)}`,
      min: s.minHeight === 'auto' || s.minHeight === '0px' ? '·' : px(s.minHeight),
      icon: box ? `${Math.round(box.width)}×${Math.round(box.height)}` : '·',
      // Where the icon sits against the text it stands beside, which is what a
      // reader notices when it is wrong and no single measurement reports.
      off: box && glyph.length ? Math.round(box.top + box.height / 2 - (glyph[0].top + glyph[0].height / 2)) : '·',
      gap: s.gap === 'normal' ? '·' : px(s.gap),
      align: s.alignItems,
      says: (e.textContent || '').trim().slice(0, 18),
    }
    if (!seen.has(k)) seen.set(k, [])
    seen.get(k).push(row)
  }
  return Object.fromEntries(seen)
}

/**
 * Where the composition's rows sit.
 *
 * Two builds share no class names and no element identity, so a box cannot be
 * matched to its counterpart by name. What can be matched is rhythm: the set of
 * y positions where visible content starts, and how wide and how far left the
 * widest thing at each of those positions is. A person reading the two
 * screenshots side by side is comparing exactly that.
 *
 * Reported as bands rather than elements because a band survives a different
 * markup of the same design, and an element does not.
 */
export function bands(scope = 'view') {
  const root = scope === 'view' || scope === 'page' ? document.body : document.querySelector(scope) ?? document.body
  const fold = scope === 'view' ? innerHeight : Infinity

  // The same question `census` asks of a text node, asked of a box: does an
  // ancestor hide it, and does every ancestor that cuts still leave it on the
  // screen. Written out a second time because an evaluated function is
  // serialised on its own and reaches nothing this module holds.
  //
  // Skipping it read the reference's chrome as composition. `lux.exchange`
  // keeps its network menu and its wallet menu mounted inside a 1x1 wrapper at
  // opacity 1e-08 with overflow hidden — eleven chain names and seven wallet
  // rows, all of them laid out, none of them painted. Counting the boxes made
  // Explore's reference 51 rows deep where a reader sees 13, and every screen
  // was then scored against a column of menu items nobody can see.
  const meets = (a, b) =>
    a.bottom > b.top + 0.5 && a.top < b.bottom - 0.5 && a.right > b.left + 0.5 && a.left < b.right - 0.5
  const painted = (e, b) => {
    for (let n = e.parentElement, from = e; n; from = n, n = n.parentElement) {
      const s = getComputedStyle(n)
      if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') return false
      if (s.opacity !== '' && Number.parseFloat(s.opacity) < 0.01) return false
      if (n.tagName === 'DETAILS' && !n.open && from.tagName !== 'SUMMARY') return false
      if ((s.overflow !== 'visible' || s.clipPath !== 'none') && !meets(b, n.getBoundingClientRect())) return false
    }
    return true
  }

  const at = new Map()
  for (const e of root.querySelectorAll('*')) {
    const b = e.getBoundingClientRect()
    if (b.width < 8 || b.height < 8 || b.top > fold) continue
    const s = getComputedStyle(e)
    if (s.display === 'none' || s.visibility === 'hidden' || Number.parseFloat(s.opacity || '1') < 0.01) continue
    if (!(e.textContent || '').trim() && !e.querySelector('svg,img,canvas')) continue
    if (!painted(e, b)) continue
    // Bands are 4px wide: finer than that and a one-pixel border opens a band
    // of its own, coarser and two genuinely separate rows merge into one.
    const y = Math.round(b.top / 4) * 4
    const had = at.get(y)
    if (!had || b.width > had.w) at.set(y, { y, x: Math.round(b.left), w: Math.round(b.width), h: Math.round(b.height) })
  }
  return [...at.values()].sort((a, z) => a.y - z.y)
}

/**
 * How closely one composition's rhythm follows another's.
 *
 * Each of the reference's bands is paired with the nearest of ours, and the
 * score is the share of them landing within a band's own width — twelve
 * pixels, which is about where a reader stops seeing two rows as aligned. The
 * largest single displacement is reported beside the score, because a
 * percentage says how much agrees and never says what does not.
 */
export function rhythm(ref, ours) {
  if (!ref.length || !ours.length) return { score: null, why: 'a build has no bands' }

  // Symmetric, and it has to be. Pairing each reference row to its nearest of
  // ours rewards having MORE rows: every extra row is another chance to sit
  // within tolerance of a reference row, and a page with a row every four
  // pixels scores a hundred per cent. Measured — feeding the local indexer real
  // data took explore from 43% to 94% without moving a single row, purely by
  // adding them.
  //
  // So both directions are counted: a reference row nothing of ours lands on is
  // a row we are missing, and a row of ours nothing of theirs lands on is a row
  // we invented. The score is the harmonic mean, which a build cannot raise by
  // adding rows and cannot raise by removing them either.
  const near = (a, b) => {
    let best = Infinity
    for (const y of b) best = Math.min(best, Math.abs(a - y))
    return best
  }
  const theirs = ref.map((r) => r.y)
  const mine = ours.map((r) => r.y)
  const found = ref.filter((r) => near(r.y, mine) <= 12).length
  const kept = ours.filter((r) => near(r.y, theirs) <= 12).length
  const recall = found / ref.length
  const precision = kept / ours.length
  const score = recall + precision === 0 ? 0 : (2 * recall * precision) / (recall + precision)

  let worst = { d: -1 }
  for (const a of ref) {
    const d = near(a.y, mine)
    if (d > worst.d) worst = { d, at: a }
  }
  const dx = ref[0] && ours[0] ? Math.round(ours[0].x - ref[0].x) : 0
  return {
    score: Math.round(score * 100),
    found: Math.round(recall * 100),
    kept: Math.round(precision * 100),
    bands: `${ref.length}/${ours.length}`,
    worst: worst.d > 0 ? `reference row at y${worst.at.y} (w ${worst.at.w}) has nothing of ours within ${worst.d}px` : '·',
    inset: dx,
  }
}
