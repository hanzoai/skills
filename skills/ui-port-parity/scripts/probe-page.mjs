#!/usr/bin/env node
import fs from 'node:fs';

function usage() {
  console.error('Usage: probe-page.mjs <url> <output.json> [selector ...]');
  process.exit(2);
}

const [url, output, ...selectors] = process.argv.slice(2);
if (!url || !output) usage();

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (err) {
  console.error('Playwright is required from the host project. Install/use the project dependency, then rerun.');
  process.exit(3);
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
await page.goto(url, { waitUntil: 'networkidle' });
await page.evaluate(() => document.fonts?.ready);

const effectiveSelectors = selectors.length ? selectors : ['[data-parity-key]'];
const result = await page.evaluate((sels) => {
  const styleFields = [
    'display','position','padding','margin','gap','gridTemplateColumns','gridTemplateRows',
    'alignItems','justifyContent','fontFamily','fontSize','fontWeight','lineHeight',
    'letterSpacing','color','backgroundColor','border','borderRadius','boxShadow',
    'opacity','overflow','zIndex','transform','transition'
  ];
  // A layout box is not visibility. A screen-reader wrapper clips its child to
  // one pixel and the child still reports a box; a closed <details> keeps its
  // contents laid out and paints none of them. So ask the text: take the Range
  // rects over the node and require one that survives every clipping ancestor.
  const painted = (el) => {
    const cuts = [];
    for (let n = el.parentElement; n; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.display === 'none' || cs.visibility === 'hidden' || cs.visibility === 'collapse') return false;
      if (parseFloat(cs.opacity || '1') < 0.01) return false;
      if (n.tagName === 'DETAILS' && !n.open) return false;
      if (n.tagName === 'SUMMARY') continue;
      if (cs.overflow !== 'visible' || cs.clipPath !== 'none') cuts.push(n.getBoundingClientRect());
    }
    const own = getComputedStyle(el);
    if (own.display === 'none' || own.visibility === 'hidden' || parseFloat(own.opacity || '1') < 0.01) return false;
    const meets = (a, b) => a.bottom > b.top + 0.5 && a.top < b.bottom - 0.5 && a.right > b.left + 0.5 && a.left < b.right - 0.5;
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = [...range.getClientRects(), el.getBoundingClientRect()];
    return rects.some((b) => b.width > 1 && b.height > 1 && cuts.every((c) => meets(b, c)));
  };
  const out = {};
  for (const selector of sels) {
    const nodes = [...document.querySelectorAll(selector)];
    out[selector] = nodes.map((el) => {
      const r = el.getBoundingClientRect();
      const s = getComputedStyle(el);
      const styles = {};
      for (const field of styleFields) styles[field] = s[field];
      return {
        parityKey: el.getAttribute('data-parity-key'),
        tag: el.tagName.toLowerCase(),
        painted: painted(el),
        rect: { x: r.x, y: r.y, width: r.width, height: r.height },
        styles,
      };
    });
  }
  return {
    url: location.href,
    viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
    elements: out,
  };
}, effectiveSelectors);

fs.writeFileSync(output, JSON.stringify(result, null, 2));
await browser.close();
console.log(output);
