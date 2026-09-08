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
