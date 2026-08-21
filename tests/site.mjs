/**
 * Browser checks for the landing page: mark, copy-prompt selector, agent bar,
 * button radius. Run against a local server or the live host:
 *   node tests/site.mjs [base-url] [shot-dir]
 */
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.argv[2] || 'http://127.0.0.1:8787';
const SHOTS = process.argv[3] || '/tmp/skills-shots';
mkdirSync(SHOTS, { recursive: true });

let failed = 0;
const ok = (name, cond, detail = '') => {
  if (!cond) failed++;
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1280, height: 900 },
  permissions: ['clipboard-read', 'clipboard-write'],
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(BASE + '/', { waitUntil: 'load' });

// 1 — the mark renders
const mark = page.locator('nav .brand svg.mark');
const markBox = await mark.boundingBox();
ok('logo: one mark in nav', (await mark.count()) === 1);
ok('logo: mark has area', markBox && markBox.width > 8 && markBox.height > 8,
  markBox ? `${Math.round(markBox.width)}x${Math.round(markBox.height)}` : 'no box');
ok('logo: mark has canonical geometry',
  (await mark.locator('path').count()) === 5,
  `${await mark.locator('path').count()} paths`);
ok('favicon: declared', (await page.locator('link[rel="icon"]').count()) === 1);
const favStatus = (await page.request.get(BASE + '/favicon.svg')).status();
ok('favicon: served', favStatus === 200, `HTTP ${favStatus}`);

// 2 — buttons are rounded
const radii = await page.$$eval('.btn', (els) =>
  els.map((e) => ({ t: e.textContent.trim().slice(0, 24), r: getComputedStyle(e).borderTopLeftRadius,
                    h: Math.round(e.getBoundingClientRect().height) })));
for (const r of radii) {
  const px = parseFloat(r.r);
  ok(`radius: "${r.t}" is a pill`, px >= r.h / 2, `${r.r} on ${r.h}px tall`);
}
const inputRadius = await page.$eval('#q', (e) => getComputedStyle(e).borderTopLeftRadius);
ok('radius: filter input rounded', parseFloat(inputRadius) > 8, inputRadius);

// 3 — the selector opens and each item copies a DIFFERENT prompt
const btn = page.locator('#copybtn');
ok('selector: button present', (await btn.count()) === 1);
ok('selector: labelled', (await btn.innerText()).toLowerCase().includes('copy setup prompt'));
ok('selector: has chevron', (await btn.locator('svg.chev').count()) === 1);
ok('selector: closed initially', await page.locator('#copymenu').isHidden());
await btn.click();
await page.waitForTimeout(120);
ok('selector: opens on click', await page.locator('#copymenu').isVisible());
ok('selector: aria-expanded true', (await btn.getAttribute('aria-expanded')) === 'true');

const items = page.locator('#copymenu .mi');
const n = await items.count();
ok('selector: four rows', n === 4, `${n} rows`);
for (let i = 0; i < n; i++) {
  const row = items.nth(i);
  ok(`selector: row ${i} has a mark`, (await row.locator('svg.mark').count()) === 1,
    (await row.innerText()).split('\n')[0]);
}

const copied = [];
for (let i = 0; i < n; i++) {
  const row = items.nth(i);
  const label = (await row.innerText()).replace(/\s+/g, ' ').trim();
  if (!(await row.getAttribute('data-a'))) {
    ok(`selector: "${label}" links to the manifest`,
      (await row.getAttribute('href')) === '/skill');
    continue;
  }
  if (await page.locator('#copymenu').isHidden()) { await btn.click(); await page.waitForTimeout(100); }
  await row.click();
  await page.waitForTimeout(150);
  const text = await page.evaluate(() => navigator.clipboard.readText());
  copied.push({ label, text });
  ok(`copy: "${label}" put a prompt on the clipboard`, text.length > 120, `${text.length} chars`);
}
const uniq = new Set(copied.map((c) => c.text));
ok('copy: every target copies a DIFFERENT prompt', uniq.size === copied.length,
  `${uniq.size} distinct of ${copied.length}`);
for (const c of copied) {
  const first = c.text.split('\n')[0];
  console.log(`        ${c.label.padEnd(28)} -> ${first}`);
}
// each prompt must name the path that tool actually uses
const paths = { 'Claude Code': '~/.claude/skills', Codex: 'AGENTS.md', Cursor: '.cursor/rules' };
for (const c of copied) {
  const key = Object.keys(paths).find((k) => c.label.includes(k));
  if (key) ok(`copy: "${key}" prompt names ${paths[key]}`, c.text.includes(paths[key]));
}

// 4 — the agent bar scrolls
const bar = page.locator('.bar');
ok('bar: present', (await bar.count()) === 1);
const chips = bar.locator('.agent');
const chipCount = await chips.count();
ok('bar: covers more than the reference seven', chipCount >= 8, `${chipCount} agents`);
ok('bar: every agent has a mark',
  (await bar.locator('.agent svg.mark').count()) === chipCount);
const names = await chips.allInnerTexts();
console.log('        agents: ' + names.map((s) => s.trim()).join(' · '));
const scroll = await bar.evaluate((e) => {
  const before = e.scrollLeft;
  e.scrollLeft = e.scrollWidth;
  return { over: e.scrollWidth > e.clientWidth + 4, before, after: e.scrollLeft,
           sw: e.scrollWidth, cw: e.clientWidth };
});
ok('bar: overflows its track', scroll.over, `scrollWidth ${scroll.sw} > clientWidth ${scroll.cw}`);
ok('bar: actually scrolls', scroll.after > scroll.before, `scrollLeft ${scroll.before} -> ${scroll.after}`);
await bar.evaluate((e) => { e.scrollLeft = 0; });

ok('no page errors', errors.length === 0, errors.join(' | '));

// 5 — screenshots
for (const [w, h, tag] of [[1280, 900, 'desktop-1280'], [390, 844, 'mobile-390']]) {
  await page.setViewportSize({ width: w, height: h });
  await page.waitForTimeout(200);
  await page.screenshot({ path: `${SHOTS}/${tag}.png` });
  await page.locator('#copybtn').click();
  await page.waitForTimeout(200);
  await page.screenshot({ path: `${SHOTS}/${tag}-menu.png` });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(120);
}
console.log(`\nshots -> ${SHOTS}`);

// a content page keeps the same chrome
await page.setViewportSize({ width: 1280, height: 900 });
await page.goto(BASE + '/hip-0026.html', { waitUntil: 'load' });
ok('subpage: carries the mark', (await page.locator('nav .brand svg.mark path').count()) === 5);
const subRadius = await page.$eval('pre', (e) => getComputedStyle(e).borderTopLeftRadius).catch(() => '0px');
ok('subpage: surfaces rounded', parseFloat(subRadius) > 4, subRadius);
await page.screenshot({ path: `${SHOTS}/subpage-1280.png` });

await browser.close();
console.log(failed ? `\n${failed} FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
