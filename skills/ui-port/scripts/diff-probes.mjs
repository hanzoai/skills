#!/usr/bin/env node
import fs from 'node:fs';

const [refPath, targetPath] = process.argv.slice(2);
if (!refPath || !targetPath) {
  console.error('Usage: diff-probes.mjs <reference.json> <target.json>');
  process.exit(2);
}

const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
const target = JSON.parse(fs.readFileSync(targetPath, 'utf8'));
const rows = [];

function flattenByKey(doc) {
  const map = new Map();
  for (const nodes of Object.values(doc.elements || {})) {
    for (const node of nodes) {
      if (node.parityKey) map.set(node.parityKey, node);
    }
  }
  return map;
}

const a = flattenByKey(ref);
const b = flattenByKey(target);
const all = [...new Set([...a.keys(), ...b.keys()])].sort();

for (const key of all) {
  const ra = a.get(key), rb = b.get(key);
  if (!ra || !rb) {
    rows.push({ key, field: 'presence', reference: !!ra, target: !!rb });
    continue;
  }
  for (const field of ['x','y','width','height']) {
    const av = ra.rect[field], bv = rb.rect[field];
    if (Math.abs(av - bv) > 0.5) rows.push({ key, field: `rect.${field}`, reference: av, target: bv, delta: bv - av });
  }
  const styleKeys = new Set([...Object.keys(ra.styles || {}), ...Object.keys(rb.styles || {})]);
  for (const field of styleKeys) {
    const av = ra.styles?.[field], bv = rb.styles?.[field];
    if (av !== bv) rows.push({ key, field: `style.${field}`, reference: av, target: bv });
  }
}

console.log(JSON.stringify({ differences: rows, count: rows.length }, null, 2));
process.exit(rows.length ? 1 : 0);
