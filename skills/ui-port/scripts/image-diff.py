#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageChops

p = argparse.ArgumentParser(description='Compute a simple pixel-difference score for equal-sized screenshots.')
p.add_argument('reference')
p.add_argument('target')
p.add_argument('--out', default='diff.png')
p.add_argument('--threshold', type=int, default=8, help='Per-channel difference threshold, 0-255')
a = p.parse_args()

ref = Image.open(a.reference).convert('RGB')
tgt = Image.open(a.target).convert('RGB')
if ref.size != tgt.size:
    raise SystemExit(f'image sizes differ: reference={ref.size}, target={tgt.size}')

diff = ImageChops.difference(ref, tgt)
mask = diff.point(lambda v: 255 if v > a.threshold else 0).convert('L')
changed = sum(1 for v in mask.getdata() if v)
total = ref.width * ref.height
pct = 100.0 * changed / total if total else 0.0
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
diff.save(a.out)
print(f'changed_pixels={changed}')
print(f'total_pixels={total}')
print(f'changed_percent={pct:.6f}')
print(f'diff_image={a.out}')
