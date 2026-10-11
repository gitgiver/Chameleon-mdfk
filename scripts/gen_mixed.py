#!/usr/bin/env python3
"""Generate a dataset that is deliberately half uniform and half locally skewed, so the
ordered-vs-hash leaf decision splits it roughly half by key count.

Region A (keys in [0, A_END)) is uniformly spaced -> each leaf is locally uniform -> ordered.
Region B (keys in [A_END, A_END + B_END)) packs each bump into its first DENSE_FRAC of width,
leaving a gap -> any leaf spanning a bump sees a sharp density jump -> hash.

Both regions are generated already sorted and A < B, so the concatenation is sorted; no sort
needed. Output is binary <double key, double value> (value = key), like data_set/*.data.
"""
import sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
out = sys.argv[2] if len(sys.argv) > 2 else "data/data_set/mixed.data"

HALF = N // 2
A_END = 100_000_000.0   # uniform region spans [0, A_END)
B_END = 1_000_000.0     # skewed region spans [A_END, A_END + B_END)
K = 5_000_000           # bumps in region B (must be >> 50000 so a leaf spans several bumps)
DENSE_FRAC = 0.1        # each bump packs into the first 10% of its width, then a gap
PER_BUMP = HALF // K

a = np.arange(HALF, dtype=np.float64) * (A_END / HALF)          # region A: uniform

bump_w = B_END / K
bases = (A_END + np.arange(K, dtype=np.float64) * bump_w)[:, None]
js = (np.arange(PER_BUMP, dtype=np.float64) * (bump_w * DENSE_FRAC / PER_BUMP))[None, :]
b = (bases + js).ravel()                                        # region B: clustered

keys = np.concatenate([a, b])
assert keys.size == HALF + K * PER_BUMP
assert np.all(np.diff(keys) > 0), "keys must be strictly increasing"

rec = np.empty((keys.size, 2), dtype=np.float64)
rec[:, 0] = keys
rec[:, 1] = keys
rec.tofile(out)
print(f"wrote {keys.size} keys ({keys.size*16/1e6:.0f} MB) to {out}; "
      f"A=[0,{A_END:.0f}) uniform, B=[{A_END:.0f},{A_END+B_END:.0f}) {K} bumps x {PER_BUMP}")
