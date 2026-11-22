#!/usr/bin/env python3
"""Convert ALEX/SIGMOD'20 raw .bin datasets to Chameleon's <double,double> .data format.

Source files are headerless, unsorted, fixed 8-byte records (no count header):
  longitudes-200M.bin  float64
  longlat-200M.bin     float64
  lognormal-190M.bin   int64
  ycsb-200M.bin        uint64

Output: strictly increasing unique float64 keys, each record written as a
(key, key) pair, i.e. what dataset_source::get_dataset<std::pair<double,double>>
expects (see include/DataSet.hpp).

Usage: python3 convert_alex_bin.py <src.bin> <dst.data> <float64|int64|uint64>
"""
import os
import sys

import numpy as np

DTYPES = {"float64": np.float64, "int64": np.int64, "uint64": np.uint64}


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 1
    src, dst, dt_name = sys.argv[1], sys.argv[2], sys.argv[3]
    if dt_name not in DTYPES:
        print(f"dtype must be one of {list(DTYPES)}", file=sys.stderr)
        return 1
    dt = DTYPES[dt_name]

    print(f"reading {src} ({os.path.getsize(src) / 1e9:.2f} GB) as {dt_name}")
    keys = np.fromfile(src, dtype=dt)
    print(f"  {len(keys)} records, raw range [{keys.min()}, {keys.max()}]")

    d = keys.astype(np.float64)
    if dt != np.float64:
        lossy = int((d > 2**53).sum()) if dt == np.uint64 else 0
        if lossy:
            print(f"  WARNING: {lossy} keys > 2^53, uint64->float64 is lossy")
    d.sort()

    keep = np.empty(len(d), dtype=bool)
    keep[0] = True
    np.not_equal(d[1:], d[:-1], out=keep[1:])
    if not keep.all():
        print(f"  dropped {len(d) - int(keep.sum())} duplicate keys after cast")
        d = d[keep]

    pairs = np.empty((len(d), 2), dtype=np.float64)
    pairs[:, 0] = d
    pairs[:, 1] = d
    pairs.tofile(dst)
    print(f"wrote {dst}: {len(d)} pairs, {os.path.getsize(dst) / 1e9:.2f} GB")

    check = np.fromfile(dst, dtype=np.float64).reshape(-1, 2)
    ck, cv = check[:, 0], check[:, 1]
    ok_sorted = bool(np.all(np.diff(ck) > 0))
    ok_vals = bool(np.array_equal(ck, cv))
    print(f"verify: strictly increasing={ok_sorted}, value==key={ok_vals}, "
          f"range=[{ck[0]}, {ck[-1]}]")
    return 0 if (ok_sorted and ok_vals) else 2


if __name__ == "__main__":
    sys.exit(main())
