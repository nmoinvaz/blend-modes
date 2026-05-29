#!/usr/bin/env python3
"""Export every atlas mode's 256x256 reference LUT for the SIMD coverage test.

The atlas (blend_all.py) is the single source of truth. This writes:
  bench/ref_luts.bin   - concatenated 256x256 uint8 tables, index = a*256 + b (a=top)
  bench/ref_names.txt  - one mode name per line, in the same order
Run from the scripts/ directory; then `cd ../bench && ./build/blend_coverage`.
"""
import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(here, "blend_all.py")).read().split("# ---- big annotated chart")[0], ns)

gx = np.arange(256).reshape(256, 1) * np.ones((1, 256))   # top    A -> row index a
gy = np.ones((256, 1)) * np.arange(256).reshape(1, 256)   # bottom B -> col index b

names, blob = [], bytearray()
for name, (formula, f) in ns["MODES"].items():
    disp = name.split("*")[0].split("~")[0]
    lut = np.clip(np.round(f(gx, gy)), 0, 255).astype(np.uint8)
    names.append(disp)
    blob += lut.tobytes()

out = os.path.normpath(os.path.join(here, "..", "bench"))
open(os.path.join(out, "ref_luts.bin"), "wb").write(bytes(blob))
open(os.path.join(out, "ref_names.txt"), "w").write("\n".join(names) + "\n")
print(f"exported {len(names)} reference LUTs ({len(blob)} bytes) to {out}")
