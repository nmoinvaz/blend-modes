#!/usr/bin/env python3
"""For every mode in the atlas, decide whether it has a symmetric partner.
Two involutions:  DUAL D f = 255-f(255-A,255-B)   and   SWAP S f = f(B,A).
Report: self-symmetric / pairs-with-X / MISSING, with exact mismatch and the
number of differing pixels (so boundary-only ties are flagged, not miscalled)."""
import numpy as np
M, H = 255.0, 128.0
def clip(v): return np.clip(v, 0, 255)
def dodge(A, B): return np.where(A >= M, M, np.minimum(M, B * M / np.maximum(M - A, 1e-9)))
def burn(A, B):  return np.where(A <= 0, 0, np.maximum(0, M - (M - B) * M / np.maximum(A, 1e-9)))

MODES = {
 "normal":      lambda A, B: A,
 "multiply":    lambda A, B: A * B / M,
 "screen":      lambda A, B: M - (M - A) * (M - B) / M,
 "darken":      lambda A, B: np.minimum(A, B),
 "lighten":     lambda A, B: np.maximum(A, B),
 "burn":        burn,
 "dodge":       dodge,
 "subtract":    lambda A, B: np.maximum(0, A - B),
 "addition":    lambda A, B: np.minimum(M, A + B),
 "difference":  lambda A, B: np.abs(A - B),
 "phoenix":     lambda A, B: np.minimum(A, B) - np.maximum(A, B) + M,
 "negation":    lambda A, B: M - np.abs(M - A - B),
 "extremity":   lambda A, B: np.abs(M - A - B),
 "exclusion":   lambda A, B: A + B - 2 * A * B / M,
 "inclusion":   lambda A, B: M - A - B + 2 * A * B / M,
 "reflect":     lambda A, B: np.where(B >= M, B, np.minimum(M, A * A / np.maximum(M - B, 1e-9))),
 "glow":        lambda A, B: np.where(A >= M, A, np.minimum(M, B * B / np.maximum(M - A, 1e-9))),
 "heat":        lambda A, B: np.where(A <= 0, 0, M - np.minimum(M, (M - B) ** 2 / np.maximum(A, 1e-9))),
 "freeze":      lambda A, B: np.where(B <= 0, 0, M - np.minimum(M, (M - A) ** 2 / np.maximum(B, 1e-9))),
 "bleach":      lambda A, B: (M - B) + (M - A) - M,
 "stain":       lambda A, B: 2 * M - A - B,
 "overlay":     lambda A, B: np.where(A < H, 2 * A * B / M, M - 2 * (M - A) * (M - B) / M),
 "hardlight":   lambda A, B: np.where(B < H, 2 * A * B / M, M - 2 * (M - A) * (M - B) / M),
 "softlight":   lambda A, B: A * A / M + 2 * (B * (A * (M - A) / M) / M),
 "vividlight":  lambda A, B: np.where(A < H, burn(2 * A, B), dodge(2 * (A - H), B)),
 "linearlight": lambda A, B: clip(np.where(A < H, B + 2 * A - M, B + 2 * (A - H))),
 "pinlight":    lambda A, B: np.where(B < H, np.minimum(A, 2 * B), np.maximum(A, 2 * (B - H))),
 "hardmix":     lambda A, B: np.where(A < (M - B), 0.0, M),
 "average":     lambda A, B: (A + B) / 2,
 "geometric":   lambda A, B: np.sqrt(A * B),
 "harmonic":    lambda A, B: np.where((A + B) == 0, 0, 2 * A * B / np.maximum(A + B, 1e-9)),
 "grainextract":lambda A, B: clip(H + A - B),
 "grainmerge":  lambda A, B: clip(A + B - H),
 "divide":      lambda A, B: clip(np.where(B == 0, M, M * A / np.maximum(B, 1e-9))),
}

gA = np.arange(256).reshape(256, 1) * np.ones((1, 256))
gB = np.ones((256, 1)) * np.arange(256).reshape(1, 256)
TAB = {n: np.round(clip(f(gA, gB))) for n, f in MODES.items()}
N = 256 * 256

def dual(t): return np.round(255 - t[::-1, ::-1])
def swap(t): return t.T

def classify(self_name, transformed):
    """Return (verdict, partner, maxdiff, ndiff_pixels)."""
    best, bd = None, 1e9
    for n, nt in TAB.items():
        d = np.max(np.abs(transformed - nt))
        if d < bd: best, bd = n, d
    nd = int(np.sum(transformed != TAB[best]))
    selfd = np.max(np.abs(transformed - TAB[self_name]))
    self_nd = int(np.sum(transformed != TAB[self_name]))
    if selfd <= 1:
        return ("self", self_name, selfd, self_nd)
    # self-symmetric except a thin boundary (<1% of pixels)?
    if self_nd < N * 0.01:
        return ("self*", self_name, selfd, self_nd)
    if bd <= 1 and best != self_name:
        return ("pair", best, bd, nd)
    return ("MISSING", best, bd, nd)

print(f"{'mode':13} | {'DUAL partner':26} | {'SWAP partner':22}")
print("-" * 70)
dual_missing, swap_missing = [], []
for n in MODES:
    dv, dp, dd, dn = classify(n, dual(TAB[n]))
    sv, sp, sd, sn = classify(n, swap(TAB[n]))
    def fmt(v, p, d, nd):
        if v == "self":    return "self-dual" if d == 0 else f"self-dual(±{int(d)})"
        if v == "self*":   return f"self ({nd}px differ on boundary)"
        if v == "pair":    return f"{p}" + ("" if d == 0 else f" (±{int(d)})")
        return f"MISSING (nearest {p}, Δ{int(d)})"
    dtag = fmt(dv, dp, dd, dn)
    # for SWAP reuse same formatter but 'self' wording -> swap-symmetric
    if sv == "self":   stag = "swap-symmetric" if sd == 0 else f"swap-sym(±{int(sd)})"
    elif sv == "self*":stag = f"swap-sym ({sn}px)"
    elif sv == "pair": stag = f"{sp}" + ("" if sd == 0 else f" (±{int(sd)})")
    else:              stag = f"MISSING (≈{sp},Δ{int(sd)})"
    print(f"{n:13} | {dtag:26} | {stag:22}")
    if dv == "MISSING": dual_missing.append(n)
    if sv == "MISSING": swap_missing.append(n)

print("\nDUAL partner MISSING from named set:", ", ".join(dual_missing) or "none")
print("SWAP partner MISSING from named set:", ", ".join(swap_missing) or "none")
