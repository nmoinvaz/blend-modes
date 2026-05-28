#!/usr/bin/env python3
"""Can one parametric formula cover many modes? Test two unifying families.

(I) Power mean  M_p = ((A^p+B^p)/2)^(1/p)  -> the averaging spectrum.
(II) Frank t-norm  T_s(a,b) = log_s(1 + (s^a-1)(s^b-1)/(s-1))  and its dual
     t-conorm S_s = 1 - T_s(1-a,1-b).  Classic limits:
        s->0   : T=min (darken),  S=max (lighten)
        s->1   : T=product (multiply),  S=a+b-ab (screen)
        s->inf : T=max(0,a+b-1) (linearburn),  S=min(1,a+b) (addition)
     And the midpoint fuse  C_s(a,b)= a<.5 ? T_s(2a,b) : S_s(2a-1,b)
     gives pinlight / hardlight / linearlight at s = 0 / 1 / inf.
"""
import numpy as np
g = np.linspace(0, 1, 65)
A, B = np.meshgrid(g, g)

def md(x, y): return float(np.nanmax(np.abs(x - y)))

# (I) power mean limits
print("POWER MEAN  M_p:")
for p, name, tgt in [(-50, "p->-inf  -> min(darken)", np.minimum(A, B)),
                     (1, "p=1     -> average", (A + B) / 2),
                     (2, "p=2     -> rms", np.sqrt((A**2 + B**2) / 2)),
                     (50, "p->+inf -> max(lighten)", np.maximum(A, B))]:
    mp = ((A**p + B**p) / 2) ** (1 / p)
    print(f"  {name:28} max|M_p - target| = {md(mp, tgt):.4f}")

# (II) Frank t-norm / conorm
def frank_T(a, b, s):
    if abs(s - 1) < 1e-9: return a * b
    return np.log1p((np.power(s, a) - 1) * (np.power(s, b) - 1) / (s - 1)) / np.log(s)
def frank_S(a, b, s): return 1 - frank_T(1 - a, 1 - b, s)

print("\nFRANK t-NORM  T_s  (darkening continuum):")
for s, name, tgt in [(1e-4, "s->0    -> min (darken)", np.minimum(A, B)),
                     (1.0,  "s=1     -> A*B (multiply)", A * B),
                     (1e6,  "s->inf  -> max(0,A+B-1) (linearburn)", np.maximum(0, A + B - 1))]:
    T = np.clip(frank_T(A, B, s), 0, 1)
    print(f"  {name:36} max|T_s - target| = {md(T, tgt):.4f}")

print("\nFRANK t-CONORM  S_s  (lightening continuum):")
for s, name, tgt in [(1e-4, "s->0    -> max (lighten)", np.maximum(A, B)),
                     (1.0,  "s=1     -> A+B-A*B (screen)", A + B - A * B),
                     (1e6,  "s->inf  -> min(1,A+B) (addition)", np.minimum(1, A + B))]:
    S = np.clip(frank_S(A, B, s), 0, 1)
    print(f"  {name:36} max|S_s - target| = {md(S, tgt):.4f}")

print("\nFRANK midpoint-FUSE  C_s  (contrast continuum):")
def fuse(a, b, s): return np.where(a < 0.5, frank_T(2 * a, b, s), frank_S(2 * a - 1, b, s))
pinlight  = np.where(A < 0.5, np.minimum(2 * A, B), np.maximum(2 * A - 1, B))
hardlight = np.where(A < 0.5, 2 * A * B, (2 * A - 1) + B - (2 * A - 1) * B)
linearlt  = np.clip(np.where(A < 0.5, np.maximum(0, 2 * A + B - 1), np.minimum(1, 2 * A - 1 + B)), 0, 1)
for s, name, tgt in [(1e-4, "s->0    -> pinlight", pinlight),
                     (1.0,  "s=1     -> hardlight/overlay", hardlight),
                     (1e6,  "s->inf  -> linearlight", linearlt)]:
    C = np.clip(fuse(A, B, s), 0, 1)
    print(f"  {name:28} max|C_s - target| = {md(C, tgt):.4f}")

# Why NO single formula spans everything: invariants that differ
print("\nINVARIANTS that separate the species (so they cannot share one formula):")
diff = np.abs(A - B)
# monotonicity in B at fixed A: means/t-norms are non-decreasing; difference is not
mono_mean = np.all(np.diff((A + B) / 2, axis=0) >= -1e-9)
mono_diff = np.all(np.diff(diff, axis=0) >= -1e-9)
print(f"  monotone-increasing in B?   average: {mono_mean}    difference: {mono_diff}")
print("  identity element:           multiply->white(1)   screen->black(0)   normal->ignores B")
print("  continuity:                 arithmetic modes: continuous   bitwise(xor): discontinuous")
