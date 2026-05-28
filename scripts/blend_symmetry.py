#!/usr/bin/env python3
"""
Exhaustively test the symmetry structure of Photoshop/FFmpeg blend modes.

Premise (Nathan's hypothesis): the blend-mode macros are related by symmetry,
so every mode should have a partner "on the other end of the spectrum".

Two natural involutions act on a blend function f(x, y) over 8-bit channels:

    SWAP   (S f)(x, y) = f(y, x)                  # swap the two layers
    DUAL   (D f)(x, y) = 255 - f(255 - x, 255 - y)  # De Morgan / complement dual

S and D commute and each is its own inverse, so {I, S, D, SD} is a Klein
four-group. Orbits under this group are the "symmetry families". A darkening
mode and its lightening mirror are a D-pair.

Test: the 8-bit modes are finite functions on [0,255]^2 (65 536 points). Build
the full table for every named mode, apply each transform, and search for the
named mode that matches. Anything a transform maps to with NO named match is a
predicted-but-unnamed mode -- the "missing end of the spectrum".
"""

M = 255

def clamp(v):
    return 0 if v < 0 else (255 if v > 255 else int(v))

# ---- Nathan's macros (B = bottom/base, L = layer/top), exact integer math ----
# Source: nathanm.com/photoshop-blending-math  (the StackOverflow answer)

def normal(B, L):     return L
def lighten(B, L):    return max(B, L)
def darken(B, L):     return min(B, L)
def multiply(B, L):   return (B * L) // 255
def average(B, L):    return (B + L) // 2
def add(B, L):        return min(255, B + L)
def subtract(B, L):   return 0 if (B + L) < 255 else (B + L - 255)   # == LinearBurn
def difference(B, L): return abs(B - L)
def negation(B, L):   return 255 - abs(255 - B - L)
def screen(B, L):     return 255 - (((255 - B) * (255 - L)) >> 8)
def exclusion(B, L):  return clamp(B + L - 2 * B * L // 255)

def overlay(B, L):
    return (2 * B * L // 255) if L < 128 else (255 - 2 * (255 - B) * (255 - L) // 255)
def hardlight(B, L):  return overlay(L, B)

def colordodge(B, L): return B if B == 255 else min(255, (L << 8) // (255 - B))
def colorburn(B, L):  return B if B == 0 else max(0, 255 - (((255 - L) << 8) // B))
def lineardodge(B, L):return add(B, L)
def linearburn(B, L): return subtract(B, L)

def linearlight(B, L):return linearburn(2 * B, L) if B < 128 else lineardodge(2 * (B - 128), L)
def vividlight(B, L): return colorburn(2 * B, L)   if B < 128 else colordodge(2 * (B - 128), L)
def pinlight(B, L):   return darken(2 * B, L)       if B < 128 else lighten(2 * (B - 128), L)
def hardmix(B, L):    return 0 if vividlight(B, L) < 128 else 255

def softlight(B, L):
    if L < 128:
        return (2 * ((B >> 1) + 64)) * L // 255
    return 255 - (2 * (255 - ((B >> 1) + 64)) * (255 - L) // 255)

def reflect(B, L):    return L if L == 255 else min(255, (B * B) // (255 - L))
def glow(B, L):       return reflect(L, B)
def phoenix(B, L):    return clamp(min(B, L) - max(B, L) + 255)   # == 255 - |B-L|

# ---- Extra modes FFmpeg added AFTER Nathan's original list -------------------
def freeze(B, L):     return 0 if B == 0 else clamp(255 - min(255, (255 - L) ** 2 // B))
def heat(B, L):       return 0 if L == 0 else clamp(255 - min(255, (255 - B) ** 2 // L))
def extremity(B, L):  return abs(255 - B - L)
def grainextract(B, L): return clamp(B - L + 128)
def grainmerge(B, L):   return clamp(B + L - 128)

NAMED = {
    "normal": normal, "lighten": lighten, "darken": darken, "multiply": multiply,
    "average": average, "add": add, "subtract/linearburn": subtract,
    "difference": difference, "negation": negation, "screen": screen,
    "exclusion": exclusion, "overlay": overlay, "hardlight": hardlight,
    "colordodge": colordodge, "colorburn": colorburn, "lineardodge": lineardodge,
    "linearlight": linearlight, "vividlight": vividlight, "pinlight": pinlight,
    "hardmix": hardmix, "softlight": softlight, "reflect": reflect, "glow": glow,
    "phoenix": phoenix, "freeze": freeze, "heat": heat, "extremity": extremity,
    "grainextract": grainextract, "grainmerge": grainmerge,
}

# ---- Build full 256x256 tables ----------------------------------------------
def table(f):
    return [[clamp(f(x, y)) for y in range(256)] for x in range(256)]

def maxdiff(t1, t2):
    return max(abs(t1[x][y] - t2[x][y]) for x in range(256) for y in range(256))

TABLES = {name: table(f) for name, f in NAMED.items()}

def swap(t):  return [[t[y][x] for y in range(256)] for x in range(256)]
def dual(t):  return [[255 - t[255 - x][255 - y] for y in range(256)] for x in range(256)]

def best_match(t):
    """Return (name, maxdiff) of the closest named mode to table t."""
    best, bestd = None, 10**9
    for name, nt in TABLES.items():
        d = maxdiff(t, nt)
        if d < bestd:
            best, bestd = name, d
    return best, bestd

print("=" * 78)
print("SWAP partner  S f(x,y)=f(y,x)   and   DUAL partner  D f(x,y)=255-f(255-x,255-y)")
print("=" * 78)
print(f"{'mode':22} {'SWAP =':22} {'DUAL =':22}")
print("-" * 78)
new_modes = {}
for name in NAMED:
    t = TABLES[name]
    sname, sd = best_match(swap(t))
    dname, dd = best_match(dual(t))
    stag = sname if sd == 0 else (f"{sname}~{sd}" if sd <= 1 else f"UNNAMED(min {sname}/{sd})")
    dtag = dname if dd == 0 else (f"{dname}~{dd}" if dd <= 1 else f"UNNAMED(min {dname}/{dd})")
    self_s = " [self]" if sname == name and sd <= 1 else ""
    self_d = " [self]" if dname == name and dd <= 1 else ""
    print(f"{name:22} {stag+self_s:22} {dtag+self_d:22}")
    if dd > 1:
        new_modes[f"DUAL({name})"] = dual(t)
    if sd > 1:
        new_modes[f"SWAP({name})"] = swap(t)

if new_modes:
    print("\n" + "=" * 78)
    print("PREDICTED MODES with NO existing named match (the 'missing ends'):")
    print("=" * 78)
    seen = []
    for label, t in new_modes.items():
        # de-dup identical predicted tables
        dup = next((s for s in seen if maxdiff(t, s[1]) == 0), None)
        if dup:
            print(f"  {label}  ==  {dup[0]}  (same function)")
            continue
        seen.append((label, t))
        corners = {(x, y): t[x][y] for x in (0, 255) for y in (0, 255)}
        mid = t[128][128]
        print(f"  {label}: corners(0,0)={corners[(0,0)]} (0,255)={corners[(0,255)]} "
              f"(255,0)={corners[(255,0)]} (255,255)={corners[(255,255)]} center={mid}")
