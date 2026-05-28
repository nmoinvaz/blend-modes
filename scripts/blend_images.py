#!/usr/bin/env python3
"""Apply blend modes to two real photos and verify the symmetry identities
hold on actual pixels, then render a contact sheet."""
import numpy as np
from PIL import Image

A = np.asarray(Image.open("a.jpg").convert("RGB").resize((512, 512)), float)  # top  / layer L
B = np.asarray(Image.open("b.jpg").convert("RGB").resize((512, 512)), float)  # bottom / base B

def c(v): return np.clip(v, 0, 255)

# Modes as f(B, L): B = base/bottom, L = layer/top  (Nathan's convention)
multiply  = lambda B, L: B * L / 255
screen    = lambda B, L: 255 - (255 - B) * (255 - L) / 255
darken    = lambda B, L: np.minimum(B, L)
lighten   = lambda B, L: np.maximum(B, L)
add       = lambda B, L: np.minimum(255, B + L)
subtract  = lambda B, L: np.maximum(0, B + L - 255)
difference= lambda B, L: np.abs(B - L)
phoenix   = lambda B, L: np.minimum(B, L) - np.maximum(B, L) + 255
negation  = lambda B, L: 255 - np.abs(255 - B - L)
extremity = lambda B, L: np.abs(255 - B - L)
exclusion = lambda B, L: B + L - 2 * B * L / 255
reflect   = lambda B, L: c(np.where(L >= 255, L, B * B / np.maximum(255 - L, 1e-9)))
glow      = lambda B, L: reflect(L, B)
freeze    = lambda B, L: c(np.where(B <= 0, 0, 255 - np.minimum(255, (255 - L) ** 2 / np.maximum(B, 1e-9))))
heat      = lambda B, L: freeze(L, B)

# Symmetry operators on a mode function
def DUAL(f): return lambda B, L: 255 - f(255 - B, 255 - L)   # De Morgan / complement dual
def SWAP(f): return lambda B, L: f(L, B)

# The dual pairs the test predicts (lightening end, darkening end)
PAIRS = [
    ("multiply",  "screen",    multiply,  screen),
    ("darken",    "lighten",   darken,    lighten),
    ("subtract",  "add",       subtract,  add),
    ("difference","phoenix",   difference,phoenix),    # <-- Phoenix = DUAL(Difference)
    ("negation",  "extremity", negation,  extremity),
    ("reflect",   "heat",      reflect,   heat),
    ("glow",      "freeze",    glow,      freeze),
]

print(f"{'mode X':12}{'mode Y':12}  max|Y - DUAL(X)| over both photos   verdict")
print("-" * 72)
for nx, ny, fx, fy in PAIRS:
    d = np.max(np.abs(np.round(fy(B, L=A)) - np.round(DUAL(fx)(B, L=A))))
    print(f"{nx:12}{ny:12}  {d:>10.1f} {'(8-bit levels)':<18}  "
          f"{'IDENTICAL' if d <= 1 else 'differ'}")

# swap identities
print("\nSWAP checks:")
for nm, f, g in [("reflect->glow", reflect, glow), ("heat->freeze", heat, freeze)]:
    d = np.max(np.abs(np.round(SWAP(f)(B, L=A)) - np.round(g(B, L=A))))
    print(f"  SWAP({nm.split('->')[0]:8}) == {nm.split('->')[1]:8}  max diff = {d:.1f}")

# The one still-unnamed predicted mode: DUAL(exclusion)
dual_excl = DUAL(exclusion)

# ---- Contact sheet ----------------------------------------------------------
def img(arr): return Image.fromarray(c(np.round(arr)).astype(np.uint8))
panels = [("A (top)", A), ("B (bottom)", B)]
for nx, ny, fx, fy in PAIRS:
    panels.append((nx, fx(B, L=A)))
    panels.append((ny + " = DUAL", fy(B, L=A)))
panels.append(("exclusion", exclusion(B, L=A)))
panels.append(("DUAL(exclusion)*NEW", dual_excl(B, L=A)))

from PIL import ImageDraw
cols, cell = 4, 200
rows = (len(panels) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
draw = ImageDraw.Draw(sheet)
for i, (name, arr) in enumerate(panels):
    r, cc = divmod(i, cols)
    sheet.paste(img(arr).resize((cell, cell)), (cc * cell, r * (cell + 16)))
    draw.text((cc * cell + 3, r * (cell + 16) + cell + 2), name, fill="black")
sheet.save("contact_sheet.png")
print("\nWrote contact_sheet.png  ({} panels)".format(len(panels)))
