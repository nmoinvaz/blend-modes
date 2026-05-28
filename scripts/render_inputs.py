#!/usr/bin/env python3
"""Compare input choices for showcasing blend modes. Rows = input pairs,
columns = A, B, then diagnostic modes. The clearest 'math view' is a 2D ramp
(A = horizontal 0..255, B = vertical 0..255): the output IS the mode's lookup table."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
N = 256
X = np.tile(np.arange(N), (N, 1)).astype(float)            # horizontal ramp
Y = np.tile(np.arange(N).reshape(-1, 1), (1, N)).astype(float)  # vertical ramp

def clip(v): return np.clip(v, 0, 255)
multiply   = lambda A, B: A * B / 255
screen     = lambda A, B: 255 - (255 - A) * (255 - B) / 255
difference = lambda A, B: np.abs(A - B)
overlay    = lambda A, B: np.where(A < 128, 2 * A * B / 255, 255 - 2 * (255 - A) * (255 - B) / 255)
reflect    = lambda A, B: np.where(B >= 255, B, np.minimum(255, A * A / np.maximum(255 - B, 1e-9)))
MODES = [("multiply", multiply), ("screen", screen), ("difference", difference),
         ("overlay", overlay), ("reflect", reflect)]

photo  = np.asarray(Image.open("a.jpg").convert("RGB").resize((N, N)), float)   # landscape
dog    = np.asarray(Image.open("b.jpg").convert("RGB").resize((N, N)), float)   # current B
gray_A = np.dstack([X, X, X])                       # horizontal grey ramp
gray_B = np.dstack([Y, Y, Y])                       # vertical grey ramp
hsv    = np.dstack([X, np.full((N, N), 255.0), Y]).astype(np.uint8)
color  = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"), float)  # hue x value chart

PAIRS = [
    ("current: photo + photo", photo, dog),
    ("RAMP CHART (math view)", gray_A, gray_B),
    ("color gradient + photo", color, photo),
]

cell, lbl, rowlbl = 150, 20, 150
cols = 2 + len(MODES)
W = rowlbl + cols * cell
Hpx = len(PAIRS) * (cell + lbl) + 24
sheet = Image.new("RGB", (W, Hpx), (245, 245, 245))
d = ImageDraw.Draw(sheet)
def font(s):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, s)
        except Exception: pass
    return ImageFont.load_default()
F = font(14)
def put(img, x, y, name):
    sheet.paste(Image.fromarray(clip(np.round(img)).astype(np.uint8)).resize((cell, cell)), (x, y))
    d.text((x + 3, y + cell + 3), name, fill=(0, 0, 0), font=F)

# column headers
d.text((rowlbl + 3, 4), "A", fill=(0,0,0), font=F)
d.text((rowlbl + cell + 3, 4), "B", fill=(0,0,0), font=F)
for j, (mn, _) in enumerate(MODES):
    d.text((rowlbl + (2 + j) * cell + 3, 4), mn, fill=(0,0,0), font=F)

for i, (label, A, B) in enumerate(PAIRS):
    y = 24 + i * (cell + lbl)
    d.text((4, y + cell // 2), label, fill=(0, 0, 90), font=F)
    put(A, rowlbl, y, "")
    put(B, rowlbl + cell, y, "")
    for j, (mn, f) in enumerate(MODES):
        put(f(A, B), rowlbl + (2 + j) * cell, y, "")

sheet.save("input_comparison.png")
print("wrote input_comparison.png", sheet.size)
