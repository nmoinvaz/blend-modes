#!/usr/bin/env python3
"""Compare candidate B-inputs for the color-gradient reference (A = hue x value)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
N = 256
X = np.tile(np.arange(N), (N, 1)).astype(float)
Y = np.tile(np.arange(N).reshape(-1, 1), (1, N)).astype(float)
def clip(v): return np.clip(v, 0, 255)

A = np.asarray(Image.fromarray(np.dstack([X, np.full((N, N), 255.0), Y]).astype(np.uint8), "HSV").convert("RGB"), float)

landscape = np.asarray(Image.open("a.jpg").convert("RGB").resize((N, N)), float)
vramp = np.dstack([Y, Y, Y])                                   # vertical grey ramp
huev = np.asarray(Image.fromarray(np.dstack([Y, np.full((N, N), 255.0), np.full((N, N), 255.0)]).astype(np.uint8), "HSV").convert("RGB"), float)  # vertical hue
cx, cy = N / 2, N / 2
radial = clip(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / (N / 2) * 255)
radial = np.dstack([radial, radial, radial])
dots = np.full((N, N, 3), 235.0)                               # polka dots: light bg, dark dots
yy, xx = np.mgrid[0:N, 0:N]
mask = ((xx % 48 - 24) ** 2 + (yy % 48 - 24) ** 2) < 16 ** 2
dots[mask] = 40.0
CANDS = [("landscape", landscape), ("polka dots", dots), ("vert ramp", vramp),
         ("radial", radial), ("vert hue", huev)]

multiply = lambda A, B: A * B / 255
screen   = lambda A, B: 255 - (255 - A) * (255 - B) / 255
overlay  = lambda A, B: np.where(A < 128, 2 * A * B / 255, 255 - 2 * (255 - A) * (255 - B) / 255)
difference = lambda A, B: np.abs(A - B)
MODES = [("multiply", multiply), ("screen", screen), ("overlay", overlay), ("difference", difference)]

big, tiny, lbl, rl = 96, 32, 18, 90
cols = 2 + len(MODES)                       # B preview, 32px preview, then modes
W = rl + cols * big
Hpx = len(CANDS) * (big + lbl) + 20
sheet = Image.new("RGB", (W, Hpx), (245, 245, 245))
d = ImageDraw.Draw(sheet)
def font(s):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, s)
        except Exception: pass
    return ImageFont.load_default()
F = font(13)
def put(arr, x, y, w, name=""):
    sheet.paste(Image.fromarray(clip(np.round(arr)).astype(np.uint8)).resize((w, w)), (x, y))
    if name: d.text((x + 2, y + w + 2), name, fill=(0, 0, 0), font=F)

d.text((rl + 2, 4), "B input", fill=(0,0,0), font=F)
d.text((rl + big + 2, 4), "result @32px", fill=(0,0,0), font=F)
for j, (mn, _) in enumerate(MODES):
    d.text((rl + (2 + j) * big + 2, 4), mn, fill=(0,0,0), font=F)
for i, (cname, B) in enumerate(CANDS):
    y = 20 + i * (big + lbl)
    d.text((4, y + big // 2), cname, fill=(0, 0, 90), font=F)
    put(B, rl, y, big)
    put(multiply(A, B), rl + big, y, tiny)        # show how multiply looks at final 32px
    for j, (mn, f) in enumerate(MODES):
        put(f(A, B), rl + (2 + j) * big, y, big)
sheet.save("grad_compare.png")
print("wrote grad_compare.png", sheet.size)
