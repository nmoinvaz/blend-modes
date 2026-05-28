#!/usr/bin/env python3
"""Hypothesize new blend modes from the structure of the existing ones, and test.

Three generating families explain most existing modes and predict missing members:
  (1) Power (Holder) means  -> darken/harmonic/geometric/average are p=-inf,-1,0,1.
      Missing: p=2 (RMS) and the contraharmonic mean, between average and lighten.
  (2) Bitwise lattice -> and/or/xor exist. Missing: xnor (= dual of xor), nand, nor.
  (3) "Contrast" family = fuse(darkening_op, its dual lightening_op) at the midpoint
      -> overlay(mul,screen), linearlight, vividlight(burn,dodge), pinlight(min,max).
      The quadratic dodge/burn pair (reflect/glow/freeze/heat) has NO fused member.
      Hypothesis: "glowlight".
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
M = 255.0
def clip(v): return np.clip(v, 0, 255)
gA = np.arange(256).reshape(256, 1) * np.ones((1, 256))
gB = np.ones((256, 1)) * np.arange(256).reshape(1, 256)
s = gA + gB

# ---- (1) the power-mean spectrum --------------------------------------------
darken   = np.minimum(gA, gB)                                  # p -> -inf
harmonic = np.where(s == 0, 0, 2 * gA * gB / np.maximum(s, 1e-9))   # p = -1
geometric= np.sqrt(gA * gB)                                    # p ->  0
average  = (gA + gB) / 2                                       # p =  1
rms      = np.sqrt((gA ** 2 + gB ** 2) / 2)                    # p =  2   *NEW
contra   = np.where(s == 0, 0, (gA ** 2 + gB ** 2) / np.maximum(s, 1e-9))  # *NEW
lighten  = np.maximum(gA, gB)                                  # p -> +inf

chain = [("darken  p=-inf", darken), ("harmonic  p=-1", harmonic),
         ("geometric  p=0", geometric), ("average  p=1", average),
         ("rms  p=2 *NEW", rms), ("contraharmonic *NEW", contra),
         ("lighten  p=+inf", lighten)]
print("POWER-MEAN SPECTRUM  (each rung must be >= the previous, everywhere):")
prev = None
for name, t in chain:
    if prev is not None:
        print(f"  {name:22}  min(this - prev) = {np.min(t - prev):+.3f}   "
              f"{'monotone OK' if np.min(t - prev) >= -1e-6 else 'VIOLATION'}")
    prev = t

# ---- (2) bitwise completion: xnor = dual of xor -----------------------------
iA, iB = gA.astype(int), gB.astype(int)
xor  = iA ^ iB
xnor = 255 - xor                                               # *NEW
dual_xor = 255 - ((255 - iA) ^ (255 - iB))
print("\nBITWISE: xnor == 255 - xor ;  also xnor == DUAL(xor)?  "
      f"max|xnor - DUAL(xor)| = {np.max(np.abs(xnor - dual_xor))}")

# ---- (3) the missing contrast mode: glowlight -------------------------------
def burn_q(c, b):  return np.where(c <= 0, 0, 255 - np.minimum(255, (255 - b) ** 2 / np.maximum(c, 1e-9)))
def dodge_q(c, b): return np.where(c >= 255, 255, np.minimum(255, b ** 2 / np.maximum(255 - c, 1e-9)))
def glowlight(A, B):   # mirrors vividlight, but with the quadratic dodge/burn pair
    return np.where(A < 128, burn_q(2 * A, B), dodge_q(2 * (A - 128), B))
gl = clip(glowlight(gA, gB))
print("\nCONTRAST FAMILY completeness:")
print("  overlay=fuse(multiply,screen) | linearlight=fuse(Lburn,Ldodge) |")
print("  vividlight=fuse(colorburn,colordodge) | pinlight=fuse(min,max) |")
print("  glowlight=fuse(freeze,reflect)  <-- the previously-missing quadratic member")
print(f"  glowlight range: [{gl.min():.0f}, {gl.max():.0f}],  center px = {gl[128,128]:.0f}")

# ---- render the new candidates on the real images ---------------------------
A = np.asarray(Image.open("a.jpg").convert("RGB").resize((400, 400)), float)
B = np.asarray(Image.open("b.jpg").convert("RGB").resize((400, 400)), float)
def img(arrfn): return np.broadcast_to(0, (1,))  # placeholder

def mode_rgb(fn): return Image.fromarray(clip(np.round(fn(A, B))).astype(np.uint8))
panels = [
 ("darken",      lambda A,B: np.minimum(A,B)),
 ("harmonic",    lambda A,B: np.where(A+B==0,0,2*A*B/np.maximum(A+B,1e-9))),
 ("geometric",   lambda A,B: np.sqrt(A*B)),
 ("average",     lambda A,B: (A+B)/2),
 ("rms *NEW",    lambda A,B: np.sqrt((A**2+B**2)/2)),
 ("contraharm *NEW", lambda A,B: np.where(A+B==0,0,(A**2+B**2)/np.maximum(A+B,1e-9))),
 ("lighten",     lambda A,B: np.maximum(A,B)),
 ("xor",         lambda A,B: (A.astype(int)^B.astype(int)).astype(float)),
 ("xnor *NEW",   lambda A,B: 255.0-(A.astype(int)^B.astype(int))),
 ("vividlight",  lambda A,B: np.where(A<128, np.where(2*A<=0,0,np.maximum(0,255-(255-B)*255/np.maximum(2*A,1e-9))),
                                              np.where(2*(A-128)>=255,255,np.minimum(255,B*255/np.maximum(255-2*(A-128),1e-9))))),
 ("glowlight *NEW", glowlight),
]
cell, lblh, cols = 200, 26, 4
rows = (len(panels)+cols-1)//cols
def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf","/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
fn = font(15)
sheet = Image.new("RGB",(cols*cell, rows*(cell+lblh)),(245,245,245))
d = ImageDraw.Draw(sheet)
for i,(name,f) in enumerate(panels):
    r,c = divmod(i,cols); x0,y0 = c*cell, r*(cell+lblh)
    sheet.paste(mode_rgb(f).resize((cell,cell)),(x0,y0))
    hi = "NEW" in name
    d.rectangle([x0,y0+cell,x0+cell,y0+cell+lblh], fill=(255,244,200) if hi else (255,255,255))
    d.text((x0+5,y0+cell+5), name, fill=(160,0,0) if hi else (0,0,0), font=fn)
sheet.save("hypotheses.png")
print("\nWrote hypotheses.png")
