#!/usr/bin/env python3
"""Investigate the two big gaps:
   (A) other t-norm families (Frank is only one path min->product->Lukasiewicz);
   (B) the classical-means zoo beyond the 5 power means FFmpeg samples.
Inputs a,b in [0,1]."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
g = np.linspace(0, 1, 257); a, b = np.meshgrid(g, g)
def mx(x, y): return float(np.nanmax(np.abs(np.clip(x,0,1) - np.clip(y,0,1))))
def W(c, x, y): return np.where(c, x, y)

# ===================== (A) t-NORM FAMILIES ===================================
def frank(a, b, s):
    if abs(s-1) < 1e-9: return a*b
    return np.log1p((np.power(s,a)-1)*(np.power(s,b)-1)/(s-1))/np.log(s)
def hamacher(a, b, gm):                       # gm=1 product, 0 Hamacher-prod, 2 Einstein
    den = gm + (1-gm)*(a+b-a*b)
    return np.where(den <= 0, 0.0, a*b/np.maximum(den, 1e-12))
def yager(a, b, p):  return np.maximum(0, 1-((1-a)**p+(1-b)**p)**(1/p))   # p=1 Luk, p->inf min
def schweizer(a, b, p): return np.maximum(0, a**p+b**p-1)**(1/p)          # p->0 product, 1 Luk

einstein_prod = a*b/(1+(1-a)*(1-b))           # = hamacher gm=2
einstein_sum  = (a+b)/(1+a*b)                 # its conorm
ham_prod      = np.where(a+b-a*b<=0, 0, a*b/np.maximum(a+b-a*b,1e-12))

print("=== (A) t-NORM FAMILIES — all hit the same corners, differ in between ===")
print(" Frank   : s->0 min  | s=1 product | s->inf Lukasiewicz")
print(f"   s=1 == product?            {mx(frank(a,b,1.0), a*b):.4f}")
print(f"   s=1e6 == max(0,a+b-1)?     {mx(np.clip(frank(a,b,1e6),0,1), np.maximum(0,a+b-1)):.4f}  (->0 in limit)")
print(" Hamacher: gm=1 product | gm=2 Einstein product | gm=0 Hamacher product")
print(f"   gm=1 == product?           {mx(hamacher(a,b,1), a*b):.4f}")
print(f"   gm=2 == Einstein ab/(1+(1-a)(1-b))? {mx(hamacher(a,b,2), einstein_prod):.4f}")
print(" Yager   : p=1 Lukasiewicz | p->inf min")
print(f"   p=1 == max(0,a+b-1)?       {mx(yager(a,b,1), np.maximum(0,a+b-1)):.4f}")
print(" Schweizer-Sklar: p->0 product | p=1 Lukasiewicz")
print(f"   p=1 == max(0,a+b-1)?       {mx(schweizer(a,b,1), np.maximum(0,a+b-1)):.4f}")

# Is Einstein product just Frank at some s?  Scan s -> show no match => new mode
ss = np.concatenate([np.linspace(1e-3,1,40), np.linspace(1,1000,40)])
best = min(mx(einstein_prod, np.clip(frank(a,b,s),0,1)) for s in ss)
print(f"\n Einstein product vs BEST-fit Frank over s in [1e-3,1000]: min max-diff = {best:.4f}")
print(" => Einstein product is NOT any Frank member: a genuinely distinct soft-multiply mode.")
print(" Concrete addable modes: einstein_prod ab/(1+(1-a)(1-b)), einstein_sum (a+b)/(1+ab),")
print("                         hamacher_prod ab/(a+b-ab).")

# ===================== (B) MEANS ZOO =========================================
eps = 1e-9
A2, B2 = np.maximum(a,eps), np.maximum(b,eps)
power   = lambda p: ((a**p+b**p)/2)**(1/p)
lehmer  = lambda p: (a**p+b**p)/np.maximum(a**(p-1)+b**(p-1), eps)
darken, lighten = np.minimum(a,b), np.maximum(a,b)
harmonic  = W(a+b==0, 0, 2*a*b/np.maximum(a+b,eps))
geometric = np.sqrt(a*b)
arithmetic= (a+b)/2
rms       = np.sqrt((a**2+b**2)/2)
contra    = lehmer(2)                                   # Lehmer p=2
logm      = W(np.abs(a-b)<1e-9, a, (a-b)/(np.log(A2)-np.log(B2)))
identric  = W(np.abs(a-b)<1e-9, a, np.exp(-1+(A2*np.log(A2)-B2*np.log(B2))/(a-b)))
heronian  = (a+np.sqrt(a*b)+b)/3
centroid  = W(a+b==0, 0, 2*(a**2+a*b+b**2)/np.maximum(3*(a+b),eps))

zoo = [("darken",darken),("harmonic",harmonic),("geometric",geometric),
       ("logarithmic",logm),("Heronian",heronian),("identric",identric),
       ("arithmetic",arithmetic),("rms",rms),("centroidal",centroid),
       ("contraharmonic",contra),("lighten",lighten)]
print("\n=== (B) MEANS ZOO — verify the ordering chain (each <= next everywhere) ===")
prev=None
for name,t in zoo:
    if prev is not None:
        viol = float(np.min(t-prevt))
        print(f"  {name:16} >= prev ? min(this-prev)={viol:+.4f}  {'OK' if viol>=-1e-6 else 'CROSSES'}")
    prevt=t; prev=name
print("  FFmpeg ships darken/harmonic/geometric/arithmetic/lighten; the rest are unused rungs.")

# ===================== render two figures ====================================
A = np.asarray(Image.open("a.jpg").convert("RGB").resize((360,360)),float)/255
B = np.asarray(Image.open("b.jpg").convert("RGB").resize((360,360)),float)/255
def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf","/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p,sz)
        except Exception: pass
    return ImageFont.load_default()
F=font(15)
def sheet(name, panels, cols, fn):
    cell,lbl=180,24; rows=(len(panels)+cols-1)//cols
    im=Image.new("RGB",(cols*cell,rows*(cell+lbl)),(245,245,245)); d=ImageDraw.Draw(im)
    for i,(t,arr) in enumerate(panels):
        r,c=divmod(i,cols); x,y=c*cell,r*(cell+lbl)
        px=Image.fromarray((np.clip(arr,0,1)*255).round().astype(np.uint8)).resize((cell,cell))
        im.paste(px,(x,y))
        hi="NEW" in t
        d.rectangle([x,y+cell,x+cell,y+cell+lbl],fill=(255,244,200) if hi else (255,255,255))
        d.text((x+4,y+cell+4),t,fill=(160,0,0) if hi else (0,0,0),font=F)
    im.save(name); print("wrote",name)

# t-norm figure: darken/multiply families and conorm/lighten families
tn=[("darken (min)",np.minimum(A,B)),("multiply (Frank s=1)",A*B),
    ("Hamacher prod *NEW",A*B/np.maximum(A+B-A*B,1e-9)),("Einstein prod *NEW",A*B/(1+(1-A)*(1-B))),
    ("lighten (max)",np.maximum(A,B)),("screen (Frank s=1)",A+B-A*B),
    ("Hamacher sum *NEW",1-(1-A)*(1-B)/np.maximum((1-A)+(1-B)-(1-A)*(1-B),1e-9)),
    ("Einstein sum *NEW",(A+B)/(1+A*B))]
sheet("tnorms.png", tn, 4, F)

# means figure: full ordered spectrum on the dog
def m(f): return f(A,B)
mns=[("darken",np.minimum(A,B)),("harmonic",np.where(A+B==0,0,2*A*B/np.maximum(A+B,1e-9))),
     ("geometric",np.sqrt(A*B)),
     ("logarithmic *NEW",np.where(np.abs(A-B)<1e-6,A,(A-B)/(np.log(np.maximum(A,1e-9))-np.log(np.maximum(B,1e-9))))),
     ("Heronian *NEW",(A+np.sqrt(A*B)+B)/3),
     ("identric *NEW",np.where(np.abs(A-B)<1e-6,A,np.exp(-1+(np.maximum(A,1e-9)*np.log(np.maximum(A,1e-9))-np.maximum(B,1e-9)*np.log(np.maximum(B,1e-9)))/(A-B)))),
     ("arithmetic",(A+B)/2),("rms *NEW",np.sqrt((A**2+B**2)/2)),
     ("centroidal *NEW",np.where(A+B==0,0,2*(A**2+A*B+B**2)/np.maximum(3*(A+B),1e-9))),
     ("contraharm *NEW",np.where(A+B==0,0,(A**2+B**2)/np.maximum(A+B,1e-9))),
     ("lighten",np.maximum(A,B))]
sheet("means.png", mns, 4, F)
