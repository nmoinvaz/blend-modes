#!/usr/bin/env python3
"""
Reconcile the symmetry analysis against the CURRENT FFmpeg mode set (which has
grown: bleach, stain, softdifference, geometric, harmonic, interpolate,
hardoverlay, divide ...), then render one big annotated chart of every mode
applied to two real photos, labelled with name + math, grouped so symmetry
duals sit side by side.

Conventions match FFmpeg blend_modes.c exactly:  A = top,  B = bottom,
MAX = 255, HALF = 128.  Modes are evaluated in float to keep the algebraic
symmetry exact, then rounded for display / comparison.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

M, H = 255.0, 128.0
def clip(v): return np.clip(v, 0, 255)

# ---- every current FFmpeg blend mode, f(A, B), A=top B=bottom ----------------
def dodge(A, B): return np.where(A >= M, M, np.minimum(M, B * M / np.maximum(M - A, 1e-9)))
def burn(A, B):  return np.where(A <= 0, 0, np.maximum(0, M - (M - B) * M / np.maximum(A, 1e-9)))

MODES = {
 # name           formula string (for the chart)            lambda
 "normal":        ("A",                                     lambda A, B: A),
 "multiply":      ("A·B / 255",                             lambda A, B: A * B / M),
 "screen":        ("255 − (255−A)(255−B)/255",              lambda A, B: M - (M - A) * (M - B) / M),
 "darken":        ("min(A, B)",                             lambda A, B: np.minimum(A, B)),
 "lighten":       ("max(A, B)",                             lambda A, B: np.maximum(A, B)),
 "burn":          ("255 − (255−B)·255/A",                   burn),
 "dodge":         ("B·255 / (255−A)",                       dodge),
 "subtract":      ("max(0, A − B)",                         lambda A, B: np.maximum(0, A - B)),
 "addition":      ("min(255, A + B)",                       lambda A, B: np.minimum(M, A + B)),
 "difference":    ("|A − B|",                               lambda A, B: np.abs(A - B)),
 "phoenix":       ("255 − |A − B|",                         lambda A, B: np.minimum(A, B) - np.maximum(A, B) + M),
 "negation":      ("255 − |255 − A − B|",                   lambda A, B: M - np.abs(M - A - B)),
 "extremity":     ("|255 − A − B|",                         lambda A, B: np.abs(M - A - B)),
 "exclusion":     ("A + B − 2·A·B/255",                     lambda A, B: A + B - 2 * A * B / M),
 "inclusion*NEW": ("255 − A − B + 2·A·B/255",               lambda A, B: M - A - B + 2 * A * B / M),
 "reflect":       ("A² / (255 − B)",                        lambda A, B: np.where(B >= M, B, np.minimum(M, A * A / np.maximum(M - B, 1e-9)))),
 "glow":          ("B² / (255 − A)",                        lambda A, B: np.where(A >= M, A, np.minimum(M, B * B / np.maximum(M - A, 1e-9)))),
 "heat":          ("255 − (255−B)²/A",                      lambda A, B: np.where(A <= 0, 0, M - np.minimum(M, (M - B) ** 2 / np.maximum(A, 1e-9)))),
 "freeze":        ("255 − (255−A)²/B",                      lambda A, B: np.where(B <= 0, 0, M - np.minimum(M, (M - A) ** 2 / np.maximum(B, 1e-9)))),
 "bleach":        ("255 − A − B",                           lambda A, B: (M - B) + (M - A) - M),
 "stain":         ("510 − A − B",                           lambda A, B: 2 * M - A - B),
 "overlay":       ("A<128: 2AB/255  else  scrn",            lambda A, B: np.where(A < H, 2 * A * B / M, M - 2 * (M - A) * (M - B) / M)),
 "hardlight":     ("overlay(B, A)",                         lambda A, B: np.where(B < H, 2 * A * B / M, M - 2 * (M - A) * (M - B) / M)),
 "softlight":     ("A²/255 + 2B·A(255−A)/255²",             lambda A, B: A * A / M + 2 * (B * (A * (M - A) / M) / M)),
 "vividlight":    ("A<128: burn(2A,B) else dodge",          lambda A, B: np.where(A < H, burn(2 * A, B), dodge(2 * (A - H), B))),
 "linearlight":   ("B + 2A − 255",                          lambda A, B: clip(np.where(A < H, B + 2 * A - M, B + 2 * (A - H)))),
 "pinlight":      ("B<128: min(A,2B) else max",             lambda A, B: np.where(B < H, np.minimum(A, 2 * B), np.maximum(A, 2 * (B - H)))),
 "hardmix":       ("A < 255−B ? 0 : 255",                   lambda A, B: np.where(A < (M - B), 0.0, M)),
 "average":       ("(A + B) / 2",                           lambda A, B: (A + B) / 2),
 "geometric":     ("√(A·B)",                                lambda A, B: np.sqrt(A * B)),
 "harmonic":      ("2·A·B / (A + B)",                       lambda A, B: np.where((A + B) == 0, 0, 2 * A * B / np.maximum(A + B, 1e-9))),
 "grainextract":  ("128 + A − B",                           lambda A, B: clip(H + A - B)),
 "grainmerge":    ("A + B − 128",                           lambda A, B: clip(A + B - H)),
 "divide":        ("255·A / B",                             lambda A, B: clip(np.where(B == 0, M, M * A / np.maximum(B, 1e-9)))),
}

# ---- fill in the 6 modes whose DUAL partner was missing from the named set ---
# Each is exactly the De Morgan dual D·f = 255 − f(255−A, 255−B) of its base.
def _D(name):
    f = MODES[name][1]
    return lambda A, B: 255 - f(255 - A, 255 - B)
MODES["linearburn*NEW"] = ("max(0, A+B−255)  =dual·add",   _D("addition"))
MODES["lift*NEW"]    = ("255 − max(0, B−A)  =dual·sub", _D("subtract"))
MODES["mirage*NEW"]  = ("255 − vivid(255−A,255−B)",     _D("vividlight"))
MODES["sheen*NEW"]    = ("255 − √((255−A)(255−B))",      _D("geometric"))
MODES["bloom*NEW"]   = ("255 − 2(255−A)(255−B)/(510−A−B)", _D("harmonic"))
MODES["quench*NEW"]    = ("255 − 255(255−A)/(255−B)",     _D("divide"))

# ---- hypothesized modes, predicted from the generating families (speculative)
def _bq(c, b): return np.where(c <= 0, 0, 255 - np.minimum(255, (255 - b) ** 2 / np.maximum(c, 1e-9)))
def _dq(c, b): return np.where(c >= 255, 255, np.minimum(255, b ** 2 / np.maximum(255 - c, 1e-9)))
MODES["rms~HYPO"]        = ("√((A²+B²)/2)  power-mean p=2",  lambda A, B: np.sqrt((A ** 2 + B ** 2) / 2))
MODES["contraharm~HYPO"] = ("(A²+B²)/(A+B)  contraharmonic", lambda A, B: np.where(A + B == 0, 0, (A ** 2 + B ** 2) / np.maximum(A + B, 1e-9)))
MODES["glowlight~HYPO"]  = ("A<128: freeze(2A,B) else reflect", lambda A, B: np.where(A < 128, _bq(2 * A, B), _dq(2 * (A - 128), B)))

# bitwise family: and/or/xor exist in FFmpeg; nand/nor/xnor are the missing complements
MODES["and"]      = ("A & B",         lambda A, B: (A.astype(int) & B.astype(int)).astype(float))
MODES["or"]       = ("A | B",         lambda A, B: (A.astype(int) | B.astype(int)).astype(float))
MODES["xor"]      = ("A ^ B",         lambda A, B: (A.astype(int) ^ B.astype(int)).astype(float))
MODES["nand*NEW"] = ("255 − (A & B)", lambda A, B: 255.0 - (A.astype(int) & B.astype(int)))
MODES["nor*NEW"]  = ("255 − (A | B)", lambda A, B: 255.0 - (A.astype(int) | B.astype(int)))
MODES["xnor*NEW"] = ("255 − (A ^ B)", lambda A, B: 255.0 - (A.astype(int) ^ B.astype(int)))

# ---- gap modes surfaced by the family investigation (added) -----------------
def _logm(A, B):
    a, b = A / 255.0, B / 255.0
    out = np.where(np.abs(a - b) < 1e-6, a, (a - b) / (np.log(np.maximum(a, 1e-9)) - np.log(np.maximum(b, 1e-9))))
    return 255 * np.clip(out, 0, 1)
def _identric(A, B):
    a, b = np.maximum(A / 255.0, 1e-9), np.maximum(B / 255.0, 1e-9)
    out = np.where(np.abs(a - b) < 1e-6, a, np.exp(-1 + (a * np.log(a) - b * np.log(b)) / (a - b)))
    return 255 * np.clip(out, 0, 1)
def _norm(A, B, f):  # evaluate a [0,1] binary op on normalized inputs, return [0,255]
    a, b = A / 255.0, B / 255.0
    return 255 * np.clip(f(a, b), 0, 1)
MODES["logarithmic*NEW"] = ("(A−B)/(ln A−ln B)",          _logm)                    # mean
MODES["heronian*NEW"]    = ("(A+√(AB)+B)/3",              lambda A, B: (A + np.sqrt(A * B) + B) / 3)
MODES["identric*NEW"]    = ("(1/e)(Aᴬ/Bᴮ)^(1/(A−B))",     _identric)
MODES["centroidal*NEW"]  = ("2(A²+AB+B²)/(3(A+B))",       lambda A, B: np.where(A + B == 0, 0, 2 * (A**2 + A*B + B**2) / np.maximum(3 * (A + B), 1e-9)))
MODES["einprod*NEW"]     = ("Einstein:  ab/(1+(1−a)(1−b))", lambda A, B: _norm(A, B, lambda a, b: a * b / (1 + (1 - a) * (1 - b))))
MODES["einsum*NEW"]      = ("Einstein:  (a+b)/(1+ab)",      lambda A, B: _norm(A, B, lambda a, b: (a + b) / (1 + a * b)))
MODES["hamprod*NEW"]     = ("Hamacher:  ab/(a+b−ab)",       lambda A, B: _norm(A, B, lambda a, b: np.where(a + b - a*b <= 0, 0, a * b / np.maximum(a + b - a*b, 1e-9))))
MODES["hamsum*NEW"]      = ("Hamacher:  dual of ab/(a+b−ab)", lambda A, B: _norm(A, B, lambda a, b: 1 - np.where((1-a)+(1-b)-(1-a)*(1-b) <= 0, 0, (1-a)*(1-b) / np.maximum((1-a)+(1-b)-(1-a)*(1-b), 1e-9))))

# ---- existing FFmpeg modes that were not yet in the atlas -------------------
MODES["multiply128"]   = ("(A−128)·B/32 + 128",            lambda A, B: clip((A - 128) * B / 32 + 128))
MODES["softdifference"]= ("contrast-stretched |A−B|",      lambda A, B: clip(np.where(A > B, np.where(B >= 255, 0, (A - B) * 255 / np.maximum(255 - B, 1e-9)), np.where(B <= 0, 0, (B - A) * 255 / np.maximum(B, 1e-9)))))
MODES["interpolate"]   = ("¼(2 − cos πA/255 − cos πB/255)", lambda A, B: 255 * (2 - np.cos(A * np.pi / 255) - np.cos(B * np.pi / 255)) * 0.25)
MODES["hardoverlay"]   = ("hardlight via dodge/half-mult",  lambda A, B: np.where(A >= 255, 255.0, np.minimum(255, np.where(A > 128, 255 * B / np.maximum(510 - 2 * A, 1e-9), 2 * A * B / 255))))

# ---- hybrid / extension modes from the family analysis (hypotheses) ---------
MODES["hamxor~HYPO"] = ("A+B−2·hamacher(A,B)  (bitwise×t-norm)", lambda A, B: clip(A + B - 2 * _norm(A, B, lambda a, b: np.where(a + b - a*b <= 0, 0, a * b / np.maximum(a + b - a*b, 1e-9)))))
MODES["dodge3~HYPO"] = ("B³/(255−A)²  dodge k=3",          lambda A, B: _norm(A, B, lambda a, b: np.minimum(1, b**3 / np.maximum(1 - a, 1e-9))))
MODES["burn3~HYPO"]  = ("1 − (255−B)³/A²  burn k=3",       lambda A, B: _norm(A, B, lambda a, b: np.maximum(0, 1 - (1 - b)**3 / np.maximum(a, 1e-9))))
# missing pairs spotted in the atlas: Yager t-norm/conorm (multiplicative), and the
# dual of multiply128 (linear) which had no named partner.
MODES["yagerprod*NEW"] = ("Yager:  max(0,1−√((1−a)²+(1−b)²))", lambda A, B: _norm(A, B, lambda a, b: np.maximum(0, 1 - np.sqrt((1 - a)**2 + (1 - b)**2))))
MODES["yagersum*NEW"]  = ("Yager:  min(1,√(a²+b²))",          lambda A, B: _norm(A, B, lambda a, b: np.minimum(1, np.sqrt(a**2 + b**2))))
MODES["screen128*NEW"] = ("255 − mul128(255−A,255−B)",        lambda A, B: clip(255 - clip((127 - A) * (255 - B) / 32 + 128)))
# audit found these singles were unpaired (distinct unnamed dual) -> add the dual partners
MODES["embers*NEW"] = ("255 − softdiff(255−A,255−B)",   _D("softdifference"))
MODES["veil*NEW"]   = ("255 − hardoverlay(255−A,255−B)", _D("hardoverlay"))
MODES["afterglow~HYPO"]    = ("255 − glowlight(255−A,255−B)",  _D("glowlight~HYPO"))
MODES["rift~HYPO"]  = ("255 − hamxor(255−A,255−B)",     _D("hamxor~HYPO"))

# ---- Krita modes from source (KoCompositeOpFunctions.h).  s=source=A, d=dest=B ----
G = lambda g: (lambda A, B: _norm(A, B, g))   # wrap a [0,1] op g(a,b) into a [0,255] mode
def _pen_b(a, b):                              # cfPenumbraB(s=a, d=b)
    cd = np.minimum(1.0, b / np.maximum(1 - a, 1e-9))
    return np.where(b >= 1, 1.0, np.where(a + b < 1, cd / 2,
           np.where(a <= 0, 0.0, 1 - (1 - b) / np.maximum(a, 1e-9) / 2)))
def _pen_d(a, b): return np.where(b >= 1, 1.0, 2 * np.arctan2(a, np.maximum(1 - b, 0)) / np.pi)
def _super(a, b):
    p = 2.875
    return np.where(a < 0.5, 1 - np.power(np.power(np.maximum(1-b,0),p) + np.power(np.maximum(1-2*a,0),p), 1/p),
                    np.power(np.power(b,p) + np.power(np.maximum(2*a-1,0),p), 1/p))
def _svg(a, b):
    D = np.where(b <= 0.25, ((16*b - 12)*b + 4)*b, np.sqrt(np.maximum(b, 0)))
    return np.where(a > 0.5, b + (2*a - 1)*(D - b), b - (1 - 2*a)*b*(1 - b))
def _modc(a, b):
    a2 = np.maximum(a, 1e-9); q = np.floor(b / a2); m = np.mod(b, a2)
    return np.where(a <= 0, 0.0, np.where(q % 2 == 0, m, a2 - m))
MODES["additivesub*NEW"] = ("A − √B",          G(lambda a, b: a - np.sqrt(b)))
MODES["arctan*NEW"]      = ("(2/π)·atan(B/A)",  G(lambda a, b: (2/np.pi) * np.arctan(b / np.maximum(a, 1e-9))))
MODES["gammadark*NEW"]   = ("B^(1/A)",          G(lambda a, b: np.power(np.maximum(b,1e-9), 1/np.maximum(a,1e-9))))
MODES["gammalight*NEW"]  = ("B^A",              G(lambda a, b: np.power(np.maximum(b,0), a)))
MODES["gammaillum*NEW"]  = ("1−(1−A)^(1/(1−B))", G(lambda a, b: 1 - np.power(np.maximum(1-a,0), 1/np.maximum(1-b,1e-9))))
MODES["pnorma*NEW"]      = ("(A^2.33+B^2.33)^.43", G(lambda a, b: np.power(np.power(a,2.3333)+np.power(b,2.3333), 1/2.3333)))
MODES["pnormb*NEW"]      = ("(A^4+B^4)^0.25",   G(lambda a, b: np.power(a**4 + b**4, 0.25)))
MODES["penumbraa*NEW"]   = ("linear penumbra",  G(lambda a, b: _pen_b(b, a)))
MODES["penumbrab*NEW"]   = ("linear penumbra↔", G(lambda a, b: _pen_b(a, b)))
MODES["penumbrac*NEW"]   = ("atan penumbra",    G(lambda a, b: _pen_d(b, a)))
MODES["penumbrad*NEW"]   = ("atan penumbra↔",   G(lambda a, b: _pen_d(a, b)))
MODES["easyburn*NEW"]    = ("1−(1−A)^(1.04B)",  G(lambda a, b: 1 - np.power(np.maximum(1-a,0), 1.04*b)))
MODES["easydodge*NEW"]   = ("dual of easy burn", _D("easyburn*NEW"))
MODES["superlight*NEW"]  = ("p=2.875 hard light", G(_super))
MODES["softpegtop*NEW"]  = ("(1−2A)B²+2AB",     G(lambda a, b: (1-2*a)*b*b + 2*a*b))
MODES["softsvg*NEW"]     = ("W3C/SVG soft light", G(_svg))
MODES["softillus*NEW"]   = ("B^(2^(1−2A))",     G(lambda a, b: np.power(np.maximum(b,1e-9), np.power(2.0, 1-2*a))))
MODES["modulo*NEW"]      = ("B mod A",          G(lambda a, b: np.where(a <= 0, 0.0, np.mod(b, np.maximum(a,1e-9)))))
MODES["modcont*NEW"]     = ("B mod A continuous", G(_modc))
MODES["divmodulo*NEW"]   = ("(B/A) mod 1",      G(lambda a, b: np.mod(b / np.maximum(a, 1e-9), 1.0)))

# generating family of each mode (keyed by display name, suffixes stripped)
FAMILY = {
    "normal": "passthrough",
    "multiply": "multiplicative", "screen": "multiplicative",
    "darken": "power-mean", "lighten": "power-mean", "average": "power-mean",
    "geometric": "power-mean", "harmonic": "power-mean", "sheen": "power-mean",
    "bloom": "power-mean", "rms": "power-mean", "contraharm": "power-mean",
    "logarithmic": "power-mean", "heronian": "power-mean", "identric": "power-mean",
    "centroidal": "power-mean",
    "einprod": "multiplicative", "einsum": "multiplicative",
    "hamprod": "multiplicative", "hamsum": "multiplicative",
    "burn": "dodge/burn", "dodge": "dodge/burn", "divide": "dodge/burn", "quench": "dodge/burn",
    "addition": "linear", "subtract": "linear", "linearburn": "linear", "lift": "linear",
    "grainextract": "linear", "grainmerge": "linear", "bleach": "linear", "stain": "linear",
    "difference": "difference", "phoenix": "difference", "negation": "difference",
    "extremity": "difference", "exclusion": "difference", "inclusion": "difference",
    "reflect": "quadratic", "glow": "quadratic", "heat": "quadratic", "freeze": "quadratic",
    "overlay": "contrast", "hardlight": "contrast", "softlight": "contrast",
    "vividlight": "contrast", "mirage": "contrast", "linearlight": "contrast",
    "pinlight": "contrast", "hardmix": "contrast", "glowlight": "contrast",
    "and": "bitwise", "or": "bitwise", "xor": "bitwise",
    "nand": "bitwise", "nor": "bitwise", "xnor": "bitwise",
    "multiply128": "multiplicative", "softdifference": "difference", "interpolate": "power-mean",
    "hardoverlay": "contrast", "hamxor": "difference", "dodge3": "dodge/burn", "burn3": "dodge/burn",
    "yagerprod": "multiplicative", "yagersum": "multiplicative", "screen128": "multiplicative",
    "embers": "difference", "rift": "difference",
    "veil": "contrast", "afterglow": "contrast",
    "additivesub": "difference", "arctan": "difference",
    "modulo": "modulo", "modcont": "modulo", "divmodulo": "modulo",
    "gammadark": "gamma", "gammalight": "gamma", "gammaillum": "gamma",
    "pnorma": "multiplicative", "pnormb": "multiplicative",
    "penumbraa": "penumbra", "penumbrab": "penumbra", "penumbrac": "penumbra", "penumbrad": "penumbra",
    "easydodge": "dodge/burn", "easyburn": "dodge/burn",
    "superlight": "contrast", "softpegtop": "contrast", "softsvg": "contrast", "softillus": "contrast",
}
def disp(name): return name.split("*")[0].split("~")[0]
def fam(name):  return FAMILY.get(disp(name), "?")

# cross-reference names: ours -> (Krita name, mathematical name).  "" = none known.
ALIASES = {
 "multiply": ("Multiply", "ab (product t-norm)"), "screen": ("Screen", "a+b−ab"),
 "darken": ("Darken", "min"), "lighten": ("Lighten", "max"),
 "burn": ("Color Burn", ""), "dodge": ("Color Dodge", ""),
 "subtract": ("Subtract", "a−b"), "addition": ("Addition", "a+b"),
 "linearburn": ("Linear Burn / Inverse Subtract", "Łukasiewicz max(0,a+b−1)"),
 "difference": ("Difference", "|a−b|"), "phoenix": ("Equivalence", "1−|a−b|"),
 "negation": ("Negation", "1−|1−a−b|"), "extremity": ("", "|1−a−b|"),
 "exclusion": ("Exclusion", "a+b−2ab"), "inclusion": ("", "1−a−b+2ab (unclaimed)"),
 "reflect": ("Reflect", "a²/(1−b)"), "glow": ("Glow", "b²/(1−a)"),
 "heat": ("Heat", ""), "freeze": ("Freeze", ""),
 "bleach": ("", "1−a−b"), "stain": ("", "2−a−b"),
 "overlay": ("Overlay", ""), "hardlight": ("Hard Light", ""), "softlight": ("Soft Light", ""),
 "vividlight": ("Vivid Light", ""), "linearlight": ("Linear Light", ""),
 "pinlight": ("Pin Light", ""), "hardmix": ("Hard Mix", ""),
 "average": ("Allanon", "arithmetic mean"), "geometric": ("Geometric Mean", "√(ab)"),
 "harmonic": ("Parallel", "harmonic 2ab/(a+b)"), "grainextract": ("Grain Extract", ""),
 "grainmerge": ("Grain Merge", ""), "divide": ("Divide", "a/b"),
 "and": ("AND", "∧"), "or": ("OR", "∨"), "xor": ("XOR", "⊕"),
 "nand": ("NAND", ""), "nor": ("NOR", ""), "xnor": ("XNOR", "≡"),
 "glowlight": ("Reflect-Freeze hybrid", "quadratic fuse"), "hardoverlay": ("Hard Overlay", ""),
 "interpolate": ("Interpolation", "cosine"), "multiply128": ("", "signed multiply @128"),
 "screen128": ("", "dual of multiply128"), "rms": ("P-Norm (≈)", "quadratic mean p=2"),
 "contraharm": ("", "contraharmonic / Lehmer 2"), "logarithmic": ("", "logarithmic mean"),
 "heronian": ("", "Heronian mean"), "identric": ("", "identric mean"),
 "centroidal": ("", "centroidal mean"), "einprod": ("", "Einstein t-norm"),
 "einsum": ("", "Einstein t-conorm"), "hamprod": ("", "Hamacher t-norm"),
 "hamsum": ("", "Hamacher t-conorm"), "yagerprod": ("P-Norm (≈)", "Yager t-norm"),
 "yagersum": ("", "Yager t-conorm"), "sheen": ("", "dual of geometric"),
 "bloom": ("", "dual of harmonic"), "lift": ("", "dual of subtract"),
 "quench": ("", "dual of divide"), "mirage": ("", "dual of vivid light"),
 "embers": ("", "dual of softdifference"), "veil": ("", "dual of hard overlay"),
 "afterglow": ("", "dual of glowlight"), "rift": ("", "dual of hamacher-xor"),
 "softdifference": ("", "contrast-stretched |a−b|"), "hamxor": ("", "a+b−2·hamacher(a,b)"),
 "dodge3": ("", "b³/(1−a)"), "burn3": ("", "1−(1−b)³/a"),
 "additivesub": ("Additive Subtractive", "a−√b"), "arctan": ("Arcus Tangent", "(2/π)atan(b/a)"),
 "modulo": ("Modulo", "b mod a"), "modcont": ("Modulo Continuous", "reflected b mod a"),
 "divmodulo": ("Divisive Modulo", "(b/a) mod 1"),
 "gammadark": ("Gamma Dark", "b^(1/a)"), "gammalight": ("Gamma Light", "b^a"),
 "gammaillum": ("Gamma Illumination", "1−(1−a)^(1/(1−b))"),
 "pnorma": ("P-Norm A", "(a^2.33+b^2.33)^.43"), "pnormb": ("P-Norm B", "(a^4+b^4)^.25"),
 "penumbraa": ("Penumbra A", "linear falloff"), "penumbrab": ("Penumbra B", "linear falloff ↔"),
 "penumbrac": ("Penumbra C", "atan falloff"), "penumbrad": ("Penumbra D", "atan falloff ↔"),
 "easydodge": ("Easy Dodge", "b^(1.04/(1−a))"), "easyburn": ("Easy Burn", "1−(1−a)^(1.04b)"),
 "superlight": ("Super Light", "p-norm hard light"), "softpegtop": ("Soft Light (Pegtop)", "(1−2a)b²+2ab"),
 "softsvg": ("Soft Light (SVG)", "W3C"), "softillus": ("Soft Light (IFS)", "b^(2^(1−2a))"),
}

# ---- exhaustive symmetry reconciliation over all 256x256 inputs --------------
gx = np.arange(256).reshape(256, 1) * np.ones((1, 256))   # A grid (top)
gy = np.ones((256, 1)) * np.arange(256).reshape(1, 256)   # B grid (bottom)
TAB = {n: np.round(clip(f(gx, gy))) for n, (_, f) in MODES.items()}

def best(t):
    bn, bd = None, 1e9
    for n, nt in TAB.items():
        d = np.max(np.abs(t - nt))
        if d < bd: bn, bd = n, d
    return bn, bd

print(f"{'mode':16}{'SWAP f(B,A) =':24}{'DUAL 255−f(255−A,255−B) =':28}")
print("-" * 70)
unnamed = []
for n in MODES:
    t = TAB[n]
    s_t = t.T                                  # SWAP
    d_t = np.round(255 - t[::-1, ::-1])        # DUAL
    sn, sd = best(s_t); dn, dd = best(d_t)
    stag = sn if sd <= 1 else f"UNNAMED(≈{sn})"
    dtag = dn if dd <= 1 else f"UNNAMED(≈{dn})"
    print(f"{n:16}{stag:24}{dtag:28}")
    if dd > 1: unnamed.append((f"DUAL({n})", d_t))

print("\nNo named DUAL match (genuine gaps in current FFmpeg set):")
shown = []
for lbl, t in unnamed:
    if any(np.max(np.abs(t - s)) == 0 for _, s in shown):
        continue
    shown.append((lbl, t)); print(" ", lbl)

# ---- big annotated chart -----------------------------------------------------
A = np.asarray(Image.open("a.jpg").convert("RGB").resize((512, 512)), float)
B = np.asarray(Image.open("b.jpg").convert("RGB").resize((512, 512)), float)
def render(f): return Image.fromarray(clip(np.round(f(A, B))).astype(np.uint8))

def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
fname, fmath = font(20), font(14)

# family sections, sorted along the monotone (dark -> light) chain. Within a family
# modes are shown in symmetric (dual) pairs, except POWER-MEAN which IS the monotone
# chain and is drawn as an ordered strip (darken -> lighten).
PM_CHAIN = ["darken", "harmonic", "geometric", "logarithmic*NEW", "heronian*NEW",
            "identric*NEW", "average", "interpolate", "rms~HYPO", "contraharm~HYPO",
            "centroidal*NEW", "sheen*NEW", "bloom*NEW", "lighten"]
SECTIONS = [
    ["POWER-MEAN", "the monotone chain darken -> lighten;  M_p = ((A^p+B^p)/2)^(1/p) and other means",
        [], PM_CHAIN],
    ["MULTIPLICATIVE", "product-based: t-norm/conorm pairs (Frank, Einstein, Hamacher, Yager), mid-pivot multiply128, P-Norm screens",
        [("multiply", "screen"), ("einprod*NEW", "einsum*NEW"), ("hamprod*NEW", "hamsum*NEW"),
         ("yagerprod*NEW", "yagersum*NEW"), ("multiply128", "screen128*NEW")],
        ["pnorma*NEW", "pnormb*NEW"]],
    ["DODGE / BURN", "B^k / (255-A) and complement;  k=1 color, k=2 quadratic, k=3 cubic, plus Krita Easy",
        [("burn", "dodge"), ("divide", "quench*NEW"), ("burn3~HYPO", "dodge3~HYPO"), ("easyburn*NEW", "easydodge*NEW")], []],
    ["GAMMA", "power-curve modes (Krita):  B^A and B^(1/A) families",
        [], ["gammadark*NEW", "gammalight*NEW", "gammaillum*NEW"]],
    ["PENUMBRA", "Krita midtone-falloff modes:  linear (A/B) and arc-tangent (C/D)",
        [], ["penumbraa*NEW", "penumbrab*NEW", "penumbrac*NEW", "penumbrad*NEW"]],
    ["LINEAR / AFFINE", "clamped  a*A + b*B + c   (strictly affine; no product term)",
        [("subtract", "lift*NEW"), ("addition", "linearburn*NEW"), ("bleach", "stain")],
        ["grainextract", "grainmerge"]],
    ["DIFFERENCE & BITWISE", "symmetric difference  A+B-2*AND(A,B):  AND=min->difference, product->exclusion, bitwise->xor   (and=conjunction, or=disjunction)",
        [("difference", "phoenix"), ("negation", "extremity"), ("exclusion", "inclusion*NEW"),
         ("xor", "xnor*NEW"), ("and", "or"), ("nand*NEW", "nor*NEW"),
         ("softdifference", "embers*NEW"), ("hamxor~HYPO", "rift~HYPO")],
        ["additivesub*NEW", "arctan*NEW"]],
    ["MODULO", "Krita wrap-around: B mod A, its reflected variant, and divisive; abstract banding",
        [], ["modulo*NEW", "modcont*NEW", "divmodulo*NEW"]],
    ["QUADRATIC", "A^2 / (255-B) and its dual / swap partners",
        [("reflect", "freeze"), ("glow", "heat")], []],
    ["CONTRAST", "fuse(darkening op below 128, lightening op above 128)",
        [("vividlight", "mirage*NEW"), ("hardoverlay", "veil*NEW"), ("glowlight~HYPO", "afterglow~HYPO")],
        ["overlay", "hardlight", "softlight", "softpegtop*NEW", "softsvg*NEW", "softillus*NEW",
         "linearlight", "pinlight", "hardmix", "superlight*NEW"]],
    ["PASSTHROUGH", "ignores B",
        [], ["normal"]],
]

# monotone sort: mean output value per mode, then order families & members dark -> light
mv = {n: float(TAB[n].mean()) for n in MODES}
for s in SECTIONS:
    s[2] = sorted(s[2], key=lambda pr: mv[pr[0]])    # pairs by their darker (left) member
    s[3] = sorted(s[3], key=lambda n: mv[n])         # singles ascending
# families are symmetric about mid-gray (each holds a dark end + its light dual), so the
# only meaningful monotone key is how dark a family can reach -> sort by darkest member.
SECTIONS.sort(key=lambda s: min(mv[m] for m in [x for pr in s[2] for x in pr] + s[3]))

# main image = photo blend A(landscape) over B(dog). To its right a 24px-wide, full-height
# strip stacks the same mode on the ramp chart (top) and the color gradient (bottom).
N2 = 256
RX = np.tile(np.arange(N2), (N2, 1)).astype(float)
RY = np.tile(np.arange(N2).reshape(-1, 1), (1, N2)).astype(float)
VY = 255.0 - RY                                                          # white/bright at top, black at bottom
_hsv = np.dstack([RX, np.full((N2, N2), 255.0), VY]).astype(np.uint8)
grad_A = np.asarray(Image.fromarray(_hsv, "HSV").convert("RGB"), float)   # hue x value (white top)
vramp = np.dstack([VY, VY, VY])                                          # vertical ramp: white top -> black bottom
_dl = np.where(((RX + RY) % 40) < 20, 235.0, 40.0)                       # thick diagonal bands
diagband = np.dstack([_dl, _dl, _dl])
PA = np.asarray(Image.open("a.jpg").convert("RGB").resize((240, 240)), float)   # landscape (top)
PB = np.asarray(Image.open("b.jpg").convert("RGB").resize((240, 240)), float)   # dog (bottom)

main, strip_w, titleh, caph = 240, 32, 28, 64
cell = main                                 # main image full width; refs live in the caption
gapx, gapy, margin, sd_gap, headh, fam_gap = 26, 18, 16, 8, 62, 26
fname, fmath, ftitle, fhint, fxref = font(18), font(11), font(22), font(15), font(9)
cols_b, sd_cols = 2, 4
block_w, block_h, sd_h = 2 * cell, titleh + main + caph, main + caph
YELLOW, LAVENDER = (255, 244, 200), (232, 224, 255)
W = margin * 2 + cols_b * block_w + (cols_b - 1) * gapx
sheet = Image.new("RGB", (W, 11000), (245, 245, 245))
d = ImageDraw.Draw(sheet)

def render_on(f, X, Y): return Image.fromarray(clip(np.round(f(X, Y))).astype(np.uint8))

def caption(x, y, name, formula):
    new, hyp = "*NEW" in name, "~HYPO" in name
    bg = YELLOW if new else LAVENDER if hyp else (255, 255, 255)
    fg = (160, 0, 0) if new else (90, 0, 150) if hyp else (0, 0, 0)
    d.rectangle([x, y, x + cell, y + caph], fill=bg)
    d.text((x + 5, y + 3), disp(name), fill=fg, font=fname)
    d.text((x + 5, y + 24), formula, fill=(70, 70, 70), font=fmath)
    k, m = ALIASES.get(disp(name), ("", ""))
    xref = "   ".join(s for s in [("K: " + k) if k else "", ("M: " + m) if m else ""] if s)
    if xref: d.text((x + 5, y + 40), xref, fill=(0, 90, 110), font=fxref)

def refs(x, y, f):                          # one 32x32 ref: hue gradient over diagonal bands
    sheet.paste(render_on(f, grad_A, diagband).resize((strip_w, strip_w)), (x, y))

def paste_single(x, y, name):
    f = MODES[name][1]
    sheet.paste(render_on(f, PA, PB).resize((main, main)), (x, y))
    caption(x, y + main, name, MODES[name][0])
    refs(x + cell - strip_w - 4, y + main + (caph - strip_w) // 2, f)

def paste_pair(bx, by, l, r):
    d.rectangle([bx - 3, by - 3, bx + block_w + 3, by + block_h + 3], outline=(175, 175, 175))
    d.text((bx + 6, by + 5), f"{disp(l)}   <->   {disp(r)}", fill=(0, 0, 110), font=fname)
    paste_single(bx, by + titleh, l)
    paste_single(bx + cell, by + titleh, r)

d.text((margin, 10), "BLEND MODE ATLAS  -  grouped by family, each mode beside its dual.  Main = A(landscape) over "
       "B(dog);  caption ref = same mode on a hue-gradient over diagonal bands.  yellow=added, lavender=hypothesis",
       fill=(0, 0, 0), font=fmath)
# inputs block: A and B, each with its strip showing the raw ramp + gradient inputs
y = 30
d.rectangle([margin - 3, y - 3, margin + block_w + 3, y + titleh + main + caph + 3], outline=(175, 175, 175))
d.text((margin + 6, y + 5), "INPUTS", fill=(0, 0, 110), font=fname)
for k, (img, lab, swatch) in enumerate([(PA, "A  landscape (top)", grad_A),
                                        (PB, "B  dog (bottom)", diagband)]):
    cx = margin + k * cell
    sheet.paste(Image.fromarray(img.astype(np.uint8)).resize((main, main)), (cx, y + titleh))
    caption(cx, y + titleh + main, lab, "ref = hue grad over diag bands")
    rx = cx + cell - strip_w - 4
    sheet.paste(Image.fromarray(swatch.astype(np.uint8)).resize((strip_w, strip_w)), (rx, y + titleh + main + (caph - strip_w) // 2))
y += titleh + main + caph + fam_gap

for famname, hint, pairs, singles in SECTIONS:
    d.rectangle([margin - 3, y, W - margin + 3, y + headh - 6], fill=(40, 55, 75))
    d.text((margin + 8, y + 8), famname, fill=(255, 255, 255), font=ftitle)
    d.text((margin + 8, y + 34), hint, fill=(205, 214, 228), font=fhint)
    y += headh
    for i, (l, r) in enumerate(pairs):
        row, col = divmod(i, cols_b)
        paste_pair(margin + col * (block_w + gapx), y + row * (block_h + gapy), l, r)
    if pairs:
        y += ((len(pairs) + cols_b - 1) // cols_b) * (block_h + gapy)
    for i, n in enumerate(singles):
        row, col = divmod(i, sd_cols)
        paste_single(margin + col * (cell + sd_gap), y + row * (sd_h + gapy), n)
    if singles:
        y += ((len(singles) + sd_cols - 1) // sd_cols) * (sd_h + gapy)
    y += fam_gap

sheet = sheet.crop((0, 0, W, y + margin))
sheet.save("blend_atlas.png")
print(f"\nWrote blend_atlas.png  (grouped by {len(SECTIONS)} families, {W}x{y + margin})")
