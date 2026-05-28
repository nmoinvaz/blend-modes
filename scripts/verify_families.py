#!/usr/bin/env python3
"""Verify the per-family master formulas reproduce the named modes (a,b in [0,1])."""
import numpy as np
g = np.linspace(0, 1, 129); a, b = np.meshgrid(g, g)
def md(x, y): return float(np.nanmax(np.abs(np.clip(x,0,1) - np.clip(y,0,1))))

# (C) generalized dodge/burn:  exponent k  (num = base^k, over headroom)
Dodge = lambda A, B, k: np.minimum(1, B**k / np.maximum(1 - A, 1e-9))   # k=1 dodge, k=2 glow
Burn  = lambda A, B, k: np.maximum(0, 1 - (1 - B)**k / np.maximum(A, 1e-9))  # k=1 burn, k=2 heat
print("GENERALIZED DODGE/BURN  (knob k):")
print("  Dodge_1 vs colordodge B/(1-A) :", md(Dodge(a,b,1), b/np.maximum(1-a,1e-9)))
print("  Dodge_2 vs glow  B^2/(1-A)    :", md(Dodge(a,b,2), b**2/np.maximum(1-a,1e-9)))
print("  Burn_1  vs colorburn 1-(1-B)/A:", md(Burn(a,b,1), 1-(1-b)/np.maximum(a,1e-9)))
print("  Burn_2  vs heat 1-(1-B)^2/A   :", md(Burn(a,b,2), 1-(1-b)**2/np.maximum(a,1e-9)))

# (B-XOR) difference family = fuzzy symmetric difference
# Lukasiewicz XOR  ==  |a-b|  (difference)  exactly
Tl = lambda x, y: np.maximum(0, x + y - 1)        # Lukasiewicz t-norm
Sl = lambda x, y: np.minimum(1, x + y)            # Lukasiewicz t-conorm
xor_luk = Sl(Tl(a, 1 - b), Tl(1 - a, b))
print("\nDIFFERENCE family (fuzzy symmetric difference):")
print("  Lukasiewicz XOR vs |a-b| (difference):", md(xor_luk, np.abs(a - b)))
# algebraic/product XOR vs exclusion a+b-2ab
Tp = lambda x, y: x * y
Sp = lambda x, y: x + y - x * y                   # probabilistic sum
xor_prob = Sp(Tp(a, 1 - b), Tp(1 - a, b))
print("  product XOR vs exclusion a+b-2ab     :", md(xor_prob, a + b - 2*a*b),
      "(close, not exact: conorm adds a small term)")
# the exact algebraic symmetric difference uses bounded-sum conorm:
xor_excl = Sl(Tp(a, 1 - b), Tp(1 - a, b))
print("  product-tnorm + bounded-sum  vs excl :", md(xor_excl, a + b - 2*a*b), "(exact)")
