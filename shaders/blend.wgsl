// blend.wgsl - blend modes as WGSL (WebGPU). Portable: wgpu lowers this to
// Metal (macOS), D3D12 (Windows), Vulkan (Linux). Inputs a,b are the top/bottom
// channel in [0,1]; return is the blended channel. Benchmarked in bench/webgpu_bench.ts.
//
// The GPU story vs CPU SIMD: exp2/log2/atan/pow are hardware SFU intrinsics, so the
// transcendental modes (frank, gamma, trig) cost about the same as multiply - the CPU
// "tier" penalty disappears. Branches use select() to avoid warp divergence.

const PI: f32 = 3.14159265;

// ---- multiplicative ----
fn multiply(a: f32, b: f32) -> f32 { return a * b; }
fn screen(a: f32, b: f32) -> f32   { return a + b - a * b; }

// ---- power-mean / means ----
fn darken(a: f32, b: f32) -> f32    { return min(a, b); }
fn lighten(a: f32, b: f32) -> f32   { return max(a, b); }
fn average(a: f32, b: f32) -> f32   { return 0.5 * (a + b); }
fn geometric(a: f32, b: f32) -> f32 { return sqrt(a * b); }
fn harmonic(a: f32, b: f32) -> f32  { return 2.0 * a * b / max(a + b, 1e-6); }
fn rms(a: f32, b: f32) -> f32       { return sqrt(0.5 * (a*a + b*b)); }

// ---- dodge / burn (quadratic) ----
fn burn(a: f32, b: f32) -> f32    { return select(1.0 - (1.0 - b) / max(a, 1e-6), 0.0, a <= 0.0); }
fn dodge(a: f32, b: f32) -> f32   { return select(b / max(1.0 - a, 1e-6), 1.0, a >= 1.0); }
fn reflect(a: f32, b: f32) -> f32 { return select(min(1.0, a*a / max(1.0 - b, 1e-6)), b, b >= 1.0); }
fn glow(a: f32, b: f32) -> f32    { return select(min(1.0, b*b / max(1.0 - a, 1e-6)), a, a >= 1.0); }

// ---- difference / bitwise-as-arithmetic ----
fn difference(a: f32, b: f32) -> f32 { return abs(a - b); }
fn phoenix(a: f32, b: f32) -> f32    { return 1.0 - abs(a - b); }
fn exclusion(a: f32, b: f32) -> f32  { return a + b - 2.0 * a * b; }
fn negation(a: f32, b: f32) -> f32   { return 1.0 - abs(1.0 - a - b); }

// ---- linear / affine ----
fn addition(a: f32, b: f32) -> f32   { return min(1.0, a + b); }
fn subtract(a: f32, b: f32) -> f32   { return max(0.0, a - b); }
fn linearburn(a: f32, b: f32) -> f32 { return max(0.0, a + b - 1.0); }
fn grainmerge(a: f32, b: f32) -> f32 { return clamp(a + b - 0.5, 0.0, 1.0); }

// ---- contrast (fuse: dark op below 0.5, light op above) ----
fn overlay(a: f32, b: f32) -> f32     { return select(2.0*a*b, 1.0 - 2.0*(1.0-a)*(1.0-b), a >= 0.5); }
fn hardlight(a: f32, b: f32) -> f32   { return overlay(b, a); }              // SWAP
fn linearlight(a: f32, b: f32) -> f32 { return clamp(b + 2.0*a - 1.0, 0.0, 1.0); }
fn pinlight(a: f32, b: f32) -> f32    { return select(min(a, 2.0*b), max(a, 2.0*b - 1.0), b >= 0.5); }
fn vividlight(a: f32, b: f32) -> f32 {
    let lo = 1.0 - (1.0 - b) / max(2.0*a, 1e-6);
    let hi = b / max(2.0 - 2.0*a, 1e-6);
    return clamp(select(lo, hi, a >= 0.5), 0.0, 1.0);
}
fn softlight(a: f32, b: f32) -> f32 { return a*a + 2.0*b*a*(1.0 - a); }

// ---- transcendental: one SFU op each, no polynomial needed (unlike CPU NEON) ----
fn frank(a: f32, b: f32) -> f32 {         // Frank t-norm, s = 10 (covers a whole family)
    let S = 10.0; let L = log2(S);
    let t = (exp2(a*L) - 1.0) * (exp2(b*L) - 1.0) / (S - 1.0);
    return log2(1.0 + t) / L;
}
fn gammalight(a: f32, b: f32) -> f32 { return pow(b, a); }
fn gammadark(a: f32, b: f32) -> f32  { return pow(b, 1.0 / max(a, 1e-6)); }
fn arctan(a: f32, b: f32) -> f32     { return (2.0 / PI) * atan2(b, max(a, 1e-6)); }
fn interpolate(a: f32, b: f32) -> f32 { return 0.25 * (2.0 - cos(PI*a) - cos(PI*b)); }
fn sine(a: f32, b: f32) -> f32 { let s = sin(PI * (a + b) * 0.25); return s * s; }
