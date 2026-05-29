// all_modes.h - genuine elementwise SIMD (NEON) for every blend mode in the atlas.
// No lookup tables, no gather: each mode is pure register arithmetic. Transcendental
// modes use NEON polynomial exp2/log2/atan/cos/sin. Bitwise use NEON logic ops.
// On x86 the same shapes map to SSE/AVX (min/max/adds, mul+/255, rcp/sqrt, poly FMAs).
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#if defined(__ARM_NEON)
#include <arm_neon.h>


// ============================================================================
//  Float driver:  load u8 x8 -> normalize -> OP per 4 lanes -> *255 clamp store
//  EXPR is evaluated with float32x4_t `a`, `b` in [0,1]; appears twice (lo/hi).
// ============================================================================

static inline float32x4_t f_of(uint16x4_t w) {
    return vmulq_f32(vcvtq_f32_u32(vmovl_u16(w)), vdupq_n_f32(1.f / 255.f));
}
static inline uint8_t clamp8(float r) {   // round(r*255) into [0,255]
    int v = (int)lrintf(r * 255.f);
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}
static inline uint16x4_t f_pack(float32x4_t r) {
    r = vmulq_f32(r, vdupq_n_f32(255.f));
    r = vminq_f32(vmaxq_f32(r, vdupq_n_f32(0.f)), vdupq_n_f32(255.f));
    return vmovn_u32(vcvtnq_u32_f32(r));
}
#define VK(NAME, EXPR)                                                          \
static void NAME(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {    \
    for (size_t i = 0; i < n; i += 8) {                                         \
        uint16x8_t wa = vmovl_u8(vld1_u8(A + i)), wb = vmovl_u8(vld1_u8(B + i)); \
        float32x4_t a = f_of(vget_low_u16(wa)),  b = f_of(vget_low_u16(wb));    \
        uint16x4_t lo = f_pack(EXPR);                                           \
        a = f_of(vget_high_u16(wa)); b = f_of(vget_high_u16(wb));               \
        uint16x4_t hi = f_pack(EXPR);                                           \
        vst1_u8(D + i, vmovn_u16(vcombine_u16(lo, hi)));                        \
    }                                                                           \
}

#define V1     vdupq_n_f32(1.f)
#define VH     vdupq_n_f32(0.5f)
#define VP     vdupq_n_f32(128.f / 255.f)   // 8-bit mid pivot (atlas uses A-128, doubled)
#define VEPS   vdupq_n_f32(1e-6f)
static inline float32x4_t vrecip(float32x4_t x) { return vdivq_f32(V1, x); }    // exact divide
static inline float32x4_t vmaxe(float32x4_t x)  { return vmaxq_f32(x, VEPS); }
static inline float32x4_t vmix(uint32x4_t m, float32x4_t t, float32x4_t f) { return vbslq_f32(m, t, f); }


// ============================================================================
//  Transcendental NEON helpers (polynomials fit offline; see scripts)
// ============================================================================

static inline float32x4_t horn(const float* C, int d, float32x4_t x) {
    float32x4_t r = vdupq_n_f32(C[0]);
    for (int k = 1; k <= d; ++k) r = vfmaq_f32(vdupq_n_f32(C[k]), x, r);
    return r;
}
static const float CE2[6] = {1.33952854e-03f, 9.67076788e-03f, 5.55034067e-02f, 2.40222117e-01f, 6.93147200e-01f, 1.00000005e+00f};
static const float CL2[7] = {-2.48259845e-02f, 2.66862768e-01f, -1.23427990e+00f, 3.21886982e+00f, -5.26415553e+00f, 6.06585886e+00f, -3.02832497e+00f};
static const float CCOS[9] = {8.40490781e-16f, 1.84470858e-04f, -2.02836302e-03f, 1.24892325e-03f, 4.02388311e-02f, 9.30169712e-04f, -5.00315165e-01f, 4.51864702e-05f, 9.99998420e-01f};
static const float CSIN[8] = {-1.37773877e-04f, -2.04628040e-04f, 8.64045889e-03f, -2.45777913e-04f, -1.66559959e-01f, -2.34169409e-05f, 1.00000214e+00f, -4.72362615e-08f};
static const float CATAN[8] = {5.29361165e-02f, -2.23770633e-01f, 3.17735837e-01f, -3.21706056e-02f, -3.29175728e-01f, -1.47010867e-04f, 9.99991151e-01f, 3.00035562e-07f};

// 2^x via k + fraction, exponent injected into the float bits
static inline float32x4_t vexp2(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-126.f)), vdupq_n_f32(126.f));  // avoid exponent overflow
    float32x4_t k = vrndnq_f32(x), f = vsubq_f32(x, k);
    float32x4_t p = horn(CE2, 5, f);
    int32x4_t ki = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    float32x4_t scale = vreinterpretq_f32_s32(vshlq_n_s32(ki, 23));
    return vmulq_f32(p, scale);
}
// log2(x) via frexp-style mantissa/exponent split; x>0 assumed (guarded by callers)
static inline float32x4_t vlog2(float32x4_t x) {
    int32x4_t xi = vreinterpretq_s32_f32(x);
    int32x4_t e  = vsubq_s32(vshrq_n_s32(vandq_s32(xi, vdupq_n_s32(0x7f800000)), 23), vdupq_n_s32(127));
    float32x4_t m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(xi, vdupq_n_s32(0x007fffff)), vdupq_n_s32(0x3f800000)));
    return vaddq_f32(vcvtq_f32_s32(e), horn(CL2, 6, m));
}
// b^e for b>=0; matches numpy edges: 0^0 = 1, 0^(>0) = 0
static inline float32x4_t vpow(float32x4_t b, float32x4_t e) {
    float32x4_t r = vexp2(vmulq_f32(e, vlog2(vmaxe(b))));
    float32x4_t z = vbslq_f32(vcleq_f32(e, vdupq_n_f32(0.f)), V1, vdupq_n_f32(0.f));  // 0^e
    return vbslq_f32(vcleq_f32(b, vdupq_n_f32(0.f)), z, r);
}
static inline float32x4_t vcos_pi(float32x4_t a) { return horn(CCOS, 8, vmulq_f32(a, vdupq_n_f32((float)M_PI))); }
static inline float32x4_t vsin(float32x4_t x)    { return horn(CSIN, 7, x); }
// (2/pi)*atan(t) in [0,1] for t>=0, with t>1 reduced via atan(t)=pi/2-atan(1/t)
static inline float32x4_t vatan_unit(float32x4_t t) {
    float32x4_t s = vdupq_n_f32(2.f / (float)M_PI);
    uint32x4_t big = vcgtq_f32(t, V1);
    float32x4_t tt = vbslq_f32(big, vrecip(vmaxe(t)), t);
    float32x4_t p  = vmulq_f32(s, horn(CATAN, 7, tt));
    return vbslq_f32(big, vsubq_f32(V1, p), p);
}


// ============================================================================
//  Shared sub-expressions (used by a mode and its dual)
// ============================================================================

static inline float32x4_t h_vivid(float32x4_t a, float32x4_t b) {            // burn(2a,b) | dodge(2a-1,b)
    float32x4_t a2 = vmulq_f32(vdupq_n_f32(2.f), a);
    float32x4_t lo = vbslq_f32(vcleq_f32(a2, vdupq_n_f32(0.f)), vdupq_n_f32(0.f),       // burn edge: 2a<=0 -> 0
                               vsubq_f32(V1, vdivq_f32(vsubq_f32(V1, b), vmaxe(a2))));
    float32x4_t c  = vmulq_f32(vdupq_n_f32(2.f), vsubq_f32(a, VP));                       // 2(a-128/255)
    float32x4_t hi = vbslq_f32(vcgeq_f32(c, V1), V1, vdivq_f32(b, vmaxe(vsubq_f32(V1, c)))); // dodge edge: c>=1 -> 1
    return vmix(vcltq_f32(a, VH), lo, hi);   // branch at 127.5/255 = exact integer A<128 split
}
static inline float32x4_t h_glow(float32x4_t a, float32x4_t b) {             // heat(2a,b) | glow(2a-1,b)
    float32x4_t a2 = vmulq_f32(vdupq_n_f32(2.f), a);
    float32x4_t lo = vbslq_f32(vcleq_f32(a2, vdupq_n_f32(0.f)), vdupq_n_f32(0.f),       // _bq edge: c<=0 -> 0
                               vsubq_f32(V1, vminq_f32(V1, vdivq_f32(vmulq_f32(vsubq_f32(V1,b), vsubq_f32(V1,b)), vmaxe(a2)))));
    float32x4_t c  = vmulq_f32(vdupq_n_f32(2.f), vsubq_f32(a, VP));                       // 2(a-128/255)
    float32x4_t hi = vbslq_f32(vcgeq_f32(c, V1), V1,                                     // _dq edge: c>=1 -> 1
                               vminq_f32(V1, vdivq_f32(vmulq_f32(b,b), vmaxe(vsubq_f32(V1, c)))));
    return vmix(vcltq_f32(a, VH), lo, hi);   // branch at 127.5/255 = exact integer A<128 split
}
static inline float32x4_t h_hamprod(float32x4_t a, float32x4_t b) {          // ab/(a+b-ab)
    float32x4_t d = vsubq_f32(vaddq_f32(a, b), vmulq_f32(a, b));
    return vbslq_f32(vcleq_f32(d, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), vdivq_f32(vmulq_f32(a, b), vmaxe(d)));
}
static inline float32x4_t h_hamxor(float32x4_t a, float32x4_t b) {           // a+b-2*hamprod
    return vsubq_f32(vaddq_f32(a, b), vmulq_f32(vdupq_n_f32(2.f), h_hamprod(a, b)));
}
static inline float32x4_t h_softdiff(float32x4_t a, float32x4_t b) {         // contrast-stretched |a-b|
    float32x4_t gt = vbslq_f32(vcgeq_f32(b, V1), vdupq_n_f32(0.f), vdivq_f32(vsubq_f32(a, b), vmaxe(vsubq_f32(V1, b))));
    float32x4_t le = vbslq_f32(vcleq_f32(b, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), vdivq_f32(vsubq_f32(b, a), vmaxe(b)));
    return vmix(vcgtq_f32(a, b), gt, le);
}
static inline float32x4_t h_hardov(float32x4_t a, float32x4_t b) {           // a>.5: b/(2(1-a)) | 2ab
    float32x4_t hi = vdivq_f32(b, vmaxe(vmulq_f32(vdupq_n_f32(2.f), vsubq_f32(V1, a))));
    float32x4_t lo = vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a, b));
    return vbslq_f32(vcgeq_f32(a, V1), V1, vminq_f32(V1, vmix(vcgtq_f32(a, VH), hi, lo)));
}
static inline float32x4_t h_mul128(float32x4_t a, float32x4_t b) {           // ((a*255-128)*b*255/32+128)/255
    float32x4_t A8 = vmulq_f32(a, vdupq_n_f32(255.f)), B8 = vmulq_f32(b, vdupq_n_f32(255.f));
    float32x4_t r  = vaddq_f32(vmulq_f32(vsubq_f32(A8, vdupq_n_f32(128.f)), vmulq_f32(B8, vdupq_n_f32(1.f/32.f))), vdupq_n_f32(128.f));
    return vmulq_f32(r, vdupq_n_f32(1.f/255.f));
}
static inline float32x4_t h_penb(float32x4_t p, float32x4_t q) {             // Krita penumbra B(p,q)
    float32x4_t cd = vminq_f32(V1, vdivq_f32(q, vmaxe(vsubq_f32(V1, p))));
    float32x4_t inner = vbslq_f32(vcleq_f32(p, vdupq_n_f32(0.f)), vdupq_n_f32(0.f),
                                  vsubq_f32(V1, vmulq_f32(vdivq_f32(vsubq_f32(V1, q), vmaxe(p)), VH)));
    float32x4_t r = vmix(vcltq_f32(vaddq_f32(p, q), V1), vmulq_f32(cd, VH), inner);
    return vbslq_f32(vcgeq_f32(q, V1), V1, r);
}
static inline float32x4_t h_glowlight(float32x4_t a, float32x4_t b) { return h_glow(a, b); }


// ============================================================================
//  Float-elementwise modes (one VK line each)
// ============================================================================

VK(normal_k,       a)
VK(multiply_k,     vmulq_f32(a, b))
VK(screen_k,       vsubq_f32(vaddq_f32(a, b), vmulq_f32(a, b)))
VK(darken_k,       vminq_f32(a, b))
VK(lighten_k,      vmaxq_f32(a, b))
VK(burn_k,         vbslq_f32(vcleq_f32(a, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), vsubq_f32(V1, vdivq_f32(vsubq_f32(V1, b), vmaxe(a)))))
VK(dodge_k,        vbslq_f32(vcgeq_f32(a, V1), V1, vdivq_f32(b, vmaxe(vsubq_f32(V1, a)))))
VK(subtract_k,     vsubq_f32(a, b))
VK(addition_k,     vaddq_f32(a, b))
VK(difference_k,   vabdq_f32(a, b))
VK(phoenix_k,      vsubq_f32(V1, vabdq_f32(a, b)))
VK(negation_k,     vsubq_f32(V1, vabdq_f32(V1, vaddq_f32(a, b))))
VK(extremity_k,    vabdq_f32(V1, vaddq_f32(a, b)))
VK(exclusion_k,    vsubq_f32(vaddq_f32(a, b), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a, b))))
VK(inclusion_k,    vsubq_f32(V1, vsubq_f32(vaddq_f32(a, b), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a, b)))))
VK(reflect_k,      vbslq_f32(vcgeq_f32(b, V1), b, vminq_f32(V1, vdivq_f32(vmulq_f32(a, a), vmaxe(vsubq_f32(V1, b))))))
VK(glow_k,         vbslq_f32(vcgeq_f32(a, V1), a, vminq_f32(V1, vdivq_f32(vmulq_f32(b, b), vmaxe(vsubq_f32(V1, a))))))
VK(heat_k,         vbslq_f32(vcleq_f32(a, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), vsubq_f32(V1, vminq_f32(V1, vdivq_f32(vmulq_f32(vsubq_f32(V1,b),vsubq_f32(V1,b)), vmaxe(a))))))
VK(freeze_k,       vbslq_f32(vcleq_f32(b, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), vsubq_f32(V1, vminq_f32(V1, vdivq_f32(vmulq_f32(vsubq_f32(V1,a),vsubq_f32(V1,a)), vmaxe(b))))))
VK(bleach_k,       vsubq_f32(V1, vaddq_f32(a, b)))
VK(stain_k,        vsubq_f32(vdupq_n_f32(2.f), vaddq_f32(a, b)))
VK(overlay_k,      vmix(vcltq_f32(a, VH), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a,b)), vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(vsubq_f32(V1,a), vsubq_f32(V1,b))))))
VK(hardlight_k,    vmix(vcltq_f32(b, VH), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a,b)), vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(vsubq_f32(V1,a), vsubq_f32(V1,b))))))
VK(softlight_k,    vaddq_f32(vmulq_f32(a, a), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(b, vmulq_f32(a, vsubq_f32(V1, a))))))
VK(vividlight_k,   h_vivid(a, b))
VK(linearlight_k,  vsubq_f32(vaddq_f32(b, vmulq_f32(vdupq_n_f32(2.f), a)), V1))
VK(pinlight_k,     vmix(vcltq_f32(b, VH), vminq_f32(a, vmulq_f32(vdupq_n_f32(2.f), b)), vmaxq_f32(a, vmulq_f32(vdupq_n_f32(2.f), vsubq_f32(b, VP)))))
VK(hardmix_k,      vbslq_f32(vcgeq_f32(a, vsubq_f32(V1, b)), V1, vdupq_n_f32(0.f)))
VK(average_k,      vmulq_f32(vaddq_f32(a, b), VH))
VK(geometric_k,    vsqrtq_f32(vmulq_f32(a, b)))
VK(harmonic_k,     vdivq_f32(vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a,b)), vmaxe(vaddq_f32(a,b))))
VK(grainextract_k, vaddq_f32(VH, vsubq_f32(a, b)))
VK(grainmerge_k,   vsubq_f32(vaddq_f32(a, b), VH))
VK(divide_k,       vbslq_f32(vcleq_f32(b, vdupq_n_f32(0.f)), V1, vminq_f32(V1, vdivq_f32(a, vmaxe(b)))))
VK(linearburn_k,   vsubq_f32(vaddq_f32(a, b), V1))
VK(lift_k,         vsubq_f32(V1, vmaxq_f32(vdupq_n_f32(0.f), vsubq_f32(b, a))))
VK(mirage_k,       vsubq_f32(V1, h_vivid(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(sheen_k,        vsubq_f32(V1, vsqrtq_f32(vmulq_f32(vsubq_f32(V1,a), vsubq_f32(V1,b)))))
VK(bloom_k,        vsubq_f32(V1, vdivq_f32(vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(vsubq_f32(V1,a),vsubq_f32(V1,b))), vmaxe(vsubq_f32(vdupq_n_f32(2.f), vaddq_f32(a,b))))))
VK(quench_k,       vbslq_f32(vcgeq_f32(b, V1), vdupq_n_f32(0.f), vsubq_f32(V1, vminq_f32(V1, vdivq_f32(vsubq_f32(V1,a), vmaxe(vsubq_f32(V1,b)))))))
VK(rms_k,          vsqrtq_f32(vmulq_f32(vaddq_f32(vmulq_f32(a,a), vmulq_f32(b,b)), VH)))
VK(contraharm_k,   vdivq_f32(vaddq_f32(vmulq_f32(a,a), vmulq_f32(b,b)), vmaxe(vaddq_f32(a,b))))
VK(glowlight_k,    h_glow(a, b))
VK(einprod_k,      vdivq_f32(vmulq_f32(a,b), vaddq_f32(V1, vmulq_f32(vsubq_f32(V1,a), vsubq_f32(V1,b)))))
VK(einsum_k,       vdivq_f32(vaddq_f32(a,b), vaddq_f32(V1, vmulq_f32(a,b))))
VK(hamprod_k,      h_hamprod(a, b))
VK(hamsum_k,       vsubq_f32(V1, h_hamprod(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(multiply128_k,  h_mul128(a, b))
VK(screen128_k,    vsubq_f32(V1, h_mul128(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(softdifference_k, h_softdiff(a, b))
VK(embers_k,       vsubq_f32(V1, h_softdiff(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(interpolate_k,  vmulq_f32(vdupq_n_f32(0.25f), vsubq_f32(vsubq_f32(vdupq_n_f32(2.f), vcos_pi(a)), vcos_pi(b))))
VK(hardoverlay_k,  h_hardov(a, b))
VK(veil_k,         vsubq_f32(V1, h_hardov(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(hamxor_k,       h_hamxor(a, b))
VK(rift_k,         vsubq_f32(V1, h_hamxor(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(dodge3_k,       vbslq_f32(vcgeq_f32(a, V1), vbslq_f32(vcgtq_f32(b, vdupq_n_f32(0.f)), V1, vdupq_n_f32(0.f)), vminq_f32(V1, vdivq_f32(vmulq_f32(b, vmulq_f32(b,b)), vmaxe(vsubq_f32(V1,a))))))
VK(burn3_k,        vbslq_f32(vcleq_f32(a, vdupq_n_f32(0.f)), vbslq_f32(vcgeq_f32(b, V1), V1, vdupq_n_f32(0.f)), vmaxq_f32(vdupq_n_f32(0.f), vsubq_f32(V1, vdivq_f32(vmulq_f32(vsubq_f32(V1,b), vmulq_f32(vsubq_f32(V1,b),vsubq_f32(V1,b))), vmaxe(a))))))
VK(yagerprod_k,    vsubq_f32(V1, vsqrtq_f32(vaddq_f32(vmulq_f32(vsubq_f32(V1,a),vsubq_f32(V1,a)), vmulq_f32(vsubq_f32(V1,b),vsubq_f32(V1,b))))))
VK(yagersum_k,     vminq_f32(V1, vsqrtq_f32(vaddq_f32(vmulq_f32(a,a), vmulq_f32(b,b)))))
VK(additivesub_k,  vsubq_f32(a, vsqrtq_f32(vmaxq_f32(b, vdupq_n_f32(0.f)))))
VK(afterglow_k,    vsubq_f32(V1, h_glow(vsubq_f32(V1,a), vsubq_f32(V1,b))))
VK(softpegtop_k,   vaddq_f32(vmulq_f32(vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f),a)), vmulq_f32(b,b)), vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a,b))))

// trig / transcendental
VK(arctan_k,       vatan_unit(vdivq_f32(b, vmaxe(a))))
VK(splay_k,        vsubq_f32(V1, vatan_unit(vdivq_f32(vsubq_f32(V1,b), vmaxe(vsubq_f32(V1,a))))))
VK(sine_k,         (vmulq_f32(vsin(vmulq_f32(vdupq_n_f32((float)M_PI*0.25f), vaddq_f32(a,b))), vsin(vmulq_f32(vdupq_n_f32((float)M_PI*0.25f), vaddq_f32(a,b))))))
VK(gammadark_k,    vpow(b, vrecip(vmaxe(a))))
VK(gammalight_k,   vpow(b, a))
VK(gammaillum_k,   vsubq_f32(V1, vpow(vsubq_f32(V1,a), vrecip(vmaxe(vsubq_f32(V1,b))))))
VK(pnorma_k,       vpow(vaddq_f32(vpow(a, vdupq_n_f32(2.3333f)), vpow(b, vdupq_n_f32(2.3333f))), vdupq_n_f32(1.f/2.3333f)))
VK(pnormb_k,       vsqrtq_f32(vsqrtq_f32(vaddq_f32(vmulq_f32(vmulq_f32(a,a),vmulq_f32(a,a)), vmulq_f32(vmulq_f32(b,b),vmulq_f32(b,b))))))
VK(easyburn_k,     vsubq_f32(V1, vpow(vsubq_f32(V1,a), vmulq_f32(vdupq_n_f32(1.04f), b))))
VK(easydodge_k,    vpow(a, vmulq_f32(vdupq_n_f32(1.04f), vsubq_f32(V1, b))))
VK(softillus_k,    vpow(b, vexp2(vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f), a)))))
VK(penumbraa_k,    h_penb(b, a))
VK(penumbrab_k,    h_penb(a, b))
VK(penumbrac_k,    vbslq_f32(vcgeq_f32(a, V1), V1, vatan_unit(vdivq_f32(b, vmaxe(vsubq_f32(V1, a))))))
VK(penumbrad_k,    vbslq_f32(vcgeq_f32(b, V1), V1, vatan_unit(vdivq_f32(a, vmaxe(vsubq_f32(V1, b))))))
// (b/a) mod 1 - also discontinuous; scalar double to match the float64 atlas
static void divmodulo_k(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double a = A[i] / 255.0, b = B[i] / 255.0;
        double t = b / (a > 1e-9 ? a : 1e-9);
        D[i] = clamp8((float)(t - std::floor(t)));
    }
}

// logarithmic / identric mean: ln via vlog2*ln2; guard |a-b|<eps -> a
static inline float32x4_t op_log(float32x4_t a, float32x4_t b) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718f), e9 = vdupq_n_f32(1e-9f);
    float32x4_t num = vsubq_f32(a, b);
    float32x4_t den = vmulq_f32(ln2, vsubq_f32(vlog2(vmaxq_f32(a, e9)), vlog2(vmaxq_f32(b, e9))));
    float32x4_t r = vdivq_f32(num, den);  // den != 0 when a != b
    return vbslq_f32(vcltq_f32(vabdq_f32(a, b), VEPS), a, r);
}
static inline float32x4_t op_identric(float32x4_t a, float32x4_t b) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718f);
    float32x4_t la = vmulq_f32(ln2, vlog2(vmaxe(a))), lb = vmulq_f32(ln2, vlog2(vmaxe(b)));
    float32x4_t ex = vsubq_f32(vdivq_f32(vsubq_f32(vmulq_f32(a, la), vmulq_f32(b, lb)), vsubq_f32(a, b)), V1);
    float32x4_t r = vexp2(vmulq_f32(ex, vdupq_n_f32(1.4426950f)));  // exp(x)=exp2(x/ln2)
    return vbslq_f32(vcltq_f32(vabdq_f32(a, b), VEPS), a, r);
}
static inline float32x4_t op_heronian(float32x4_t a, float32x4_t b) {
    return vmulq_f32(vdupq_n_f32(1.f/3.f), vaddq_f32(vaddq_f32(a, b), vsqrtq_f32(vmulq_f32(a, b))));
}
static inline float32x4_t op_centroidal(float32x4_t a, float32x4_t b) {
    float32x4_t num = vaddq_f32(vaddq_f32(vmulq_f32(a,a), vmulq_f32(a,b)), vmulq_f32(b,b));
    return vdivq_f32(vmulq_f32(vdupq_n_f32(2.f), num), vmaxe(vmulq_f32(vdupq_n_f32(3.f), vaddq_f32(a,b))));
}
static inline float32x4_t op_super(float32x4_t a, float32x4_t b) {           // superlight p=2.875
    const float32x4_t p = vdupq_n_f32(2.875f), ip = vdupq_n_f32(1.f/2.875f);
    float32x4_t lo = vsubq_f32(V1, vpow(vaddq_f32(vpow(vsubq_f32(V1,b),p), vpow(vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f),a)),p)), ip));
    float32x4_t hi = vpow(vaddq_f32(vpow(b,p), vpow(vsubq_f32(vmulq_f32(vdupq_n_f32(2.f),a),V1),p)), ip);
    return vmix(vcltq_f32(a, VH), lo, hi);
}
static inline float32x4_t op_softsvg(float32x4_t a, float32x4_t b) {         // W3C/SVG soft light
    float32x4_t poly = vmulq_f32(vaddq_f32(vmulq_f32(vsubq_f32(vmulq_f32(vdupq_n_f32(16.f),b), vdupq_n_f32(12.f)), b), vdupq_n_f32(4.f)), b);
    float32x4_t D = vbslq_f32(vcleq_f32(b, vdupq_n_f32(0.25f)), poly, vsqrtq_f32(vmaxq_f32(b, vdupq_n_f32(0.f))));
    float32x4_t hi = vaddq_f32(b, vmulq_f32(vsubq_f32(vmulq_f32(vdupq_n_f32(2.f),a), V1), vsubq_f32(D, b)));
    float32x4_t lo = vsubq_f32(b, vmulq_f32(vsubq_f32(V1, vmulq_f32(vdupq_n_f32(2.f),a)), vmulq_f32(b, vsubq_f32(V1,b))));
    return vmix(vcgtq_f32(a, VH), hi, lo);
}
VK(logarithmic_k, op_log(a, b))
VK(identric_k,    op_identric(a, b))
VK(heronian_k,    op_heronian(a, b))
VK(centroidal_k,  op_centroidal(a, b))
VK(superlight_k,  op_super(a, b))
VK(softsvg_k,     op_softsvg(a, b))


// ============================================================================
//  Integer-only modes: bitwise (NEON logic), modulo (no SIMD mod -> scalar)
// ============================================================================

#define BITK(NAME, VEXPR) \
static void NAME(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {     \
    for (size_t i = 0; i < n; i += 16) {                                         \
        uint8x16_t va = vld1q_u8(A + i), vb = vld1q_u8(B + i);                    \
        vst1q_u8(D + i, (VEXPR));                                                 \
    }                                                                            \
}
BITK(and_k,  vandq_u8(va, vb))
BITK(or_k,   vorrq_u8(va, vb))
BITK(xor_k,  veorq_u8(va, vb))
BITK(nand_k, vmvnq_u8(vandq_u8(va, vb)))
BITK(nor_k,  vmvnq_u8(vorrq_u8(va, vb)))
BITK(xnor_k, vmvnq_u8(veorq_u8(va, vb)))

// Modulo is discontinuous: at exact integer ratios the result jumps a full period, so
// float32 rounding disagrees with the float64 atlas. There is no SIMD modulo instruction
// on any ISA, so these run as a scalar double loop (still table-free) to match exactly.
// numpy np.mod: r in [0,a); at integer ratios fmod rounding leaves a tiny negative
// that numpy folds back by adding the divisor (a full-period jump - inherent to modulo).
static inline double npmod(double b, double a) {
    double m = b - std::floor(b / a) * a;
    if (m < 0.0) m += a;
    if (m >= a)  m -= a;
    return m;
}
static void modulo_k(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double a = A[i] / 255.0, b = B[i] / 255.0;
        D[i] = clamp8((float)(a <= 0.0 ? 0.0 : npmod(b, a)));
    }
}
static void modcont_k(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double a = A[i] / 255.0, b = B[i] / 255.0;
        double r = 0.0;
        if (a > 0.0) { double q = std::floor(b / a), m = npmod(b, a); r = ((long long)q & 1) ? a - m : m; }
        D[i] = clamp8((float)r);
    }
}

#else  // ---- no NEON: scalar fallbacks so the registry still links ----
#include "blend_simd.h"
#endif


// ============================================================================
//  Registry: name -> SIMD kernel, for every atlas mode
// ============================================================================

typedef void (*BlendKernel)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
struct ModeEntry { const char* name; BlendKernel simd; };

#if defined(__ARM_NEON)
static const ModeEntry MODE_TABLE[] = {
    {"normal", normal_k}, {"multiply", multiply_k}, {"screen", screen_k},
    {"darken", darken_k}, {"lighten", lighten_k}, {"burn", burn_k}, {"dodge", dodge_k},
    {"subtract", subtract_k}, {"addition", addition_k}, {"difference", difference_k},
    {"phoenix", phoenix_k}, {"negation", negation_k}, {"extremity", extremity_k},
    {"exclusion", exclusion_k}, {"inclusion", inclusion_k}, {"reflect", reflect_k},
    {"glow", glow_k}, {"heat", heat_k}, {"freeze", freeze_k}, {"bleach", bleach_k},
    {"stain", stain_k}, {"overlay", overlay_k}, {"hardlight", hardlight_k},
    {"softlight", softlight_k}, {"vividlight", vividlight_k}, {"linearlight", linearlight_k},
    {"pinlight", pinlight_k}, {"hardmix", hardmix_k}, {"average", average_k},
    {"geometric", geometric_k}, {"harmonic", harmonic_k}, {"grainextract", grainextract_k},
    {"grainmerge", grainmerge_k}, {"divide", divide_k}, {"linearburn", linearburn_k},
    {"lift", lift_k}, {"mirage", mirage_k}, {"sheen", sheen_k}, {"bloom", bloom_k},
    {"quench", quench_k}, {"rms", rms_k}, {"contraharm", contraharm_k}, {"glowlight", glowlight_k},
    {"and", and_k}, {"or", or_k}, {"xor", xor_k}, {"nand", nand_k}, {"nor", nor_k}, {"xnor", xnor_k},
    {"logarithmic", logarithmic_k}, {"heronian", heronian_k}, {"identric", identric_k},
    {"centroidal", centroidal_k}, {"einprod", einprod_k}, {"einsum", einsum_k},
    {"hamprod", hamprod_k}, {"hamsum", hamsum_k}, {"multiply128", multiply128_k},
    {"softdifference", softdifference_k}, {"interpolate", interpolate_k}, {"hardoverlay", hardoverlay_k},
    {"hamxor", hamxor_k}, {"dodge3", dodge3_k}, {"burn3", burn3_k}, {"yagerprod", yagerprod_k},
    {"yagersum", yagersum_k}, {"screen128", screen128_k}, {"embers", embers_k}, {"veil", veil_k},
    {"afterglow", afterglow_k}, {"rift", rift_k}, {"additivesub", additivesub_k}, {"arctan", arctan_k},
    {"splay", splay_k}, {"sine", sine_k}, {"gammadark", gammadark_k}, {"gammalight", gammalight_k},
    {"gammaillum", gammaillum_k}, {"pnorma", pnorma_k}, {"pnormb", pnormb_k},
    {"penumbraa", penumbraa_k}, {"penumbrab", penumbrab_k}, {"penumbrac", penumbrac_k},
    {"penumbrad", penumbrad_k}, {"easyburn", easyburn_k}, {"easydodge", easydodge_k},
    {"superlight", superlight_k}, {"softpegtop", softpegtop_k}, {"softsvg", softsvg_k},
    {"softillus", softillus_k}, {"modulo", modulo_k}, {"modcont", modcont_k}, {"divmodulo", divmodulo_k},
};
static const int MODE_COUNT = (int)(sizeof(MODE_TABLE) / sizeof(MODE_TABLE[0]));
#else
static const ModeEntry MODE_TABLE[] = {};
static const int MODE_COUNT = 0;
#endif
