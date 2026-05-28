// Scalar vs NEON SIMD blend modes (Tier-1 "free" ops + Tier-2 multiply/screen).
// On x86 the same ops map to _mm_min_epu8 / _mm_adds_epu8 / _mm_avg_epu8 / etc.
#pragma once
#include <cstdint>
#include <cstddef>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// round(x / 255) for x in [0, 65025], integer-only.
static inline uint32_t div255(uint32_t x) { x += 0x80; return (x + (x >> 8)) >> 8; }

#define SCALAR_BODY(EXPR)                                                   \
    for (size_t i = 0; i < n; ++i) { unsigned a = A[i], b = B[i]; D[i] = (uint8_t)(EXPR); }

static inline void darken_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)  { SCALAR_BODY(a < b ? a : b) }
static inline void lighten_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { SCALAR_BODY(a > b ? a : b) }
static inline void add_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a + b > 255 ? 255 : a + b) }
static inline void sub_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a > b ? a - b : 0) }
static inline void avg_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY((a + b + 1) >> 1) }
static inline void xor_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a ^ b) }
static inline void diff_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)    { SCALAR_BODY(a > b ? a - b : b - a) }
static inline void mul_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(div255(a * b)) }
static inline void screen_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)  { SCALAR_BODY(255 - div255((255 - a) * (255 - b))) }

#if defined(__ARM_NEON)
static inline uint16x8_t div255_u16(uint16x8_t x) {
    x = vaddq_u16(x, vdupq_n_u16(0x80));
    return vshrq_n_u16(vaddq_u16(x, vshrq_n_u16(x, 8)), 8);
}
// 16 bytes per NEON op; scalar handles the < 16 tail.
#define SIMD_FN(NAME, VINIT, VEXPR, SCALARFN)                                            \
static inline void NAME(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {      \
    size_t i = 0, m = n & ~size_t(15);                                                   \
    for (; i < m; i += 16) {                                                             \
        uint8x16_t va = vld1q_u8(A + i), vb = vld1q_u8(B + i); (void)va; (void)vb;        \
        VINIT vst1q_u8(D + i, (VEXPR));                                                   \
    }                                                                                    \
    SCALARFN(A + i, B + i, D + i, n - i);                                                \
}
SIMD_FN(darken_simd, , vminq_u8(va, vb), darken_scalar)
SIMD_FN(lighten_simd, , vmaxq_u8(va, vb), lighten_scalar)
SIMD_FN(add_simd, , vqaddq_u8(va, vb), add_scalar)              // saturating add  (free clamp)
SIMD_FN(sub_simd, , vqsubq_u8(va, vb), sub_scalar)              // saturating sub  (free clamp)
SIMD_FN(avg_simd, , vrhaddq_u8(va, vb), avg_scalar)            // rounding halving add  (1 instr)
SIMD_FN(xor_simd, , veorq_u8(va, vb), xor_scalar)
SIMD_FN(diff_simd, , vabdq_u8(va, vb), diff_scalar)            // absolute difference  (1 instr)

static inline void mul_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(15);
    for (; i < m; i += 16) {
        uint8x16_t va = vld1q_u8(A + i), vb = vld1q_u8(B + i);
        uint16x8_t lo = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
        uint16x8_t hi = vmull_u8(vget_high_u8(va), vget_high_u8(vb));
        vst1q_u8(D + i, vcombine_u8(vmovn_u16(div255_u16(lo)), vmovn_u16(div255_u16(hi))));
    }
    mul_scalar(A + i, B + i, D + i, n - i);
}
static inline void screen_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(15);
    for (; i < m; i += 16) {
        uint8x16_t na = vmvnq_u8(vld1q_u8(A + i)), nb = vmvnq_u8(vld1q_u8(B + i)); // 255-a, 255-b
        uint16x8_t lo = vmull_u8(vget_low_u8(na), vget_low_u8(nb));
        uint16x8_t hi = vmull_u8(vget_high_u8(na), vget_high_u8(nb));
        uint8x16_t p = vcombine_u8(vmovn_u16(div255_u16(lo)), vmovn_u16(div255_u16(hi)));
        vst1q_u8(D + i, vmvnq_u8(p));                                              // 255 - p
    }
    screen_scalar(A + i, B + i, D + i, n - i);
}
#else  // no NEON: SIMD == scalar fallback
#define ALIAS(S, P) static inline void S(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { P(A, B, D, n); }
ALIAS(darken_simd, darken_scalar) ALIAS(lighten_simd, lighten_scalar) ALIAS(add_simd, add_scalar)
ALIAS(sub_simd, sub_scalar) ALIAS(avg_simd, avg_scalar) ALIAS(xor_simd, xor_scalar)
ALIAS(diff_simd, diff_scalar) ALIAS(mul_simd, mul_scalar) ALIAS(screen_simd, screen_scalar)
#endif

// ---- Frank t-norm system (Tier-4: pow + log).  s is constant per image. ----
#include <cmath>
static inline float frank_t(float a, float b, float s) {     // a,b in [0,1]
    if (s > 0.999f && s < 1.001f) return a * b;              // s=1 -> product (multiply)
    return logf(1.f + (powf(s, a) - 1.f) * (powf(s, b) - 1.f) / (s - 1.f)) / logf(s);
}
static const float FRANK_S = 10.f;                           // an intermediate Frank t-norm
static uint8_t FRANK_LUT[256 * 256];
static inline uint8_t clamp8(float r) { int v = (int)lrintf(r * 255.f); return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
static inline void frank_build_lut(float s) {                // run ONCE per (mode,s)
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b)
            FRANK_LUT[(a << 8) | b] = clamp8(frank_t(a / 255.f, b / 255.f, s));
}
// per-pixel transcendental: pow,pow,log,log  (the slow path)
static inline void frank_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i) D[i] = clamp8(frank_t(A[i] / 255.f, B[i] / 255.f, FRANK_S));
}
// LUT lookup: one gather per pixel; same kernel for ANY Frank-derived mode (just rebuild table)
static inline void frank_lut(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i) D[i] = FRANK_LUT[(A[i] << 8) | B[i]];
}

// ---- Trigonometric modes (also Tier-4: cos / sin / atan) -> share the LUT path ----
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef float (*Gfn)(float, float);
static inline float g_interp(float a, float b) { return 0.25f * (2.f - cosf((float)M_PI * a) - cosf((float)M_PI * b)); }
static inline float g_sine(float a, float b)   { float t = sinf((float)M_PI * (a + b) * 0.25f); return t * t; }
static inline float g_arctan(float a, float b) { return (2.f / (float)M_PI) * atanf(b / (a > 1e-6f ? a : 1e-6f)); }
static inline void interp_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { for (size_t i = 0; i < n; ++i) D[i] = clamp8(g_interp(A[i] / 255.f, B[i] / 255.f)); }
static inline void sine_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)   { for (size_t i = 0; i < n; ++i) D[i] = clamp8(g_sine(A[i] / 255.f, B[i] / 255.f)); }
static inline void arctan_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { for (size_t i = 0; i < n; ++i) D[i] = clamp8(g_arctan(A[i] / 255.f, B[i] / 255.f)); }
static inline void build_lut_g(Gfn g) { for (int a = 0; a < 256; ++a) for (int b = 0; b < 256; ++b) FRANK_LUT[(a << 8) | b] = clamp8(g(a / 255.f, b / 255.f)); }
// frank_lut() above is the shared LUT-apply kernel; timing is identical for any tabulated mode.

// ---- gather-free Frank: polynomial sa=10^a and log10 over their bounded ranges ----
// max error vs exact Frank(s=10) = 0.65 of an 8-bit level.  Pure FMA, no table/gather.
#if defined(__ARM_NEON)
static const float CA[7] = {6.84077096e-01f,-2.57628734e-01f,1.77833343e+00f,1.80124913e+00f,2.69457216e+00f,2.29926614e+00f,1.00006040e+00f};
static const float CL[8] = {1.91446413e-06f,-8.34187231e-05f,1.52474870e-03f,-1.52358431e-02f,9.12609306e-02f,-3.42288770e-01f,8.73234868e-01f,-6.05894708e-01f};
static inline float32x4_t horner(const float* C, int deg, float32x4_t x) {
    float32x4_t a = vdupq_n_f32(C[0]);
    for (int k = 1; k <= deg; ++k) a = vfmaq_f32(vdupq_n_f32(C[k]), x, a);  // a = C[k] + x*a
    return a;
}
static inline uint16x4_t frank4(uint16x4_t ai, uint16x4_t bi) {
    const float32x4_t inv = vdupq_n_f32(1.f / 255.f), one = vdupq_n_f32(1.f);
    float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u16(ai)), inv);
    float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u16(bi)), inv);
    float32x4_t prod = vmulq_f32(vmulq_f32(vsubq_f32(horner(CA, 6, a), one),
                                            vsubq_f32(horner(CA, 6, b), one)),
                                  vdupq_n_f32(1.f / 9.f));               // (10^a-1)(10^b-1)/(s-1)
    float32x4_t r = vmulq_f32(horner(CL, 7, vaddq_f32(one, prod)), vdupq_n_f32(255.f));
    r = vminq_f32(vmaxq_f32(r, vdupq_n_f32(0.f)), vdupq_n_f32(255.f));
    return vmovn_u32(vcvtnq_u32_f32(r));                                 // round to nearest
}
static inline void frank_poly(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(7);
    for (; i < m; i += 8) {
        uint16x8_t wa = vmovl_u8(vld1_u8(A + i)), wb = vmovl_u8(vld1_u8(B + i));
        uint16x4_t lo = frank4(vget_low_u16(wa), vget_low_u16(wb));
        uint16x4_t hi = frank4(vget_high_u16(wa), vget_high_u16(wb));
        vst1_u8(D + i, vmovn_u16(vcombine_u16(lo, hi)));
    }
    frank_scalar(A + i, B + i, D + i, n - i);
}
#else
static inline void frank_poly(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { frank_scalar(A, B, D, n); }
#endif
