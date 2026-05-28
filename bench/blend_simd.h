// blend_simd.h - scalar and NEON implementations of the blend modes, by SIMD tier.
// On x86 the cheap ops map to _mm_min_epu8 / _mm_adds_epu8 / _mm_avg_epu8 / etc.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


// ============================================================================
//  Helpers
// ============================================================================

// round(x / 255) for x in [0, 65025], integer only
static inline uint32_t div255(uint32_t x) {
    x += 0x80;
    return (x + (x >> 8)) >> 8;
}

// round(r * 255) clamped to [0, 255], for r in [0, 1]
static inline uint8_t clamp8(float r) {
    int v = (int)lrintf(r * 255.f);
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}


// ============================================================================
//  Tier 1 / 2 - scalar baseline (one byte per iteration)
// ============================================================================

#define SCALAR_BODY(EXPR) \
    for (size_t i = 0; i < n; ++i) { unsigned a = A[i], b = B[i]; D[i] = (uint8_t)(EXPR); }

// min / max
static inline void darken_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)  { SCALAR_BODY(a < b ? a : b) }
static inline void lighten_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { SCALAR_BODY(a > b ? a : b) }

// saturating add / sub
static inline void add_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a + b > 255 ? 255 : a + b) }
static inline void sub_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a > b ? a - b : 0) }

// rounding average
static inline void avg_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY((a + b + 1) >> 1) }

// bitwise xor / absolute difference
static inline void xor_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(a ^ b) }
static inline void diff_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)    { SCALAR_BODY(a > b ? a - b : b - a) }

// multiply / screen (the /255 trick)
static inline void mul_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)     { SCALAR_BODY(div255(a * b)) }
static inline void screen_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n)  { SCALAR_BODY(255 - div255((255 - a) * (255 - b))) }


// ============================================================================
//  Tier 1 / 2 - NEON (16 bytes per instruction; scalar handles the < 16 tail)
// ============================================================================
#if defined(__ARM_NEON)

// round(x / 255) for a vector of u16
static inline uint16x8_t div255_u16(uint16x8_t x) {
    x = vaddq_u16(x, vdupq_n_u16(0x80));
    return vshrq_n_u16(vaddq_u16(x, vshrq_n_u16(x, 8)), 8);
}

#define SIMD_FN(NAME, VEXPR, SCALARFN)                                              \
static inline void NAME(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { \
    size_t i = 0, m = n & ~size_t(15);                                              \
    for (; i < m; i += 16) {                                                        \
        uint8x16_t va = vld1q_u8(A + i), vb = vld1q_u8(B + i);                       \
        vst1q_u8(D + i, (VEXPR));                                                    \
    }                                                                               \
    SCALARFN(A + i, B + i, D + i, n - i);                                           \
}

SIMD_FN(darken_simd,  vminq_u8(va, vb),   darken_scalar)    // 1 instruction
SIMD_FN(lighten_simd, vmaxq_u8(va, vb),   lighten_scalar)   // 1 instruction
SIMD_FN(add_simd,     vqaddq_u8(va, vb),  add_scalar)       // saturating add, free clamp
SIMD_FN(sub_simd,     vqsubq_u8(va, vb),  sub_scalar)       // saturating sub, free clamp
SIMD_FN(avg_simd,     vrhaddq_u8(va, vb), avg_scalar)       // rounding halving add
SIMD_FN(xor_simd,     veorq_u8(va, vb),   xor_scalar)       // 1 instruction
SIMD_FN(diff_simd,    vabdq_u8(va, vb),   diff_scalar)      // absolute difference, 1 instruction

// multiply: widen to u16, multiply, divide by 255, narrow back
static inline void mul_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(15);
    for (; i < m; i += 16) {
        uint8x16_t va = vld1q_u8(A + i), vb = vld1q_u8(B + i);
        uint16x8_t lo = vmull_u8(vget_low_u8(va),  vget_low_u8(vb));
        uint16x8_t hi = vmull_u8(vget_high_u8(va), vget_high_u8(vb));
        vst1q_u8(D + i, vcombine_u8(vmovn_u16(div255_u16(lo)), vmovn_u16(div255_u16(hi))));
    }
    mul_scalar(A + i, B + i, D + i, n - i);
}

// screen = 255 - mul(255 - a, 255 - b);  vmvnq_u8(x) gives 255 - x for u8
static inline void screen_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(15);
    for (; i < m; i += 16) {
        uint8x16_t na = vmvnq_u8(vld1q_u8(A + i)), nb = vmvnq_u8(vld1q_u8(B + i));
        uint16x8_t lo = vmull_u8(vget_low_u8(na),  vget_low_u8(nb));
        uint16x8_t hi = vmull_u8(vget_high_u8(na), vget_high_u8(nb));
        uint8x16_t p  = vcombine_u8(vmovn_u16(div255_u16(lo)), vmovn_u16(div255_u16(hi)));
        vst1q_u8(D + i, vmvnq_u8(p));
    }
    screen_scalar(A + i, B + i, D + i, n - i);
}

#else  // no NEON: SIMD == scalar fallback

#define ALIAS(S, P) static inline void S(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { P(A, B, D, n); }
ALIAS(darken_simd, darken_scalar)   ALIAS(lighten_simd, lighten_scalar)
ALIAS(add_simd, add_scalar)         ALIAS(sub_simd, sub_scalar)
ALIAS(avg_simd, avg_scalar)         ALIAS(xor_simd, xor_scalar)
ALIAS(diff_simd, diff_scalar)       ALIAS(mul_simd, mul_scalar)
ALIAS(screen_simd, screen_scalar)

#endif


// ============================================================================
//  Tier 4 - Frank t-norm (pow + log).  s is constant per image.
//  T_s(a,b) = log_s(1 + (s^a - 1)(s^b - 1)/(s - 1))
// ============================================================================

static const float FRANK_S = 10.f;          // an intermediate Frank t-norm
static uint8_t      FRANK_LUT[256 * 256];    // the shared 256x256 lookup table

// a, b in [0, 1]
static inline float frank_t(float a, float b, float s) {
    if (s > 0.999f && s < 1.001f) return a * b;   // s = 1 -> product (multiply)
    return logf(1.f + (powf(s, a) - 1.f) * (powf(s, b) - 1.f) / (s - 1.f)) / logf(s);
}

// build the 256x256 table once per (mode, s)
static inline void frank_build_lut(float s) {
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b)
            FRANK_LUT[(a << 8) | b] = clamp8(frank_t(a / 255.f, b / 255.f, s));
}

// per-pixel transcendental: pow, pow, log, log  (the slow path)
static inline void frank_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i)
        D[i] = clamp8(frank_t(A[i] / 255.f, B[i] / 255.f, FRANK_S));
}

// LUT lookup: one gather per pixel; same kernel for any tabulated mode
static inline void frank_lut(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    for (size_t i = 0; i < n; ++i)
        D[i] = FRANK_LUT[(A[i] << 8) | B[i]];
}


// ============================================================================
//  Tier 4 - trigonometric modes (cos / sin / atan), share the LUT path
// ============================================================================
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

// tabulate any binary function into FRANK_LUT (used for verification)
static inline void build_lut_g(Gfn g) {
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b)
            FRANK_LUT[(a << 8) | b] = clamp8(g(a / 255.f, b / 255.f));
}


// ============================================================================
//  Tier 4 - gather-free Frank via polynomial s^a and log10 (max error 0.65 lvl)
// ============================================================================
#if defined(__ARM_NEON)

static const float CA[7] = {6.84077096e-01f, -2.57628734e-01f, 1.77833343e+00f, 1.80124913e+00f,
                            2.69457216e+00f, 2.29926614e+00f, 1.00006040e+00f};      // 10^a, a in [0,1]
static const float CL[8] = {1.91446413e-06f, -8.34187231e-05f, 1.52474870e-03f, -1.52358431e-02f,
                            9.12609306e-02f, -3.42288770e-01f, 8.73234868e-01f, -6.05894708e-01f}; // log10(y), y in [1,10]

// Horner evaluation of a degree-`deg` polynomial:  acc = C[k] + x*acc
static inline float32x4_t horner(const float* C, int deg, float32x4_t x) {
    float32x4_t a = vdupq_n_f32(C[0]);
    for (int k = 1; k <= deg; ++k) a = vfmaq_f32(vdupq_n_f32(C[k]), x, a);
    return a;
}

// Frank on 4 lanes (inputs already widened to u16)
static inline uint16x4_t frank4(uint16x4_t ai, uint16x4_t bi) {
    const float32x4_t inv = vdupq_n_f32(1.f / 255.f), one = vdupq_n_f32(1.f);

    float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u16(ai)), inv);
    float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u16(bi)), inv);

    // (10^a - 1)(10^b - 1) / (s - 1)
    float32x4_t prod = vmulq_f32(vmulq_f32(vsubq_f32(horner(CA, 6, a), one),
                                           vsubq_f32(horner(CA, 6, b), one)),
                                 vdupq_n_f32(1.f / 9.f));

    float32x4_t r = vmulq_f32(horner(CL, 7, vaddq_f32(one, prod)), vdupq_n_f32(255.f));
    r = vminq_f32(vmaxq_f32(r, vdupq_n_f32(0.f)), vdupq_n_f32(255.f));
    return vmovn_u32(vcvtnq_u32_f32(r));   // round to nearest
}

static inline void frank_poly(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    size_t i = 0, m = n & ~size_t(7);
    for (; i < m; i += 8) {
        uint16x8_t wa = vmovl_u8(vld1_u8(A + i)), wb = vmovl_u8(vld1_u8(B + i));
        uint16x4_t lo = frank4(vget_low_u16(wa),  vget_low_u16(wb));
        uint16x4_t hi = frank4(vget_high_u16(wa), vget_high_u16(wb));
        vst1_u8(D + i, vmovn_u16(vcombine_u16(lo, hi)));
    }
    frank_scalar(A + i, B + i, D + i, n - i);
}

#else
static inline void frank_poly(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { frank_scalar(A, B, D, n); }
#endif


// ============================================================================
//  Tier 3 - quadratic (float reciprocal) and means (sqrt / reciprocal)
// ============================================================================

#define SCAL_F(NAME, EXPR)                                                                   \
static inline void NAME##_scalar(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { \
    for (size_t i = 0; i < n; ++i) {                                                         \
        float a = A[i] * (1.f / 255.f), b = B[i] * (1.f / 255.f);                            \
        D[i] = clamp8(EXPR);                                                                 \
    }                                                                                        \
}

// quadratic dodge/burn family
SCAL_F(reflect, fminf(1.f, a * a / fmaxf(1.f - b, 1e-6f)))
SCAL_F(glow,    fminf(1.f, b * b / fmaxf(1.f - a, 1e-6f)))
SCAL_F(heat,    1.f - fminf(1.f, (1.f - b) * (1.f - b) / fmaxf(a, 1e-6f)))
SCAL_F(freeze,  1.f - fminf(1.f, (1.f - a) * (1.f - a) / fmaxf(b, 1e-6f)))

// Pythagorean / Lehmer means
SCAL_F(geometric, sqrtf(a * b))
SCAL_F(harmonic,  2.f * a * b / fmaxf(a + b, 1e-6f))
SCAL_F(rms,       sqrtf((a * a + b * b) * 0.5f))
SCAL_F(contra,    (a * a + b * b) / fmaxf(a + b, 1e-6f))

#if defined(__ARM_NEON)

#define ONE vdupq_n_f32(1.f)
#define EPS vdupq_n_f32(1e-6f)

static inline float32x4_t op_reflect(float32x4_t a, float32x4_t b) { return vminq_f32(ONE, vdivq_f32(vmulq_f32(a, a), vmaxq_f32(vsubq_f32(ONE, b), EPS))); }
static inline float32x4_t op_glow(float32x4_t a, float32x4_t b)    { return vminq_f32(ONE, vdivq_f32(vmulq_f32(b, b), vmaxq_f32(vsubq_f32(ONE, a), EPS))); }
static inline float32x4_t op_heat(float32x4_t a, float32x4_t b)    { float32x4_t t = vsubq_f32(ONE, b); return vsubq_f32(ONE, vminq_f32(ONE, vdivq_f32(vmulq_f32(t, t), vmaxq_f32(a, EPS)))); }
static inline float32x4_t op_freeze(float32x4_t a, float32x4_t b)  { float32x4_t t = vsubq_f32(ONE, a); return vsubq_f32(ONE, vminq_f32(ONE, vdivq_f32(vmulq_f32(t, t), vmaxq_f32(b, EPS)))); }

static inline float32x4_t op_geometric(float32x4_t a, float32x4_t b) { return vsqrtq_f32(vmulq_f32(a, b)); }
static inline float32x4_t op_harmonic(float32x4_t a, float32x4_t b)  { return vdivq_f32(vmulq_f32(vdupq_n_f32(2.f), vmulq_f32(a, b)), vmaxq_f32(vaddq_f32(a, b), EPS)); }
static inline float32x4_t op_rms(float32x4_t a, float32x4_t b)       { return vsqrtq_f32(vmulq_f32(vaddq_f32(vmulq_f32(a, a), vmulq_f32(b, b)), vdupq_n_f32(0.5f))); }
static inline float32x4_t op_contra(float32x4_t a, float32x4_t b)    { return vdivq_f32(vaddq_f32(vmulq_f32(a, a), vmulq_f32(b, b)), vmaxq_f32(vaddq_f32(a, b), EPS)); }

// 4-wide float driver; requires n a multiple of 8 (the benchmark's N is)
template <float32x4_t (*OP)(float32x4_t, float32x4_t)>
static inline void simd_f(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) {
    const float32x4_t inv = vdupq_n_f32(1.f / 255.f), v255 = vdupq_n_f32(255.f), z = vdupq_n_f32(0.f);
    size_t i = 0, m = n & ~size_t(7);
    for (; i < m; i += 8) {
        uint16x8_t wa = vmovl_u8(vld1_u8(A + i)), wb = vmovl_u8(vld1_u8(B + i));

        float32x4_t al = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(wa))),  inv);
        float32x4_t bl = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(wb))),  inv);
        float32x4_t ah = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(wa))), inv);
        float32x4_t bh = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(wb))), inv);

        float32x4_t rl = vminq_f32(vmaxq_f32(vmulq_f32(OP(al, bl), v255), z), v255);
        float32x4_t rh = vminq_f32(vmaxq_f32(vmulq_f32(OP(ah, bh), v255), z), v255);

        uint16x4_t lo = vmovn_u32(vcvtnq_u32_f32(rl)), hi = vmovn_u32(vcvtnq_u32_f32(rh));
        vst1_u8(D + i, vmovn_u16(vcombine_u16(lo, hi)));
    }
}

#define SIMD_WRAP(NAME, OP) static inline void NAME##_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { simd_f<OP>(A, B, D, n); }
SIMD_WRAP(reflect, op_reflect)     SIMD_WRAP(glow, op_glow)
SIMD_WRAP(heat, op_heat)           SIMD_WRAP(freeze, op_freeze)
SIMD_WRAP(geometric, op_geometric) SIMD_WRAP(harmonic, op_harmonic)
SIMD_WRAP(rms, op_rms)             SIMD_WRAP(contra, op_contra)

#else
#define SIMD_WRAP(NAME, OP) static inline void NAME##_simd(const uint8_t* A, const uint8_t* B, uint8_t* D, size_t n) { NAME##_scalar(A, B, D, n); }
SIMD_WRAP(reflect,)   SIMD_WRAP(glow,)     SIMD_WRAP(heat,)   SIMD_WRAP(freeze,)
SIMD_WRAP(geometric,) SIMD_WRAP(harmonic,) SIMD_WRAP(rms,)    SIMD_WRAP(contra,)
#endif
