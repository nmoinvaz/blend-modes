// bench.cpp - Google Benchmark harness: scalar vs SIMD across the blend-mode tiers.
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "blend_simd.h"


// ============================================================================
//  Test buffers
// ============================================================================

// 16 KiB per buffer -> A + B + D = 48 KiB, fits the 64 KiB L1d (compute-bound)
static constexpr size_t N = 1 << 14;

struct Buffers {
    std::vector<uint8_t> A, B, D;
    Buffers() : A(N), B(N), D(N) {
        unsigned s = 12345;   // deterministic pseudo-random fill
        for (size_t i = 0; i < N; ++i) { s = s * 1103515245u + 12345u; A[i] = s >> 24; }
        for (size_t i = 0; i < N; ++i) { s = s * 1103515245u + 12345u; B[i] = s >> 24; }
    }
};
static Buffers g;


// ============================================================================
//  Runner
// ============================================================================

using Fn = void(*)(const uint8_t*, const uint8_t*, uint8_t*, size_t);

// passes per timed iteration; amortizes per-call overhead on the small buffer
static constexpr int REPEAT = 256;

template <Fn F>
static void Run(benchmark::State& state) {
    for (auto _ : state) {
        for (int r = 0; r < REPEAT; ++r) {
            F(g.A.data(), g.B.data(), g.D.data(), N);
            benchmark::DoNotOptimize(g.D.data());
        }
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * N * REPEAT);
}

// report max|simd - scalar|; 0 = bit-exact, 1 = rounding-only
static int verify(Fn s, Fn v, const char* name) {
    std::vector<uint8_t> ds(N), dv(N);
    s(g.A.data(), g.B.data(), ds.data(), N);
    v(g.A.data(), g.B.data(), dv.data(), N);

    int maxd = 0;
    for (size_t i = 0; i < N; ++i) {
        int d = std::abs(int(ds[i]) - int(dv[i]));
        if (d > maxd) maxd = d;
    }
    std::printf("verify %-11s max|simd-scalar| = %d\n", name, maxd);
    return maxd;
}


// ============================================================================
//  Benchmark registration
// ============================================================================

#define PAIR(NAME, SC, SV)                            \
    BENCHMARK_TEMPLATE(Run, SC)->Name(NAME "/scalar"); \
    BENCHMARK_TEMPLATE(Run, SV)->Name(NAME "/simd");

// Tier 1/2: 16-wide uint8 (min/max/sat/avg/xor/diff, multiply, screen)
PAIR("darken",   darken_scalar,  darken_simd)
PAIR("lighten",  lighten_scalar, lighten_simd)
PAIR("add",      add_scalar,     add_simd)
PAIR("subtract", sub_scalar,     sub_simd)
PAIR("average",  avg_scalar,     avg_simd)
PAIR("xor",      xor_scalar,     xor_simd)
PAIR("diff",     diff_scalar,    diff_simd)
PAIR("multiply", mul_scalar,     mul_simd)
PAIR("screen",   screen_scalar,  screen_simd)

// Tier 3: 4-wide float (quadratic via reciprocal; means via sqrt/reciprocal)
PAIR("reflect",   reflect_scalar,   reflect_simd)
PAIR("glow",      glow_scalar,      glow_simd)
PAIR("heat",      heat_scalar,      heat_simd)
PAIR("freeze",    freeze_scalar,    freeze_simd)
PAIR("geometric", geometric_scalar, geometric_simd)
PAIR("harmonic",  harmonic_scalar,  harmonic_simd)
PAIR("rms",       rms_scalar,       rms_simd)
PAIR("contra",    contra_scalar,    contra_simd)

// Tier 4: transcendental - per-pixel compute vs LUT vs gather-free polynomial
BENCHMARK_TEMPLATE(Run, frank_scalar)->Name("frank/scalar");
BENCHMARK_TEMPLATE(Run, interp_scalar)->Name("interpolate/scalar");
BENCHMARK_TEMPLATE(Run, sine_scalar)->Name("sine/scalar");
BENCHMARK_TEMPLATE(Run, arctan_scalar)->Name("arctan/scalar");
BENCHMARK_TEMPLATE(Run, arctan_simd)->Name("arctan/simd-poly");
BENCHMARK_TEMPLATE(Run, frank_lut)->Name("frank/LUT-apply (gather)");
BENCHMARK_TEMPLATE(Run, frank_poly)->Name("frank/NEON-poly (no gather)");

// Optimization experiments: fast reciprocal/rsqrt (Tier 3) and unrolling
BENCHMARK_TEMPLATE(Run, reflect_fast_simd)->Name("reflect/simd-recip");
BENCHMARK_TEMPLATE(Run, harmonic_fast_simd)->Name("harmonic/simd-recip");
BENCHMARK_TEMPLATE(Run, geometric_fast_simd)->Name("geometric/simd-rsqrt");
BENCHMARK_TEMPLATE(Run, rms_fast_simd)->Name("rms/simd-rsqrt");
BENCHMARK_TEMPLATE(Run, add_simd_u4)->Name("add/simd-unroll4");
BENCHMARK_TEMPLATE(Run, frank_poly_u)->Name("frank/NEON-poly-unroll16");

// across-the-board unroll: more Tier-1 ops, and Tier-3 float with native div/sqrt
BENCHMARK_TEMPLATE(Run, darken_u4)->Name("darken/simd-unroll4");
BENCHMARK_TEMPLATE(Run, xor_u4)->Name("xor/simd-unroll4");
BENCHMARK_TEMPLATE(Run, diff_u4)->Name("diff/simd-unroll4");
BENCHMARK_TEMPLATE(Run, avg_u4)->Name("average/simd-unroll4");
BENCHMARK_TEMPLATE(Run, reflect_u_simd)->Name("reflect/simd-unroll16");
BENCHMARK_TEMPLATE(Run, geometric_u_simd)->Name("geometric/simd-unroll16");
BENCHMARK_TEMPLATE(Run, harmonic_u_simd)->Name("harmonic/simd-unroll16");


// ============================================================================
//  main: verify correctness, then run
// ============================================================================

int main(int argc, char** argv) {
    // Tier 1/2/3: SIMD vs scalar (bit-exact, or +-1 from rounding)
    verify(darken_scalar, darken_simd, "darken");
    verify(add_scalar, add_simd, "add");
    verify(diff_scalar, diff_simd, "diff");
    verify(mul_scalar, mul_simd, "multiply");
    verify(screen_scalar, screen_simd, "screen");
    verify(reflect_scalar, reflect_simd, "reflect");
    verify(geometric_scalar, geometric_simd, "geometric");
    verify(harmonic_scalar, harmonic_simd, "harmonic");
    verify(contra_scalar, contra_simd, "contra");

    // Tier 4: each mode's LUT (and the polynomial) vs its scalar compute
    frank_build_lut(FRANK_S);  verify(frank_scalar,  frank_lut,  "frank");
    verify(frank_scalar, frank_poly, "frank-poly");
    build_lut_g(g_interp);     verify(interp_scalar, frank_lut,  "interpolate");
    build_lut_g(g_sine);       verify(sine_scalar,   frank_lut,  "sine");
    build_lut_g(g_arctan);     verify(arctan_scalar, frank_lut,  "arctan");
    verify(arctan_scalar, arctan_simd, "arctan-poly");
    frank_build_lut(FRANK_S);  // leave a valid table for the LUT-apply benchmark

    // optimization experiments vs their scalar baselines
    verify(reflect_scalar, reflect_fast_simd, "reflect-recip");
    verify(harmonic_scalar, harmonic_fast_simd, "harmonic-recip");
    verify(geometric_scalar, geometric_fast_simd, "geom-rsqrt");
    verify(rms_scalar, rms_fast_simd, "rms-rsqrt");
    verify(frank_scalar, frank_poly_u, "frank-poly-u");
    verify(add_scalar, add_simd_u4, "add-unroll4");
    verify(darken_scalar, darken_u4, "darken-u4");
    verify(reflect_scalar, reflect_u_simd, "reflect-u16");
    verify(geometric_scalar, geometric_u_simd, "geom-u16");
    std::printf("---\n");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
