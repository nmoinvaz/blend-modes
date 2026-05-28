// Google Benchmark: scalar vs NEON SIMD for the SIMD-friendly blend modes.
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "blend_simd.h"

static constexpr size_t N = 1 << 14;   // 16 KiB per buffer -> A+B+D = 48 KiB, fits 64 KiB L1d

struct Buffers {
    std::vector<uint8_t> A, B, D;
    Buffers() : A(N), B(N), D(N) {
        unsigned s = 12345;
        for (size_t i = 0; i < N; ++i) { s = s * 1103515245u + 12345u; A[i] = s >> 24; }
        for (size_t i = 0; i < N; ++i) { s = s * 1103515245u + 12345u; B[i] = s >> 24; }
    }
};
static Buffers g;

using Fn = void(*)(const uint8_t*, const uint8_t*, uint8_t*, size_t);

static constexpr int REPEAT = 256;     // passes per timed iteration -> amortize call overhead

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

// verify each SIMD impl matches scalar within 1 level (div255 rounding)
static int verify(Fn s, Fn v, const char* name) {
    std::vector<uint8_t> ds(N), dv(N);
    s(g.A.data(), g.B.data(), ds.data(), N);
    v(g.A.data(), g.B.data(), dv.data(), N);
    int maxd = 0;
    for (size_t i = 0; i < N; ++i) { int d = std::abs(int(ds[i]) - int(dv[i])); if (d > maxd) maxd = d; }
    std::printf("verify %-9s max|simd-scalar| = %d\n", name, maxd);
    return maxd;
}

#define PAIR(NAME, SC, SV)                                                       \
    BENCHMARK_TEMPLATE(Run, SC)->Name(NAME "/scalar");                           \
    BENCHMARK_TEMPLATE(Run, SV)->Name(NAME "/simd");

PAIR("darken",  darken_scalar,  darken_simd)
PAIR("lighten", lighten_scalar, lighten_simd)
PAIR("add",     add_scalar,     add_simd)
PAIR("subtract",sub_scalar,     sub_simd)
PAIR("average", avg_scalar,     avg_simd)
PAIR("xor",     xor_scalar,     xor_simd)
PAIR("diff",    diff_scalar,    diff_simd)
PAIR("multiply",mul_scalar,     mul_simd)
PAIR("screen",  screen_scalar,  screen_simd)

// Tier-4 transcendental modes: per-pixel compute vs the shared 256x256 LUT-apply.
BENCHMARK_TEMPLATE(Run, frank_scalar)->Name("frank/scalar");
BENCHMARK_TEMPLATE(Run, interp_scalar)->Name("interpolate/scalar");
BENCHMARK_TEMPLATE(Run, sine_scalar)->Name("sine/scalar");
BENCHMARK_TEMPLATE(Run, arctan_scalar)->Name("arctan/scalar");
BENCHMARK_TEMPLATE(Run, frank_lut)->Name("frank/LUT-apply (gather)");
BENCHMARK_TEMPLATE(Run, frank_poly)->Name("frank/NEON-poly (no gather)");

int main(int argc, char** argv) {
    int bad = 0;
    bad += verify(darken_scalar, darken_simd, "darken");
    bad += verify(add_scalar, add_simd, "add");
    bad += verify(diff_scalar, diff_simd, "diff");
    bad += verify(mul_scalar, mul_simd, "multiply");
    bad += verify(screen_scalar, screen_simd, "screen");
    // Tier-4: verify each mode's LUT matches its scalar compute (within rounding)
    frank_build_lut(FRANK_S);   bad += verify(frank_scalar,  frank_lut, "frank");
    verify(frank_scalar, frank_poly, "frank-poly");   // gather-free polynomial path
    build_lut_g(g_interp);      bad += verify(interp_scalar, frank_lut, "interpolate");
    build_lut_g(g_sine);        bad += verify(sine_scalar,   frank_lut, "sine");
    build_lut_g(g_arctan);      bad += verify(arctan_scalar, frank_lut, "arctan");
    frank_build_lut(FRANK_S);   // leave a valid table for the LUT-apply benchmark
    std::printf("---\n");
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
