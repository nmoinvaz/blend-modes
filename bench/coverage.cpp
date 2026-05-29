// coverage.cpp - verify every atlas mode has a working table-free SIMD kernel.
// Builds the full 256x256 input grid, runs each registry kernel, and compares to
// the reference LUT exported from the Python atlas (bench/ref_luts.bin).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include "all_modes.h"

static constexpr int N = 256 * 256;   // full grid: index = a*256 + b

int main() {
    // load reference names + LUTs (atlas is the source of truth)
    FILE* fn = fopen("ref_names.txt", "r");
    FILE* fl = fopen("ref_luts.bin", "rb");
    if (!fn || !fl) { std::printf("missing ref_names.txt / ref_luts.bin (run scripts export)\n"); return 1; }

    std::vector<std::string> names;
    char line[64];
    while (fgets(line, sizeof line, fn)) { line[strcspn(line, "\n")] = 0; if (line[0]) names.push_back(line); }
    std::unordered_map<std::string, int> idx;
    for (int i = 0; i < (int)names.size(); ++i) idx[names[i]] = i;

    std::vector<uint8_t> ref(names.size() * (size_t)N);
    if (fread(ref.data(), 1, ref.size(), fl) != ref.size()) { std::printf("ref_luts.bin short read\n"); return 1; }
    fclose(fn); fclose(fl);

    // input grid: A = top index a, B = bottom index b
    std::vector<uint8_t> A(N), B(N), D(N);
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b) { A[a * 256 + b] = a; B[a * 256 + b] = b; }

    // optional: dump bad cells for one mode
    const char* only = getenv("MODE");
    if (only) {
        int j = idx.count(only) ? idx[only] : -1;
        for (int i = 0; i < MODE_COUNT; ++i) if (!strcmp(MODE_TABLE[i].name, only) && j >= 0) {
            MODE_TABLE[i].simd(A.data(), B.data(), D.data(), N);
            const uint8_t* r = &ref[(size_t)j * N];
            int shown = 0;
            for (int k = 0; k < N && shown < 16; ++k)
                if (std::abs(int(D[k]) - int(r[k])) > 1)
                    { std::printf("  a=%3d b=%3d got=%3d want=%3d\n", k>>8, k&255, D[k], r[k]); shown++; }
            return 0;
        }
    }

    int worst = 0, missing = 0, over2 = 0;
    std::printf("%-16s %-8s %s\n", "mode", "maxerr", "status");
    std::printf("------------------------------------------\n");
    for (int i = 0; i < MODE_COUNT; ++i) {
        const ModeEntry& e = MODE_TABLE[i];
        auto it = idx.find(e.name);
        if (it == idx.end()) { std::printf("%-16s    -     NO REFERENCE\n", e.name); missing++; continue; }
        const uint8_t* r = &ref[(size_t)it->second * N];

        e.simd(A.data(), B.data(), D.data(), N);
        int md = 0, nbad = 0, wa = 0, wb = 0, wgot = 0, wwant = 0;
        for (int k = 0; k < N; ++k) {
            int d = std::abs(int(D[k]) - int(r[k]));
            if (d > 1) nbad++;
            if (d > md) { md = d; wa = k >> 8; wb = k & 255; wgot = D[k]; wwant = r[k]; }
        }
        if (md > worst) worst = md;
        const char* tag = md <= 1 ? "ok" : md <= 3 ? "ok~" : "CHECK";
        if (md > 3) over2++;
        if (md <= 3) std::printf("%-16s %4d     %s\n", e.name, md, tag);
        else std::printf("%-16s %4d     %s  bad=%d  worst@(a=%d,b=%d) got=%d want=%d\n",
                         e.name, md, tag, nbad, wa, wb, wgot, wwant);
    }

    // any reference modes with no kernel?
    int uncovered = 0;
    for (auto& nm : names) {
        bool found = false;
        for (int i = 0; i < MODE_COUNT; ++i) if (nm == MODE_TABLE[i].name) { found = true; break; }
        if (!found) { std::printf("UNCOVERED: %s\n", nm.c_str()); uncovered++; }
    }

    std::printf("------------------------------------------\n");
    std::printf("modes=%d  worst_err=%d  >3lvl=%d  missing_ref=%d  uncovered=%d\n",
                MODE_COUNT, worst, over2, missing, uncovered);
    return (over2 || uncovered || missing) ? 2 : 0;
}
