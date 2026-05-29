// gpu_bench.mm - Metal GPU vs scalar vs NEON for representative blend tiers.
// Apple Silicon unified memory: CPU and GPU share buffers, so there is no
// upload/download cost to confound the comparison. Same buffers, same size.
//   build: clang++ -fobjc-arc -O3 -std=c++17 -framework Metal -framework Foundation \
//          gpu_bench.mm -o gpu_bench && ./gpu_bench
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <ctime>
#include "blend_simd.h"   // CPU scalar + NEON kernels (multiply, reflect, frank)

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// uint8 in/out, float compute - mirrors the CPU kernels. S=10 Frank t-norm.
static NSString* kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void multiply_g(device const uchar* A[[buffer(0)]], device const uchar* B[[buffer(1)]],
                       device uchar* D[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    float a = A[i]/255.0f, b = B[i]/255.0f;
    D[i] = uchar(clamp(round(a*b*255.0f), 0.0f, 255.0f));
}
kernel void reflect_g(device const uchar* A[[buffer(0)]], device const uchar* B[[buffer(1)]],
                      device uchar* D[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    float a = A[i]/255.0f, b = B[i]/255.0f;
    float r = (b >= 1.0f) ? b : min(1.0f, a*a/max(1.0f-b, 1e-6f));
    D[i] = uchar(clamp(round(r*255.0f), 0.0f, 255.0f));
}
kernel void frank_g(device const uchar* A[[buffer(0)]], device const uchar* B[[buffer(1)]],
                    device uchar* D[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    float a = A[i]/255.0f, b = B[i]/255.0f, S = 10.0f, L = log2(S);
    float t = (exp2(a*L)-1.0f)*(exp2(b*L)-1.0f)/(S-1.0f);
    D[i] = uchar(clamp(round(log2(1.0f+t)/L*255.0f), 0.0f, 255.0f));
}
)";

struct CpuFn { const char* name; void(*scalar)(const uint8_t*,const uint8_t*,uint8_t*,size_t);
                                  void(*simd)(const uint8_t*,const uint8_t*,uint8_t*,size_t); };

int main() {
    const size_t N = 64ull << 20;            // 64 MPix: RAM-resident, amortizes GPU launch
    const int REPS = 60;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
    if (!lib) { printf("shader compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    // shared buffers (unified memory - no copy)
    id<MTLBuffer> bA = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    id<MTLBuffer> bD = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    uint8_t *A = (uint8_t*)bA.contents, *B = (uint8_t*)bB.contents, *D = (uint8_t*)bD.contents;
    unsigned s = 12345;
    for (size_t i = 0; i < N; ++i) { s = s*1103515245u+12345u; A[i] = s>>24; }
    for (size_t i = 0; i < N; ++i) { s = s*1103515245u+12345u; B[i] = s>>24; }

    const char* names[] = {"multiply", "reflect", "frank"};
    const char* gpuk[]  = {"multiply_g", "reflect_g", "frank_g"};
    CpuFn cpu[] = {
        {"multiply", mul_scalar,     mul_simd},
        {"reflect",  reflect_scalar, reflect_simd},
        {"frank",    frank_scalar,   frank_poly},
    };

    printf("%-10s %12s %12s %12s   %s\n", "mode", "scalar", "NEON", "GPU(Metal)", "(Gpix/s)");
    printf("---------------------------------------------------------------\n");
    for (int m = 0; m < 3; ++m) {
        // CPU scalar
        double t0 = now_s();
        for (int r = 0; r < REPS; ++r) cpu[m].scalar(A, B, D, N);
        double sc = N*(double)REPS/(now_s()-t0)/1e9;
        // CPU NEON
        t0 = now_s();
        for (int r = 0; r < REPS; ++r) cpu[m].simd(A, B, D, N);
        double ne = N*(double)REPS/(now_s()-t0)/1e9;
        // GPU
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:gpuk[m]]];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        MTLSize grid = MTLSizeMake(N,1,1), tg = MTLSizeMake(256,1,1);
        // warmup
        for (int r = 0; r < 3; ++r) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps]; [e setBuffer:bA offset:0 atIndex:0];
            [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bD offset:0 atIndex:2];
            [e dispatchThreads:grid threadsPerThreadgroup:tg]; [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        t0 = now_s();
        for (int r = 0; r < REPS; ++r) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps]; [e setBuffer:bA offset:0 atIndex:0];
            [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bD offset:0 atIndex:2];
            [e dispatchThreads:grid threadsPerThreadgroup:tg]; [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        double gp = N*(double)REPS/(now_s()-t0)/1e9;
        printf("%-10s %12.2f %12.2f %12.2f\n", names[m], sc, ne, gp);
    }
    return 0;
}
