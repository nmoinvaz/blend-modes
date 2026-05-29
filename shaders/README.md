# GPU shaders

Blend modes as GPU shaders, across the three portable targets. Same math as the CPU
kernels in [`../bench/all_modes.h`](../bench/all_modes.h), but the GPU story is different:
`exp2`/`log2`/`pow`/`atan` are hardware SFU intrinsics, so the transcendental modes (frank,
gamma, trig) cost about the same as multiply — the CPU compute-tier penalty disappears (see the
benchmark below).

| file | target | language | how the backdrop is read |
|---|---|---|---|
| [`blend.wgsl`](blend.wgsl) | WebGPU → Metal / D3D12 / Vulkan | WGSL | compute over storage buffers |
| [`blend.glsl`](blend.glsl) | OpenGL ES / desktop GL | GLSL | framebuffer fetch (tile GPUs) or texture sample |
| [`blend.hlsl`](blend.hlsl) | Direct3D 12 / 11 (Windows) | HLSL | `ByteAddressBuffer` (compute) or ROV (programmable blend) |

Branches use `select`/`mix`+`step` to stay branchless (avoids warp divergence).

## Two blend paths on a GPU

1. **Fixed-function blend** (the ROP / output-merger) does only linear `src·f ⊕ dst·f` — enough for
   normal/add/darken/lighten/multiply, nothing branchy. Free, but limited.
2. **Programmable blend** — a shader reads the backdrop and computes any `f(a,b)`. That is what these
   files do. Reading the backdrop needs: `KHR_blend_equation_advanced` (the 12 Photoshop modes baked
   into GL/Vulkan HW), framebuffer fetch (tile GPUs), ROVs (D3D), or a compute pass over storage images.

## Benchmark (WGSL, run here)

[`../bench/webgpu_bench.ts`](../bench/webgpu_bench.ts) runs `blend.wgsl` through Deno's WebGPU
(→ wgpu → Metal on this machine), 64 MPix, Apple M5:

```sh
deno run --unstable-webgpu bench/webgpu_bench.ts
```

| mode (tier) | scalar | NEON | WebGPU | Metal (native) | units |
|---|---|---|---|---|---|
| multiply (T2) | 1.86 | 2.51 | 11.6 | 15.0 | Gpix/s |
| reflect (T3)  | 0.23 | 0.43 | 15.2 | 10.0 | Gpix/s |
| frank (T4)    | 0.02 | 0.17 | 13.6 | 23.7 | Gpix/s |

Portable WGSL reaches ~75–95% of native Metal and 30–80× over NEON. The CPU compute-tier spread
(~90×) collapses to ~1.3× on GPU — `frank` is no longer the slowest. Note: unified memory (Apple
Silicon, integrated GPUs) means no upload/download cost; a discrete GPU adds PCIe transfer that can
erase the win for a one-shot blend.
