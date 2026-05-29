// webgpu_bench.ts - WGSL compute blend throughput (multiply / reflect / frank).
// Run: deno run --unstable-webgpu bench/webgpu_bench.ts
// On macOS this goes WebGPU -> wgpu -> Metal; portable to D3D12 (Windows) / Vulkan (Linux).
// Bytes are packed 4/u32 so memory traffic matches the Metal (1 byte/px) benchmark.

const N = 64 << 20;                 // 64 MPix
const NU32 = N / 4;                  // 4 bytes packed per u32
const PER = 4;                       // u32 handled per invocation (16 px)
const WORKGROUPS = NU32 / PER / 256; // = 16384, under the 65535 dim limit
const REPS = 60;

const BLENDS: Record<string, string> = {
  multiply: `return a * b;`,
  reflect:  `return select(min(1.0, a*a/max(1.0-b, 1e-6)), b, b >= 1.0);`,
  frank:    `let S = 10.0; let L = log2(S);
             let t = (exp2(a*L)-1.0)*(exp2(b*L)-1.0)/(S-1.0);
             return log2(1.0 + t) / L;`,
};

function shader(body: string): string {
  return `
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> D: array<u32>;
fn blend(a: f32, b: f32) -> f32 { ${body} }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * ${PER}u;
  for (var k = 0u; k < ${PER}u; k = k + 1u) {
    let i = base + k;
    let pa = A[i]; let pb = B[i]; var o: u32 = 0u;
    for (var c = 0u; c < 4u; c = c + 1u) {
      let a = f32((pa >> (c*8u)) & 0xffu) / 255.0;
      let b = f32((pb >> (c*8u)) & 0xffu) / 255.0;
      let r = clamp(round(blend(a, b) * 255.0), 0.0, 255.0);
      o = o | (u32(r) << (c*8u));
    }
    D[i] = o;
  }
}`;
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { console.error("no WebGPU adapter"); Deno.exit(1); }
const device = await adapter.requestDevice();
console.log("adapter:", adapter.info?.device || adapter.info?.vendor || "(unknown)");

const bytes = NU32 * 4;
const mk = () => device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const bA = mk(), bB = mk(), bD = mk();

// fill A,B with the same LCG as the CPU/Metal benchmarks
const a8 = new Uint8Array(bytes), b8 = new Uint8Array(bytes);
let s = 12345 >>> 0;
for (let i = 0; i < bytes; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a8[i] = s >>> 24; }
for (let i = 0; i < bytes; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; b8[i] = s >>> 24; }
device.queue.writeBuffer(bA, 0, a8);
device.queue.writeBuffer(bB, 0, b8);

console.log(`${"mode".padEnd(10)} ${"GPU(WebGPU)".padStart(12)}   (Gpix/s)`);
console.log("-".repeat(40));
for (const [name, body] of Object.entries(BLENDS)) {
  const mod = device.createShaderModule({ code: shader(body) });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bA } },
      { binding: 1, resource: { buffer: bB } },
      { binding: 2, resource: { buffer: bD } },
    ],
  });
  const run = () => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe); p.setBindGroup(0, bind);
    p.dispatchWorkgroups(WORKGROUPS); p.end();
    device.queue.submit([enc.finish()]);
  };
  for (let r = 0; r < 3; r++) run();                 // warmup
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let r = 0; r < REPS; r++) run();
  await device.queue.onSubmittedWorkDone();
  const sec = (performance.now() - t0) / 1000;
  const gpix = (N * REPS) / sec / 1e9;
  console.log(`${name.padEnd(10)} ${gpix.toFixed(2).padStart(12)}`);
}
