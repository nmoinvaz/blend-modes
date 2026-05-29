// blend.hlsl - blend modes as a Direct3D 12 / D3D11 compute shader (Windows).
// Bytes packed 4/u32 in ByteAddressBuffers (matches the WebGPU layout). a = top,
// b = bottom, per channel in [0,1]. D3D's fixed-function blend can only do linear
// src*f (+) dst*f, so the advanced modes live here in a shader (read the backdrop
// via an SRV or ROV). exp2/log2/pow/atan2 are HLSL intrinsics -> hardware SFUs.
//   compile: dxc -T cs_6_0 -E main blend.hlsl

ByteAddressBuffer  A : register(t0);
ByteAddressBuffer  B : register(t1);
RWByteAddressBuffer D : register(u0);

static const float PI = 3.14159265;

float multiply(float a, float b)   { return a * b; }
float screen(float a, float b)     { return a + b - a * b; }
float difference(float a, float b) { return abs(a - b); }
float exclusion(float a, float b)  { return a + b - 2.0 * a * b; }
float reflectm(float a, float b)   { return saturate(a*a / max(1.0 - b, 1e-6)); }

float overlay(float a, float b)    { return lerp(2.0*a*b, 1.0 - 2.0*(1.0-a)*(1.0-b), step(0.5, a)); }
float hardlight(float a, float b)  { return overlay(b, a); }
float vividlight(float a, float b) {
    float lo = 1.0 - (1.0 - b) / max(2.0*a, 1e-6);
    float hi = b / max(2.0 - 2.0*a, 1e-6);
    return saturate(lerp(lo, hi, step(0.5, a)));
}
float softlight(float a, float b)  { return a*a + 2.0*b*a*(1.0 - a); }

float frank(float a, float b) {
    float S = 10.0, L = log2(S);
    float t = (exp2(a*L) - 1.0) * (exp2(b*L) - 1.0) / (S - 1.0);
    return log2(1.0 + t) / L;
}
float gammalight(float a, float b) { return pow(b, a); }

uint blend_px(uint pa, uint pb) {
    uint o = 0;
    [unroll] for (uint c = 0; c < 4; c++) {
        float a = ((pa >> (c*8)) & 0xff) / 255.0;
        float b = ((pb >> (c*8)) & 0xff) / 255.0;
        uint r = (uint)clamp(round(overlay(a, b) * 255.0), 0.0, 255.0);  // swap in any mode
        o |= r << (c*8);
    }
    return o;
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint addr = gid.x * 4;                 // byte offset of one u32
    D.Store(addr, blend_px(A.Load(addr), B.Load(addr)));
}
