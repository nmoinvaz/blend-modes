#version 320 es
// blend.glsl - blend modes as a GLSL fragment shader using framebuffer fetch
// (tile GPUs: Apple/Mali/Adreno keep the destination tile in on-chip memory, so
// reading the backdrop is ~free). On desktop/immediate GPUs, sample the backdrop
// as a texture instead. a = src (top), b = dst (bottom), per channel in [0,1].
#extension GL_EXT_shader_framebuffer_fetch : require
precision highp float;

layout(location = 0) inout vec4 fragColor;   // fragColor.rgb = dst = backdrop (read + write)
in vec2 uv;
uniform sampler2D uLayer;                     // src = top layer

const float PI = 3.14159265;

vec3 multiply(vec3 a, vec3 b)   { return a * b; }
vec3 screen(vec3 a, vec3 b)     { return a + b - a * b; }
vec3 darken(vec3 a, vec3 b)     { return min(a, b); }
vec3 lighten(vec3 a, vec3 b)    { return max(a, b); }
vec3 difference(vec3 a, vec3 b) { return abs(a - b); }
vec3 exclusion(vec3 a, vec3 b)  { return a + b - 2.0 * a * b; }
vec3 reflectm(vec3 a, vec3 b)   { return clamp(a*a / max(1.0 - b, vec3(1e-6)), 0.0, 1.0); }

// contrast fuse: mix() + step() is branchless, avoids warp divergence
vec3 overlay(vec3 a, vec3 b)    { return mix(2.0*a*b, 1.0 - 2.0*(1.0-a)*(1.0-b), step(0.5, a)); }
vec3 hardlight(vec3 a, vec3 b)  { return overlay(b, a); }
vec3 vividlight(vec3 a, vec3 b) {
    vec3 lo = 1.0 - (1.0 - b) / max(2.0*a, vec3(1e-6));
    vec3 hi = b / max(2.0 - 2.0*a, vec3(1e-6));
    return clamp(mix(lo, hi, step(0.5, a)), 0.0, 1.0);
}
vec3 softlight(vec3 a, vec3 b)  { return a*a + 2.0*b*a*(1.0 - a); }

// transcendental: exp2/log2/pow are hardware intrinsics
vec3 frank(vec3 a, vec3 b) {
    float S = 10.0, L = log2(S);
    vec3 t = (exp2(a*L) - 1.0) * (exp2(b*L) - 1.0) / (S - 1.0);
    return log2(1.0 + t) / L;
}
vec3 gammalight(vec3 a, vec3 b) { return pow(b, a); }

void main() {
    vec3 a = texture(uLayer, uv).rgb;   // top
    vec3 b = fragColor.rgb;             // bottom (framebuffer fetch)
    fragColor = vec4(overlay(a, b), 1.0);   // swap in any mode above
}
