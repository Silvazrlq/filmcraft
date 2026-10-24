// Roughen Edges as a three-pass plan (crates/render/src/gpufx_roughen.rs), the GPU form of
// `roughen_edges` in crates/render/src/vfx/stylize.rs.
//
// CONCATENATION: this file uses U64, mulhi32, u64_of_i32, xor64, shr64, mul64 and hash_u64 from
// `noise_u64.wgsl`. WGSL has no include, so the integrator prepends `noise_u64.wgsl` to this file
// before building the module.
//
// Bindings (kernel contract of fx_multipass.wgsl, plus one uniform):
//   0: input0  (texture_2d<f32>, unfilterable, textureLoad)
//   1: input1  (texture_2d<f32>, unfilterable, textureLoad)
//   2: output  (texture_storage_2d<rgba32float, write>)
//   3: params  (uniform array<vec4<u32>, 8>: the pass parameter block, RoughenParams::uniform_bytes,
//               80 bytes zero-padded; read word by word, f32 words through bitcast)
//
// Passes: cs_roughen_blur_h (slot0 alpha -> slot1), cs_roughen_blur_v (slot1 -> slot2),
// cs_roughen_apply (slot0 + slot2 -> slot3).
//
// Blur note: the CPU reference blurs with three box passes per axis (boxes_for_gauss). Those need
// three different radii, and a plan pass carries no per-pass parameter, so this file blurs with a
// truncated Gaussian of the same sigma (radius ceil(3 sigma), capped at MAX_RADIUS), zero outside
// the image. The sigma cap (max(dim, 8) * 2) matches the CPU.

@group(0) @binding(0) var input0: texture_2d<f32>;
@group(0) @binding(1) var input1: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 8>;

// Word layout of the parameter block (RoughenParams::uniform_bytes):
//   0 sigma, 1 sharpness, 2 influence, 3 scale (px), 4 stretch (2^x), 5 offset x, 6 offset y,
//   7 complexity, 8 evolution (turns), 9..11 unused, 12 seed lo, 13 seed hi, 14 edge type,
//   15 unused, 16..18 edge colour (linear RGB), 19 unused.
fn pword(i: u32) -> u32 {
    return params[i / 4u][i % 4u];
}

fn pf(i: u32) -> f32 {
    return bitcast<f32>(pword(i));
}

const MAX_RADIUS: f32 = 512.0;

fn add64(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    var carry = 0u;
    if (lo < a.lo) {
        carry = 1u;
    }
    return U64(a.hi + b.hi + carry, lo);
}

// h mod 12 for a 64-bit value: 2^32 mod 12 = 4.
fn mod12(h: U64) -> u32 {
    return ((h.hi % 12u) * 4u + (h.lo % 12u)) % 12u;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// The 12 cube edge directions of effects' grad3 (same index order as the CPU match).
fn grad3(h: U64, x: f32, y: f32, z: f32) -> f32 {
    let m = mod12(h);
    var g = -y - z;
    if (m == 0u) { g = x + y; }
    else if (m == 1u) { g = -x + y; }
    else if (m == 2u) { g = x - y; }
    else if (m == 3u) { g = -x - y; }
    else if (m == 4u) { g = x + z; }
    else if (m == 5u) { g = -x + z; }
    else if (m == 6u) { g = x - z; }
    else if (m == 7u) { g = -x - z; }
    else if (m == 8u) { g = y + z; }
    else if (m == 9u) { g = -y + z; }
    else if (m == 10u) { g = y - z; }
    return g;
}

// 3-D gradient noise, about -1..1 (mirror of vfx::noise3).
fn noise3(x: f32, y: f32, z: f32, seed: U64) -> f32 {
    let fx0 = floor(x);
    let fy0 = floor(y);
    let fz0 = floor(z);
    let xf = x - fx0;
    let yf = y - fy0;
    let zf = z - fz0;
    let xi = i32(clamp(fx0, -2.0e9, 2.0e9));
    let yi = i32(clamp(fy0, -2.0e9, 2.0e9));
    let zi = i32(clamp(fz0, -2.0e9, 2.0e9));
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);
    let g000 = grad3(hash_u64(u64_of_i32(xi), u64_of_i32(yi), u64_of_i32(zi), seed), xf, yf, zf);
    let g100 = grad3(hash_u64(u64_of_i32(xi + 1), u64_of_i32(yi), u64_of_i32(zi), seed), xf - 1.0, yf, zf);
    let g010 = grad3(hash_u64(u64_of_i32(xi), u64_of_i32(yi + 1), u64_of_i32(zi), seed), xf, yf - 1.0, zf);
    let g110 = grad3(hash_u64(u64_of_i32(xi + 1), u64_of_i32(yi + 1), u64_of_i32(zi), seed), xf - 1.0, yf - 1.0, zf);
    let g001 = grad3(hash_u64(u64_of_i32(xi), u64_of_i32(yi), u64_of_i32(zi + 1), seed), xf, yf, zf - 1.0);
    let g101 = grad3(hash_u64(u64_of_i32(xi + 1), u64_of_i32(yi), u64_of_i32(zi + 1), seed), xf - 1.0, yf, zf - 1.0);
    let g011 = grad3(hash_u64(u64_of_i32(xi), u64_of_i32(yi + 1), u64_of_i32(zi + 1), seed), xf, yf - 1.0, zf - 1.0);
    let g111 = grad3(hash_u64(u64_of_i32(xi + 1), u64_of_i32(yi + 1), u64_of_i32(zi + 1), seed), xf - 1.0, yf - 1.0, zf - 1.0);
    let x00 = g000 + (g100 - g000) * u;
    let x10 = g010 + (g110 - g010) * u;
    let x01 = g001 + (g101 - g001) * u;
    let x11 = g011 + (g111 - g011) * u;
    let y0 = x00 + (x10 - x00) * v;
    let y1 = x01 + (x11 - x01) * v;
    return (y0 + (y1 - y0) * w) * 0.9;
}

// fBm with fractional octaves (mirror of vfx::fbm; octaves clamped to 1..10).
fn fbm(x: f32, y: f32, z: f32, octaves: f32, seed: U64) -> f32 {
    let n = clamp(octaves, 1.0, 10.0);
    var sum = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    var norm = 0.0;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        let fi = f32(i);
        if (fi >= n) {
            break;
        }
        let k = min(n - fi, 1.0);
        sum = sum + noise3(x * freq, y * freq, z * freq + fi * 17.13, add64(seed, U64(0u, i))) * amp * k;
        norm = norm + amp * k;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum / max(norm, 1e-6);
}

// Linear -> sRGB, as effects::enc (negatives clamp to 0).
fn enc1(c: vec3<f32>) -> vec3<f32> {
    let v = max(c, vec3<f32>(0.0));
    let lo = v * 12.92;
    let hi = 1.055 * pow(v, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, v <= vec3<f32>(0.0031308));
}

fn unpremul(p: vec4<f32>) -> vec3<f32> {
    if (p.a <= 1e-6) {
        return vec3<f32>(0.0);
    }
    return p.rgb / p.a;
}

// Truncated Gaussian along `dir`, zero outside the image. Weights are normalised by the full
// kernel sum (out-of-image taps count as zero), as the CPU's zero-padded box blur does.
fn blur_axis(pos: vec2<i32>, dir: vec2<i32>, dims: vec2<i32>, sigma: f32, alpha: bool) -> f32 {
    let sg = max(sigma, 1e-3);
    var r = 0.0;
    if (sigma > 0.3) {
        r = min(ceil(3.0 * sg), MAX_RADIUS);
    }
    let ri = i32(r);
    let inv = 1.0 / (2.0 * sg * sg);
    var acc = 0.0;
    var wsum = 0.0;
    for (var t: i32 = -ri; t <= ri; t = t + 1) {
        let ft = f32(t);
        let w = exp(-ft * ft * inv);
        let q = pos + dir * t;
        var s = 0.0;
        if (q.x >= 0 && q.y >= 0 && q.x < dims.x && q.y < dims.y) {
            let px = textureLoad(input0, q, 0);
            if (alpha) {
                s = px.a;
            } else {
                s = px.r;
            }
        }
        acc = acc + w * s;
        wsum = wsum + w;
    }
    return acc / max(wsum, 1e-12);
}

@compute @workgroup_size(8, 8, 1)
fn cs_roughen_blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let dims = vec2<i32>(textureDimensions(input0));
    let sigma = min(pf(0u), f32(max(dims.x, 8)) * 2.0);
    let texel = vec2<i32>(id.xy);
    let v = blur_axis(texel, vec2<i32>(1, 0), dims, sigma, true);
    textureStore(output, texel, vec4<f32>(v, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_roughen_blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let dims = vec2<i32>(textureDimensions(input0));
    let sigma = min(pf(0u), f32(max(dims.y, 8)) * 2.0);
    let texel = vec2<i32>(id.xy);
    let v = blur_axis(texel, vec2<i32>(0, 1), dims, sigma, false);
    textureStore(output, texel, vec4<f32>(v, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_roughen_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let src = textureLoad(input0, texel, 0);
    if (src.a <= 1e-6) {
        textureStore(output, texel, src);
        return;
    }
    let d = textureLoad(input1, texel, 0).r;
    let kind = pword(14u);
    let sharp = pf(1u);
    let infl = pf(2u);
    let scale = pf(3u);
    let stretch = pf(4u);
    let off = vec2<f32>(pf(5u), pf(6u));
    let cplx = pf(7u);
    let evo = pf(8u);
    let seed = U64(pword(13u), pword(12u));
    let edge_color = vec3<f32>(pf(16u), pf(17u), pf(18u));

    let px = vec2<f32>(f32(texel.x), f32(texel.y));
    let cu = (px.x + off.x) / (scale * stretch);
    let cv = (px.y + off.y) * stretch / scale;
    var n = fbm(cu, cv, evo, cplx, seed);
    if (kind == 3u) {
        n = 1.0 - 2.0 * abs(n);
    }
    var k = clamp((d - 0.5 + n * infl * 0.5) * (1.0 + sharp * 4.0) + 0.5, 0.0, 1.0);
    if (kind == 2u) {
        k = select(0.0, 1.0, k > 0.5);
    }
    if (kind >= 4u && kind <= 7u) {
        let hole = fbm(cu * 2.3 + 11.0, cv * 2.3, evo, cplx, add64(seed, U64(0u, 1u)));
        if (d < 0.97 && hole > 0.25) {
            k = k * (1.0 - smoothstep(0.25, 0.4, hole) * (1.0 - d));
        }
    }

    var rgb = src.rgb;
    let colored = kind == 1u || kind == 5u || kind == 7u;
    if (colored && d < 0.97) {
        let c = unpremul(src);
        let t = min((1.0 - d) * 1.5, 1.0);
        rgb = (c + (edge_color - c) * t) * src.a;
    }
    if (kind == 6u) {
        let e = enc1(unpremul(src));
        var l = 0.0;
        if (0.2126 * e.r + 0.7152 * e.g + 0.0722 * e.b > 0.5) {
            l = 1.0;
        }
        rgb = vec3<f32>(l) * src.a;
    }

    textureStore(output, texel, vec4<f32>(rgb * k, src.a * k));
}
