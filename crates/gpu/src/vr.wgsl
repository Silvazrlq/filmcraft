// FilmCraft GPU immersive-video (VR) kernels: the equirectangular blur, sharpen and denoise of
// `filmcraft_render::vfx::immersive`. The math mirrors `row_radius`, `blur_row_wrap` and the
// `vr_sharpen` / `vr_denoise` branches of `apply` operation by operation; the CPU code is the spec.
// Each view (the whole frame or one eye) is a full sphere: x wraps, and the view is addressed by
// (view_x, view_y) inside the full-size working textures.
//
// The vertical Gaussian of `vr_blur_image` is not here: the caller runs `gaussian` for it.

// One box pass along each row. Pass k reads the output of pass k-1 (ping-pong), so the caller
// dispatches three times and skips all of them when sigma <= 0.3 (`vr_blur_image` returns early).
struct VrParams {
    view_x: u32,
    view_y: u32,
    vw: u32,
    vh: u32,
    sigma: f32,
    pass_index: u32,  // informative only
    _pad0: u32,
    _pad1: u32,
};

// Sharpen and denoise: `amount` is already divided by 100; `threshold` is the denoise threshold.
struct CombineParams {
    view_x: u32,
    view_y: u32,
    vw: u32,
    vh: u32,
    amount: f32,
    show: u32,
    threshold: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> blur_p: VrParams;
@group(0) @binding(1) var blur_src: texture_2d<f32>;
@group(0) @binding(2) var blur_dst: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3) var<uniform> comb_p: CombineParams;
// The original view (`orig`) and the blurred copy of it (`blur`).
@group(0) @binding(4) var orig: texture_2d<f32>;
@group(0) @binding(5) var blurred: texture_2d<f32>;
@group(0) @binding(6) var comb_dst: texture_storage_2d<rgba32float, write>;

// No PI constant in WGSL; this is pi rounded to f32.
const PI: f32 = 3.1415927;

// `row_radius`: the box radius of row y. Rounding is floor(x + 0.5) (as in Rust). The Rust
// `min(cap).max(0)` order is kept as `min(max(f, 1), cap)`: `clamp` would be wrong when cap < 1.
fn row_radius(sigma: f32, w: u32, h: u32, y: u32) -> u32 {
    let lat = (0.5 - (f32(y) + 0.5) / f32(h)) * PI;
    let s = min(sigma / max(cos(lat), 1e-3), f32(w) / 4.0);
    if s <= 0.3 {
        return 0u;
    }
    let f = floor((sqrt(12.0 * s * s / 3.0 + 1.0) - 1.0) / 2.0 + 0.5);
    let cap = f32((w - 1u) / 2u);
    return u32(min(max(f, 1.0), cap));
}

// `rem_euclid` for a positive modulus.
fn wrap_x(a: i32, n: i32) -> i32 {
    return ((a % n) + n) % n;
}

@compute @workgroup_size(64)
fn vr_blur_row(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.x;
    let w = blur_p.vw;
    let h = blur_p.vh;
    if y >= h || w == 0u {
        return;
    }
    // vr_blur_image's early return: no blur at all for sigma <= 0.3 (the radius alone would not
    // catch it near the poles, where sigma / cos is larger).
    var r = 0u;
    if blur_p.sigma > 0.3 {
        r = row_radius(blur_p.sigma, w, h, y);
    }
    let row_y = i32(blur_p.view_y + y);
    let x0 = i32(blur_p.view_x);
    let ww = i32(w);

    if r == 0u {
        // a copy keeps the ping-pong chain exact
        for (var x = 0; x < ww; x = x + 1) {
            let c = vec2<i32>(x0 + x, row_y);
            textureStore(blur_dst, c, textureLoad(blur_src, c, 0));
        }
        return;
    }

    let ri = i32(r);
    let inv = 1.0 / f32(2 * ri + 1);
    // initial window: j = -r..=r with wrap-around
    var acc = vec4<f32>(0.0);
    for (var j = -ri; j <= ri; j = j + 1) {
        let i = wrap_x(j, ww);
        acc = acc + textureLoad(blur_src, vec2<i32>(x0 + i, row_y), 0);
    }
    for (var x = 0; x < ww; x = x + 1) {
        textureStore(blur_dst, vec2<i32>(x0 + x, row_y), acc * inv);
        let out_i = wrap_x(x - ri, ww);
        let in_i = wrap_x(x + ri + 1, ww);
        let v_in = textureLoad(blur_src, vec2<i32>(x0 + in_i, row_y), 0);
        let v_out = textureLoad(blur_src, vec2<i32>(x0 + out_i, row_y), 0);
        acc = acc + (v_in - v_out);
    }
}

@compute @workgroup_size(8, 8)
fn vr_sharpen_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= comb_p.vw || gid.y >= comb_p.vh {
        return;
    }
    let c = vec2<i32>(i32(comb_p.view_x + gid.x), i32(comb_p.view_y + gid.y));
    let p = textureLoad(orig, c, 0);
    let q = textureLoad(blurred, c, 0);
    // alpha is left as it is
    let rgb = max(p.rgb + (p.rgb - q.rgb) * comb_p.amount, vec3<f32>(0.0));
    textureStore(comb_dst, c, vec4<f32>(rgb, p.a));
}

@compute @workgroup_size(8, 8)
fn vr_denoise_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= comb_p.vw || gid.y >= comb_p.vh {
        return;
    }
    let c = vec2<i32>(i32(comb_p.view_x + gid.x), i32(comb_p.view_y + gid.y));
    let p = textureLoad(orig, c, 0);
    let q = textureLoad(blurred, c, 0);
    // weight falls off with the largest colour difference (RGB only)
    let d = max(max(abs(p.r - q.r), abs(p.g - q.g)), abs(p.b - q.b));
    let thr = comb_p.threshold;
    let k = exp(-(d * d) / (thr * thr));
    let o = p + (q - p) * k;
    if comb_p.show == 1u {
        // the noise view: the output alpha is the original alpha (p.a is not overwritten)
        let rgb = max((p.rgb - o.rgb) * 4.0 + vec3<f32>(0.18), vec3<f32>(0.0)) * p.a;
        textureStore(comb_dst, c, vec4<f32>(rgb, p.a));
    } else {
        textureStore(comb_dst, c, o);
    }
}
