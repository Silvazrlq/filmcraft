//! The immersive-video WGSL kernels (`vr.wgsl`) parse and validate. Parity with the CPU math is
//! is not checked here: see `TESTE-GPU-MULTIPASS-VR.md`.

#[test]
fn vr_wgsl_validates() {
    let module = wgpu::naga::front::wgsl::parse_str(include_str!("vr.wgsl")).expect("vr.wgsl parses");
    wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
        .validate(&module)
        .expect("vr.wgsl validates");
}
