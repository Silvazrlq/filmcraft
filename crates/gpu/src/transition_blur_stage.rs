//! GPU-only Gaussian transition stages for typed blur operations.

use filmcraft_render::transitions::gpu_plan::{BlurKind, BlurOp};
use wgpu::util::DeviceExt;

use crate::transition_catalog::TransitionCatalogError;

type Target = (wgpu::Texture, wgpu::TextureView, (u32, u32));

/// Ceiling on one transition's temporaries, whatever the device allows: a guard against an accidental
/// multi-gigabyte allocation, not a resolution limit (8K needs about 3 GB for the blur, 1 GB for light).
const TEMP_BUDGET_CEILING_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Budget for the temporary targets of one blur: what the device's limits say a buffer may hold
/// (the adapter's `max_buffer_size` when the device is created from `adapter.limits()`), not a
/// fixed guess. A size above it is refused with an error.
fn temp_budget_bytes(limits: &wgpu::Limits) -> u64 {
    limits.max_buffer_size.min(TEMP_BUDGET_CEILING_BYTES)
}

pub(crate) struct TransitionBlurStage {
    box_pipeline: wgpu::ComputePipeline,
    mix_pipeline: wgpu::ComputePipeline,
    whip_pipeline: wgpu::ComputePipeline,
    finish_pipeline: wgpu::RenderPipeline,
    targets: [Option<Target>; 6],
}

impl TransitionBlurStage {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transition-blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transition_blur.wgsl").into()),
        });
        let box_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transition-blur-box"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_box"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transition-blur-mix"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_mix"),
            compilation_options: Default::default(),
            cache: None,
        });
        let whip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transition-whip-prepare"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_whip_prepare"),
            compilation_options: Default::default(),
            cache: None,
        });
        let finish_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transition-blur-finish"),
            layout: None,
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_spherical"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { box_pipeline, mix_pipeline, whip_pipeline, finish_pipeline, targets: [None, None, None, None, None, None] }
    }

    fn target(&mut self, device: &wgpu::Device, index: usize, size: (u32, u32)) -> wgpu::TextureView {
        if let Some(target) = self.targets[index].as_ref().filter(|target| target.2 == size) {
            return target.1.clone();
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transition-blur-temp"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.targets[index] = Some((texture, view.clone(), size));
        view
    }

    fn params(size: (u32, u32), op: BlurOp, radius: u32, axis: u32) -> [u32; 32] {
        let mut words = [0; 32];
        words[0] = size.0;
        words[1] = size.1;
        words[2] = radius;
        words[3] = axis;
        words[4] = op.kind as u32;
        words[8] = op.progress.to_bits();
        words[9] = op.blur_px.to_bits();
        words
    }

    fn uniform(device: &wgpu::Device, words: &[u32; 32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("transition-blur-params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn box_pass(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output_slot: usize,
        radius: u32,
        axis: u32,
        size: (u32, u32),
        op: BlurOp,
    ) -> wgpu::TextureView {
        let output = self.target(device, output_slot, size);
        let params = Self::uniform(device, &Self::params(size, op, radius, axis));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition-blur-box"),
            layout: &self.box_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output) },
            ],
        });
        let lines = if axis == 0 { size.1 } else { size.0 };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("transition-blur-box"), timestamp_writes: None });
        pass.set_pipeline(&self.box_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(lines.div_ceil(64), 1, 1);
        drop(pass);
        output
    }

    fn mix_pass(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::TextureView,
        b: &wgpu::TextureView,
        output_slot: usize,
        size: (u32, u32),
        op: BlurOp,
    ) -> wgpu::TextureView {
        let output = self.target(device, output_slot, size);
        let params = Self::uniform(device, &Self::params(size, op, 0, 0));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition-blur-mix"),
            layout: &self.mix_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(b) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("transition-blur-mix"), timestamp_writes: None });
        pass.set_pipeline(&self.mix_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        drop(pass);
        output
    }

    fn whip_prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: [&wgpu::TextureView; 2],
        size: (u32, u32),
        op: BlurOp,
    ) -> wgpu::TextureView {
        let output = self.target(device, 4, size);
        let mut words = [0; 32];
        words[0] = size.0;
        words[1] = size.1;
        words[4] = op.kind as u32;
        words[5] = op.direction;
        words[8] = op.progress.to_bits();
        let params = Self::uniform(device, &words);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition-whip-prepare"),
            layout: &self.whip_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(inputs[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(inputs[1]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("transition-whip-prepare"), timestamp_writes: None });
        pass.set_pipeline(&self.whip_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        drop(pass);
        output
    }

    fn blur_chain(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mut input: wgpu::TextureView,
        scratch: [usize; 2],
        radii: &(Vec<u32>, Vec<u32>),
        size: (u32, u32),
        op: BlurOp,
    ) -> wgpu::TextureView {
        let mut next = 0;
        for (axis, values) in [(0, &radii.0), (1, &radii.1)] {
            for &radius in values {
                input = self.box_pass(device, encoder, &input, scratch[next], radius, axis, size, op);
                next ^= 1;
            }
        }
        input
    }

    fn finish(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, source: &wgpu::TextureView, canvas: &wgpu::TextureView) {
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition-blur-finish"),
            layout: &self.finish_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transition-blur-finish"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: canvas,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.finish_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        canvas: &wgpu::TextureView,
        inputs: [&wgpu::TextureView; 2],
        size: (u32, u32),
        op: BlurOp,
    ) -> Result<(), TransitionCatalogError> {
        let limits = device.limits();
        let budget = temp_budget_bytes(&limits);
        let bytes = u64::from(size.0).checked_mul(u64::from(size.1)).and_then(|n| n.checked_mul(16)).and_then(|n| n.checked_mul(6));
        if size.0 == 0
            || size.1 == 0
            || size.0 > limits.max_texture_dimension_2d
            || size.1 > limits.max_texture_dimension_2d
            || bytes.is_none_or(|n| n > budget)
        {
            return Err(TransitionCatalogError::InvalidSize);
        }
        if !op.gpu_ok() {
            return Err(TransitionCatalogError::InvalidPlan {
                id: match op.kind {
                    BlurKind::Dissolve => "blur_dissolve",
                    BlurKind::VrSpherical => "vr_spherical_blur",
                    BlurKind::WhipLegacy => "whip_legacy",
                }
                .into(),
            });
        }
        let p = op.progress;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("transition-blur") });
        let final_view = match op.kind {
            BlurKind::Dissolve => {
                let sigma = op.blur_px * 0.5;
                let smooth = |lo: f32, hi: f32, x: f32| {
                    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                let radii_a =
                    filmcraft_render::effects::gaussian_boxes(size.0 as usize, size.1 as usize, sigma * smooth(0.0, 0.6, p), sigma * smooth(0.0, 0.6, p));
                let radii_b = filmcraft_render::effects::gaussian_boxes(
                    size.0 as usize,
                    size.1 as usize,
                    sigma * (1.0 - smooth(0.4, 1.0, p)),
                    sigma * (1.0 - smooth(0.4, 1.0, p)),
                );
                let a = self.blur_chain(device, &mut encoder, inputs[0].clone(), [0, 1], &radii_a, size, op);
                let b = self.blur_chain(device, &mut encoder, inputs[1].clone(), [2, 3], &radii_b, size, op);
                self.mix_pass(device, &mut encoder, &a, &b, 4, size, op)
            }
            BlurKind::VrSpherical => {
                let mixed = self.mix_pass(device, &mut encoder, inputs[0], inputs[1], 4, size, op);
                let t = 4.0 * p * (1.0 - p);
                let env = t * t * (3.0 - 2.0 * t);
                let sigma = op.blur_px * 0.5 * env;
                let radii = filmcraft_render::effects::gaussian_boxes(size.0 as usize, size.1 as usize, sigma, sigma);
                self.blur_chain(device, &mut encoder, mixed, [0, 1], &radii, size, op)
            }
            BlurKind::WhipLegacy => {
                let pushed = self.whip_prepare(device, &mut encoder, inputs, size, op);
                let sigma = (1.0 - (2.0 * p - 1.0).abs()) * 40.0 * (size.0 as f32 / 1920.0);
                if sigma > 0.5 {
                    let radii = filmcraft_render::effects::gaussian_boxes(size.0 as usize, size.1 as usize, sigma, 0.0);
                    self.blur_chain(device, &mut encoder, pushed, [0, 1], &radii, size, op)
                } else {
                    pushed
                }
            }
        };
        self.finish(device, &mut encoder, &final_view, canvas);
        queue.submit([encoder.finish()]);
        Ok(())
    }
}
