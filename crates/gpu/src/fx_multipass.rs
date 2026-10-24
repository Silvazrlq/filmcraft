//! Runs a `filmcraft_render::gpufx_multipass::MultipassPlan` on the device.
//!
//! Kernel contract (see `fx_multipass.wgsl`): bindings 0 and 1 are the plan inputs (unfilterable
//! Rgba32Float, read with `textureLoad`) and binding 2 is the pass output (write-only Rgba32Float
//! storage). Every intermediate slot owns one texture, allocated on first use and kept while the
//! size stays the same, so a plan of N passes allocates at most its distinct slots. No resolution
//! cap: the device decides whether the slots fit, and an allocation failure comes back as
//! `MultipassGpuError::OutOfMemory`. There is no CPU fallback. Errors are captured with error scopes
//! so the device's uncaptured-error handler never panics.

use filmcraft_render::gpufx_multipass::{MAX_SLOTS, MultipassError, MultipassPlan, Slot};

/// Workgroup edge used by every kernel in `fx_multipass.wgsl`.
const WORKGROUP: u32 = 8;

/// Why a multi-pass route was refused or failed. No variant means a silent CPU fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultipassGpuError {
    /// The plan is structurally invalid or its byte count overflows.
    Plan(MultipassError),
    /// The plan names a kernel the runner was not built with.
    UnknownKernel(&'static str),
    /// Zero-sized frame.
    SizeMismatch,
    /// The device ran out of memory allocating a slot or bind group.
    OutOfMemory,
    /// The device rejected a resource or a pass.
    Validation(String),
    /// The error scope did not resolve synchronously, so the outcome is unknown.
    ScopePending,
}

struct Scratch {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// Compute pipelines for one kernel module plus the scratch slots they write.
pub(crate) struct MultipassRunner {
    layout: wgpu::BindGroupLayout,
    kernels: Vec<(&'static str, wgpu::ComputePipeline)>,
    slots: Vec<Option<Scratch>>,
}

fn sampled_input(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn resolve<T>(poll: Option<Option<T>>) -> Result<(), MultipassGpuError>
where
    T: std::fmt::Display,
{
    match poll {
        Some(None) => Ok(()),
        Some(Some(error)) => Err(MultipassGpuError::Validation(error.to_string())),
        None => Err(MultipassGpuError::ScopePending),
    }
}

impl MultipassRunner {
    /// Builds one compute pipeline per entry point in `kernels`, all from the same WGSL module.
    pub(crate) fn new(device: &wgpu::Device, wgsl: &str, kernels: &[&'static str]) -> Result<Self, MultipassGpuError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fx-multipass"), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx-multipass-bind"),
            entries: &[
                sampled_input(0),
                sampled_input(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx-multipass"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let kernels = kernels
            .iter()
            .map(|&name| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (name, pipeline)
            })
            .collect();
        resolve(super::poll_now(scope.pop()))?;
        Ok(Self { layout, kernels, slots: (0..MAX_SLOTS).map(|_| None).collect() })
    }

    /// Records every pass of `plan` into `encoder` and returns the view of the result slot.
    /// The caller submits the encoder. `source` must be an Rgba32Float view of `size`.
    /// An `Err` can leave passes already recorded: the caller must discard that encoder, never submit it.
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plan: &MultipassPlan,
        source: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<wgpu::TextureView, MultipassGpuError> {
        if size.0 == 0 || size.1 == 0 {
            return Err(MultipassGpuError::SizeMismatch);
        }
        let limits = device.limits();
        let max_dim = limits.max_texture_dimension_2d;
        let max_groups = limits.max_compute_workgroups_per_dimension;
        if size.0 > max_dim || size.1 > max_dim || size.0.div_ceil(WORKGROUP) > max_groups || size.1.div_ceil(WORKGROUP) > max_groups {
            return Err(MultipassGpuError::SizeMismatch);
        }
        plan.intermediate_bytes(size.0, size.1).map_err(MultipassGpuError::Plan)?;
        // Scopes are popped on every path, so a failure never leaves the device's scope stack open.
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let recorded = self.record(device, encoder, plan, source, size);
        let validated = resolve(super::poll_now(validation.pop()));
        let allocated = match super::poll_now(oom.pop()) {
            Some(None) => Ok(()),
            Some(Some(_)) => Err(MultipassGpuError::OutOfMemory),
            None => Err(MultipassGpuError::ScopePending),
        };
        let result = recorded?;
        validated?;
        allocated?;
        Ok(result)
    }

    fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plan: &MultipassPlan,
        source: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<wgpu::TextureView, MultipassGpuError> {
        for pass in &plan.passes {
            let index = usize::from(pass.output.0);
            let fresh = !matches!(self.slots.get(index), Some(Some(s)) if s.size == size);
            if fresh {
                let slot = self.slots.get_mut(index).ok_or(MultipassGpuError::Plan(MultipassError::SlotOutOfRange { slot: pass.output.0 }))?;
                *slot = Some(allocate(device, size));
            }
        }
        let groups = (size.0.div_ceil(WORKGROUP), size.1.div_ceil(WORKGROUP));
        for pass in &plan.passes {
            let pipeline = self.kernels.iter().find(|(name, _)| *name == pass.kernel).map(|(_, p)| p).ok_or(MultipassGpuError::UnknownKernel(pass.kernel))?;
            let input = |i: usize| -> Result<&wgpu::TextureView, MultipassGpuError> {
                // A kernel that reads one input gets it on both bindings; validation guarantees at least one.
                let slot = pass.inputs.get(i).or(pass.inputs.first()).copied().unwrap_or(Slot::SOURCE);
                self.view(slot, source)
            };
            let output = self.view(pass.output, source)?;
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx-multipass-pass"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input(0)?) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input(1)?) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output) },
                ],
            });
            let mut pass_encoder = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(pass.kernel), timestamp_writes: None });
            pass_encoder.set_pipeline(pipeline);
            pass_encoder.set_bind_group(0, &bind, &[]);
            pass_encoder.dispatch_workgroups(groups.0, groups.1, 1);
        }
        self.view(plan.result, source).cloned()
    }

    /// The view a slot currently holds. Validation guarantees a read follows a write; a missing
    /// view here is reported, never unwrapped.
    fn view<'a>(&'a self, slot: Slot, source: &'a wgpu::TextureView) -> Result<&'a wgpu::TextureView, MultipassGpuError> {
        if slot == Slot::SOURCE {
            return Ok(source);
        }
        self.slots
            .get(usize::from(slot.0))
            .and_then(Option::as_ref)
            .map(|s| &s.view)
            .ok_or(MultipassGpuError::Plan(MultipassError::ReadBeforeWrite { pass: 0, slot: slot.0 }))
    }
}

fn allocate(device: &wgpu::Device, size: (u32, u32)) -> Scratch {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fx-multipass-slot"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Scratch { _texture: texture, view, size }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADER: &str = include_str!("fx_multipass.wgsl");

    #[test]
    fn kernel_contract_shader_validates_with_naga() {
        let module = wgpu::naga::front::wgsl::parse_str(SHADER).expect("fx_multipass.wgsl must parse");
        wgpu::naga::valid::Validator::new(Default::default(), Default::default()).validate(&module).expect("fx_multipass.wgsl must validate");
        let entries: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert!(entries.contains(&"cs_copy") && entries.contains(&"cs_mix"), "{entries:?}");
    }

    fn adapter_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    #[test]
    fn refuses_a_bad_plan_before_recording_any_pass() {
        let Some((device, _)) = adapter_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut runner = MultipassRunner::new(&device, SHADER, &["cs_copy"]).expect("runner");
        let bad = MultipassPlan { passes: vec![], result: Slot(1) };
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = source.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        assert_eq!(runner.encode(&device, &mut encoder, &bad, &view, (4, 4)).err(), Some(MultipassGpuError::Plan(MultipassError::EmptyPlan)));
        let unknown =
            MultipassPlan { passes: vec![filmcraft_render::gpufx_multipass::Pass { kernel: "nope", inputs: vec![Slot(0)], output: Slot(1) }], result: Slot(1) };
        assert_eq!(runner.encode(&device, &mut encoder, &unknown, &view, (4, 4)).err(), Some(MultipassGpuError::UnknownKernel("nope")));
        assert_eq!(runner.encode(&device, &mut encoder, &unknown, &view, (0, 4)).err(), Some(MultipassGpuError::SizeMismatch));
    }
}
