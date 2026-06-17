use std::sync::Arc;

use crate::device::CachedPipeline;
use crate::{Device, Recorder, Tensor};

/// Get-or-compile a compute pipeline whose single bind group is `n_storage` storage buffers
/// (`var<storage, read_write>`) at bindings `0..n_storage`, followed by `n_uniform` uniform
/// buffers (`var<uniform>`) at bindings `n_storage..n_storage+n_uniform`.
///
/// Uniform bindings exist so a kernel can read scalars (e.g. matrix dims) as *uniform* values:
/// WGSL forbids `workgroupBarrier()` in control flow that depends on a non-uniform value, and
/// reads from a `read_write` storage buffer are non-uniform. Tiled GEMM (barrier inside a loop
/// bounded by K) must therefore take its dims via a uniform buffer. Cached by `key`.
fn get_pipeline(
    dev: &Arc<Device>,
    key: &str,
    wgsl: &str,
    entry: &str,
    n_storage: usize,
    n_uniform: usize,
) -> Arc<CachedPipeline> {
    if let Some(p) = dev.pipelines.lock().unwrap().get(key) {
        return p.clone();
    }
    let module = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_storage)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    for u in 0..n_uniform {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (n_storage + u) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let layout = dev
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(key),
            entries: &entries,
        });
    let pipeline_layout = dev
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(key),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let cached = Arc::new(CachedPipeline { pipeline, layout });
    dev.pipelines
        .lock()
        .unwrap()
        .insert(key.to_string(), cached.clone());
    cached
}

/// Record one compute dispatch of `wgsl`'s `entry` over `buffers` (bound at 0..N), with the
/// given workgroup counts. Pipeline compiled once per `key` and cached.
pub fn dispatch(
    rec: &mut Recorder,
    key: &str,
    wgsl: &str,
    entry: &str,
    buffers: &[&wgpu::Buffer],
    workgroups: [u32; 3],
) {
    dispatch_with_uniform(rec, key, wgsl, entry, buffers, &[], workgroups);
}

/// Like [`dispatch`], but the buffers in `uniform` are bound as `var<uniform>` at the bindings
/// immediately after the `storage` buffers. Use for kernels that must read scalars (dims) as
/// uniform values to keep `workgroupBarrier()` in uniform control flow.
pub fn dispatch_with_uniform(
    rec: &mut Recorder,
    key: &str,
    wgsl: &str,
    entry: &str,
    storage: &[&wgpu::Buffer],
    uniform: &[&wgpu::Buffer],
    workgroups: [u32; 3],
) {
    let dev = rec.dev.clone();
    let cached = get_pipeline(&dev, key, wgsl, entry, storage.len(), uniform.len());
    let entries: Vec<wgpu::BindGroupEntry> = storage
        .iter()
        .chain(uniform.iter())
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(key),
        layout: &cached.layout,
        entries: &entries,
    });
    let mut pass = rec
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(key),
            timestamp_writes: None,
        });
    pass.set_pipeline(&cached.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
}

/// Create a small UNIFORM buffer initialized from `data` (padded to 16 bytes for `vec4`-style
/// uniform reads). For passing scalar dims to a kernel as uniform values.
pub fn uniform_u32(dev: &Arc<Device>, data: &[u32]) -> wgpu::Buffer {
    let size = ((data.len() * 4).max(16) as u64 + 15) & !15;
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

/// Copy `n` 4-byte elements from `src` (element offset `src_off`) into `dst` (element offset
/// `dst_off`) via a buffer-to-buffer copy — no shader dispatch. Used e.g. to slice the last
/// hidden row before the LM head.
pub fn copy_range(rec: &mut Recorder, src: &Tensor, dst: &Tensor, src_off: usize, dst_off: usize, n: usize) {
    rec.encoder.copy_buffer_to_buffer(
        &src.buffer,
        (src_off * 4) as u64,
        &dst.buffer,
        (dst_off * 4) as u64,
        (n * 4) as u64,
    );
}

const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] + b[i];
}
"#;

/// Elementwise `out = a + b`. Records into `rec`; caller submits.
pub fn add(rec: &mut Recorder, a: &Tensor, b: &Tensor, out: &Tensor) {
    let n = out.numel() as u32;
    let groups = n.div_ceil(64);
    dispatch(
        rec,
        "add",
        ADD_WGSL,
        "main",
        &[&a.buffer, &b.buffer, &out.buffer],
        [groups, 1, 1],
    );
}
