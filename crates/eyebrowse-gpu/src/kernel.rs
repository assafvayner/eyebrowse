use std::sync::Arc;

use crate::device::CachedPipeline;
use crate::{Device, Recorder, Tensor};

/// Get-or-compile a compute pipeline whose single bind group is `n_buffers` storage
/// buffers at bindings `0..n_buffers`, all declared `var<storage, read_write>` in WGSL.
///
/// Keyed by `key` (must be stable per (wgsl, entry, n_buffers)); cached on the device.
fn get_pipeline(
    dev: &Arc<Device>,
    key: &str,
    wgsl: &str,
    entry: &str,
    n_buffers: usize,
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
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_buffers)
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
    let dev = rec.dev.clone();
    let cached = get_pipeline(&dev, key, wgsl, entry, buffers.len());
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
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
