use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eyebrowse_core::{EyebrowseError, Result};

/// A compiled compute pipeline plus the single bind-group layout it uses.
pub(crate) struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// Buffers at or below this size are recycled through the pool; larger ones (e.g. weight tensors)
/// are always freshly allocated and freed normally, so the pool never retains giant allocations.
const MAX_POOLED_BYTES: u64 = 64 * 1024 * 1024;

/// Owns the wgpu device + queue, a cache of compiled compute pipelines, and a pool of recyclable
/// GPU buffers.
///
/// Cloned cheaply as `Arc<Device>` everywhere; never duplicated.
pub struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub(crate) pipelines: Mutex<HashMap<String, Arc<CachedPipeline>>>,
    /// Free list of recyclable buffers, keyed by exact `(size, usage)`. Exact-size keying keeps
    /// `arrayLength()` correct in shaders that read a buffer's whole length.
    pub(crate) buffer_pool: Mutex<HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>>,
}

impl Device {
    /// Request a high-performance adapter and a device with the adapter's full limits.
    pub async fn new() -> Result<Arc<Device>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| EyebrowseError::DeviceUnavailable(format!("request_adapter: {e:?}")))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("eyebrowse"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| EyebrowseError::DeviceUnavailable(format!("request_device: {e:?}")))?;
        Ok(Arc::new(Device {
            device,
            queue,
            limits,
            pipelines: Mutex::new(HashMap::new()),
            buffer_pool: Mutex::new(HashMap::new()),
        }))
    }

    #[cfg(test)]
    fn pooled(&self, size: u64, usage: wgpu::BufferUsages) -> usize {
        self.buffer_pool
            .lock()
            .unwrap()
            .get(&(size, usage))
            .map_or(0, Vec::len)
    }

    /// Create a brand-new buffer of exactly `size` bytes with `usage` (never from the pool).
    pub(crate) fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Acquire a buffer of exactly `size` bytes with `usage`, reusing a recycled one when one of
    /// the matching `(size, usage)` is free. Contents are unspecified — the caller must fully write
    /// the buffer before reading it (every kernel output here does).
    ///
    /// Only buffers produced this way (`Tensor::empty`) draw from the pool: buffers initialized via
    /// `queue.write_buffer` must NOT, because multiple `write_buffer`s to one buffer within a single
    /// unsubmitted batch collapse to last-wins.
    pub(crate) fn acquire_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        if size <= MAX_POOLED_BYTES {
            if let Ok(mut pool) = self.buffer_pool.lock() {
                if let Some(buf) = pool.get_mut(&(size, usage)).and_then(Vec::pop) {
                    return buf;
                }
            }
        }
        self.create_buffer(size, usage)
    }

    /// Return a buffer to the pool for later reuse. Called from `Tensor`'s `Drop`. Buffers larger
    /// than [`MAX_POOLED_BYTES`] are dropped (freed) rather than retained.
    pub(crate) fn recycle_buffer(&self, buffer: wgpu::Buffer) {
        if buffer.size() > MAX_POOLED_BYTES {
            return;
        }
        let key = (buffer.size(), buffer.usage());
        if let Ok(mut pool) = self.buffer_pool.lock() {
            pool.entry(key).or_default().push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use eyebrowse_core::DType;

    #[test]
    fn empty_recycles_buffers_through_the_pool() {
        let dev = pollster::block_on(Device::new()).expect("device");
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let size = (64 * DType::F32.size()) as u64;

        assert_eq!(dev.pooled(size, usage), 0, "fresh device pool is empty");

        // Allocating then dropping returns one buffer to the pool.
        drop(Tensor::empty(&dev, &[64], DType::F32));
        assert_eq!(dev.pooled(size, usage), 1, "drop recycles");

        // The next same-size allocation reuses it (pool drains)...
        let t = Tensor::empty(&dev, &[64], DType::F32);
        assert_eq!(dev.pooled(size, usage), 0, "empty reuses a pooled buffer");
        // ...and dropping returns it again.
        drop(t);
        assert_eq!(dev.pooled(size, usage), 1, "recycled again");
    }

    #[test]
    fn host_initialized_tensors_do_not_draw_from_the_pool() {
        let dev = pollster::block_on(Device::new()).expect("device");
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let size = (4 * DType::F32.size()) as u64;

        // Seed the pool with one recyclable buffer of this size.
        drop(Tensor::empty(&dev, &[4], DType::F32));
        assert_eq!(dev.pooled(size, usage), 1);

        // `from_f32` must allocate fresh (never pull): the pooled buffer stays put.
        let _t = Tensor::from_f32(&dev, &[4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dev.pooled(size, usage), 1, "from_f32 does not drain the pool");
    }
}
