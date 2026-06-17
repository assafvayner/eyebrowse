use std::sync::Arc;

use eyebrowse_core::{DType, EyebrowseError, Result};

use crate::Device;

/// Pack f32 values into the kernels' f16 storage format: two f16 lanes per u32 word
/// (`u32[w] = bits(f16(x[2w])) | (bits(f16(x[2w+1])) << 16)`). Odd tail is zero-padded.
pub fn pack_f16(x: &[f32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(x.len().div_ceil(2));
    let mut i = 0;
    while i < x.len() {
        let lo = half::f16::from_f32(x[i]).to_bits() as u32;
        let hi = if i + 1 < x.len() {
            half::f16::from_f32(x[i + 1]).to_bits() as u32
        } else {
            0
        };
        out.push(lo | (hi << 16));
        i += 2;
    }
    out
}

/// A typed handle over a GPU storage buffer. Owns no host-side copy of the data.
pub struct Tensor {
    pub(crate) dev: Arc<Device>,
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }

    /// Allocate an uninitialized storage buffer (usable as kernel input/output and copyable).
    pub fn empty(dev: &Arc<Device>, shape: &[usize], dtype: DType) -> Tensor {
        let numel: usize = shape.iter().product();
        let size = (numel * dtype.size()).max(dtype.size()) as u64;
        let buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Tensor {
            dev: dev.clone(),
            buffer,
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Allocate an f32 tensor and upload `data` via `queue.write_buffer`.
    pub fn from_f32(dev: &Arc<Device>, shape: &[usize], data: &[f32]) -> Tensor {
        let t = Tensor::empty(dev, shape, DType::F32);
        assert_eq!(data.len(), t.numel(), "from_f32: data len != shape numel");
        dev.queue
            .write_buffer(&t.buffer, 0, bytemuck::cast_slice(data));
        t
    }

    /// Allocate a u32 tensor (used for packed f16 weights and integer ids) and upload `data`.
    pub fn from_u32(dev: &Arc<Device>, shape: &[usize], data: &[u32]) -> Tensor {
        let t = Tensor::empty(dev, shape, DType::U32);
        assert_eq!(data.len(), t.numel(), "from_u32: data len != shape numel");
        dev.queue
            .write_buffer(&t.buffer, 0, bytemuck::cast_slice(data));
        t
    }

    /// Upload f32 `data` as a packed-u32 f16 tensor (the kernels' weight storage format):
    /// `u32[w] = bits(f16(data[2w])) | (bits(f16(data[2w+1])) << 16)`. Returns a `U32` tensor
    /// of `ceil(len/2)` words; `logical_shape` records the logical dims for the caller.
    pub fn from_f16_packed(dev: &Arc<Device>, logical_shape: &[usize], data: &[f32]) -> Tensor {
        let n: usize = logical_shape.iter().product();
        assert_eq!(data.len(), n, "from_f16_packed: data len != shape numel");
        let packed = pack_f16(data);
        Tensor::from_u32(dev, &[packed.len()], &packed)
    }

    /// Read the buffer back to the host as f32. Async: it drives the queue to completion and
    /// awaits the buffer mapping.
    pub async fn to_f32(&self) -> Result<Vec<f32>> {
        let bytes = self.read_bytes().await?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Read the raw bytes of the buffer back to the host.
    pub async fn read_bytes(&self) -> Result<Vec<u8>> {
        let size = self.byte_len() as u64;
        let staging = self.dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        self.dev.queue.submit(Some(enc.finish()));

        let (tx, rx) = flume::bounded(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.dev.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv_async()
            .await
            .map_err(|e| EyebrowseError::Other(format!("readback channel: {e}")))?
            .map_err(|e| EyebrowseError::Other(format!("buffer map: {e:?}")))?;
        let data = staging.slice(..).get_mapped_range();
        let out = data.to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}
