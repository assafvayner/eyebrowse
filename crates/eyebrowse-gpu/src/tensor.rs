use std::sync::Arc;

use eyebrowse_core::{DType, EyebrowseError, Result};

use crate::Device;

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

    /// Allocate an f32 tensor and upload `data` via `queue.write_buffer` (the WASM-preferred path).
    pub fn from_f32(dev: &Arc<Device>, shape: &[usize], data: &[f32]) -> Tensor {
        let t = Tensor::empty(dev, shape, DType::F32);
        assert_eq!(data.len(), t.numel(), "from_f32: data len != shape numel");
        dev.queue.write_buffer(&t.buffer, 0, bytemuck::cast_slice(data));
        t
    }

    /// Allocate a u32 tensor (used for packed f16 weights and integer ids) and upload `data`.
    pub fn from_u32(dev: &Arc<Device>, shape: &[usize], data: &[u32]) -> Tensor {
        let t = Tensor::empty(dev, shape, DType::U32);
        assert_eq!(data.len(), t.numel(), "from_u32: data len != shape numel");
        dev.queue.write_buffer(&t.buffer, 0, bytemuck::cast_slice(data));
        t
    }

    /// Read the buffer back to the host as f32. Async: on native it drives the queue to
    /// completion; on wasm the `.await` yields to the browser which resolves the mapping.
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        self.dev.queue.submit(Some(enc.finish()));

        let (tx, rx) = flume::bounded(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
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
