use std::sync::Arc;

use crate::Device;

/// Accumulates GPU work into a single command encoder, submitted once.
///
/// This is the project's core performance primitive: a whole forward step records
/// many compute dispatches into one `Recorder` and `submit()`s once, minimizing
/// per-dispatch CPU overhead (the measured bottleneck for WebGPU model inference).
pub struct Recorder {
    pub(crate) dev: Arc<Device>,
    pub(crate) encoder: wgpu::CommandEncoder,
}

impl Recorder {
    pub fn new(dev: &Arc<Device>) -> Self {
        let encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eyebrowse-recorder"),
            });
        Recorder {
            dev: dev.clone(),
            encoder,
        }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }

    /// Finalize and queue all recorded work.
    pub fn submit(self) {
        self.dev.queue.submit(Some(self.encoder.finish()));
    }
}
