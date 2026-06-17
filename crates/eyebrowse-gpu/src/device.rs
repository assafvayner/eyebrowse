use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eyebrowse_core::{EyebrowseError, Result};

/// A compiled compute pipeline plus the single bind-group layout it uses.
pub(crate) struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// Owns the wgpu device + queue and a cache of compiled compute pipelines.
///
/// Cloned cheaply as `Arc<Device>` everywhere; never duplicated.
pub struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub(crate) pipelines: Mutex<HashMap<&'static str, Arc<CachedPipeline>>>,
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
        }))
    }
}
