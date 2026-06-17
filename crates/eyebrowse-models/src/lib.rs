//! Per-architecture model loaders. Each builds a shared [`Decoder`] from a `WeightSource`;
//! `load_model` selects the loader by the source's architecture.

mod decoder;
mod mistral;
mod qwen3;
mod upload;

pub use decoder::{Decoder, DecoderOpts};
pub use upload::{raw_to_f32, upload_f16, upload_f32};

use std::sync::Arc;

use eyebrowse_core::{EyebrowseError, Result};
use eyebrowse_gpu::Device;
use eyebrowse_load::WeightSource;

/// Build the decoder for the source's architecture (`Config.arch`), sized to `max_seq`.
pub fn load_model(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<Decoder> {
    match src.config().arch.as_str() {
        "qwen3" => qwen3::load(dev, src, max_seq),
        "llama" | "mistral" => mistral::load(dev, src, max_seq),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "architecture {other}"
        ))),
    }
}
