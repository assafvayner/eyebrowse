//! Per-architecture model loaders. Every architecture builds a shared [`LanguageModel`] (an
//! embedding + a stack of `Block`s + final norm + LM head); the per-arch modules differ only in
//! the block they construct and a few hooks. `load_model` selects the loader by the source's
//! architecture and returns the unified [`LanguageModel`], so callers stay architecture-agnostic.

mod decoder;
mod gemma4;
mod mistral;
mod model;
mod qwen3;
mod upload;

pub use model::{Block, LanguageModel};
pub use upload::{raw_to_f32, upload_f16, upload_f32};

use std::sync::Arc;

use eyebrowse_core::{EyebrowseError, Result};
use eyebrowse_gpu::Device;
use eyebrowse_load::WeightSource;

/// Build the model for the source's architecture (`Config.arch`), sized to `max_seq`.
pub fn load_model(
    dev: &Arc<Device>,
    src: &dyn WeightSource,
    max_seq: usize,
) -> Result<LanguageModel> {
    match src.config().arch.as_str() {
        "qwen3" => qwen3::load(dev, src, max_seq),
        "llama" | "mistral" => mistral::load(dev, src, max_seq),
        "gemma4" | "gemma4_text" => gemma4::load(dev, src, max_seq),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "architecture {other}"
        ))),
    }
}
