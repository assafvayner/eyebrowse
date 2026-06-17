//! Per-architecture model loaders. Most architectures build a shared [`Decoder`]; Gemma 4 has a
//! dedicated [`Gemma4`] loader. `load_model` selects the loader by the source's architecture and
//! returns a [`Model`] enum that delegates the forward/cache methods to the concrete model.

mod decoder;
mod gemma4;
mod mistral;
mod qwen3;
mod upload;

pub use decoder::{Decoder, DecoderOpts};
pub use gemma4::Gemma4;
pub use upload::{raw_to_f32, upload_f16, upload_f32};

use std::sync::Arc;

use eyebrowse_core::{EyebrowseError, Result};
use eyebrowse_gpu::Device;
use eyebrowse_load::WeightSource;
use eyebrowse_nn::KvCache;

/// A loaded model of any supported architecture. Delegates the inference methods to the concrete
/// decoder so callers can stay architecture-agnostic.
pub enum Model {
    Decoder(Decoder),
    Gemma4(Gemma4),
}

impl Model {
    /// Build a KV cache sized for this model and the given max sequence length.
    pub fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        match self {
            Model::Decoder(m) => m.new_kv_cache(max_seq),
            Model::Gemma4(m) => m.new_kv_cache(max_seq),
        }
    }

    /// Prefill over `ids`, returning logits (`[vocab]`) for the last position.
    pub async fn forward_prefill(&self, ids: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        match self {
            Model::Decoder(m) => m.forward_prefill(ids, kv).await,
            Model::Gemma4(m) => m.forward_prefill(ids, kv).await,
        }
    }

    /// Decode one `token` at absolute position `pos`, returning logits (`[vocab]`).
    pub async fn forward_decode(
        &self,
        token: u32,
        pos: usize,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>> {
        match self {
            Model::Decoder(m) => m.forward_decode(token, pos, kv).await,
            Model::Gemma4(m) => m.forward_decode(token, pos, kv).await,
        }
    }
}

/// Build the model for the source's architecture (`Config.arch`), sized to `max_seq`.
pub fn load_model(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<Model> {
    match src.config().arch.as_str() {
        "qwen3" => Ok(Model::Decoder(qwen3::load(dev, src, max_seq)?)),
        "llama" | "mistral" => Ok(Model::Decoder(mistral::load(dev, src, max_seq)?)),
        "gemma4" | "gemma4_text" => Ok(Model::Gemma4(gemma4::load(dev, src, max_seq)?)),
        other => Err(EyebrowseError::UnsupportedConfig(format!(
            "architecture {other}"
        ))),
    }
}
