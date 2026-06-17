//! Mistral / Llama-style loader: a standard decoder identical to Qwen3 minus QK-norm and biases.

use std::sync::Arc;

use eyebrowse_core::Result;
use eyebrowse_gpu::Device;
use eyebrowse_load::WeightSource;

use crate::decoder::{self, DecoderOpts};
use crate::model::LanguageModel;

pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<LanguageModel> {
    decoder::load(dev, src, max_seq, DecoderOpts { has_qk_norm: false })
}
