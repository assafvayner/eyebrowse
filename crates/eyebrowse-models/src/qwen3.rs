//! Qwen3 dense loader: a standard decoder with per-head QK-RMSNorm.

use std::sync::Arc;

use eyebrowse_core::Result;
use eyebrowse_gpu::Device;
use eyebrowse_load::WeightSource;

use crate::decoder::{self, DecoderOpts};
use crate::model::LanguageModel;

pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<LanguageModel> {
    decoder::load(dev, src, max_seq, DecoderOpts { has_qk_norm: true })
}
