//! Per-architecture model modules. Each assembles `eyebrowse-nn` primitives from a `WeightSource`
//! config. `qwen3` is the first; adding another decoder-only LLM is a sibling module.

mod qwen3;
mod upload;

pub use qwen3::Qwen3Model;
pub use upload::{raw_to_f32, upload_f16, upload_f32};
