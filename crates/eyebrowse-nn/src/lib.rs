//! Composable transformer primitives over `eyebrowse-kernels`. Each layer records GPU work
//! into a caller-owned `Recorder` and returns freshly-allocated f32 activation tensors; the
//! caller submits. Shapes follow the Qwen3 conventions (GQA + QK-RMSNorm + SwiGLU + RoPE).

mod attention;
mod embedding;
mod kv_cache;
mod linear;
mod mlp;
mod rmsnorm;
mod rope;

pub use attention::Attention;
pub use embedding::Embedding;
pub use kv_cache::KvCache;
pub use linear::Linear;
pub use mlp::Mlp;
pub use rmsnorm::RmsNorm;
pub use rope::Rope;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;
