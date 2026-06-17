//! Hand-written WGSL compute kernels and their Rust dispatch wrappers. Each kernel records
//! into a `Recorder` (caller submits) and is validated natively against a CPU reference.

mod attention;
mod matmul;
mod ops;
mod rmsnorm;
mod rope;
mod sampling;
mod softmax;

#[cfg(test)]
mod testutil;

pub use attention::{attn_decode, attn_prefill};
pub use matmul::{linear_f16w, matmul, matmul_f16w};
pub use ops::{embedding_f16, kv_write, swiglu};
pub use rmsnorm::rmsnorm;
pub use rope::rope;
pub use sampling::argmax;
pub use softmax::softmax;
