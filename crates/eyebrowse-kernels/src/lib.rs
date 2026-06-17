//! Hand-written WGSL compute kernels and their Rust dispatch wrappers. Each kernel records
//! into a `Recorder` (caller submits) and is validated natively against a CPU reference.

mod matmul;

#[cfg(test)]
mod testutil;

pub use matmul::{matmul, matmul_f16w};
