//! Test-only helpers: a shared device and CPU reference implementations the GPU kernels
//! are validated against (relative L2 error).

#![cfg(test)]

use std::sync::Arc;

use eyebrowse_gpu::Device;

/// A WebGPU device for tests (native Metal on this machine).
pub fn test_device() -> Arc<Device> {
    pollster::block_on(Device::new()).expect("device")
}

/// Relative L2 error ||a-b|| / ||b||, the standard kernel-correctness metric.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "rel_l2 length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num.sqrt() / (den.sqrt() + 1e-12)) as f32
}

/// CPU reference for row-major `C[m,n] = A[m,k] * B[k,n]`.
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Reduce an f32 slice to a packed-u32 f16 vector matching the kernels' weight layout:
/// `u32[w] = bits(f16(x[2w])) | (bits(f16(x[2w+1])) << 16)`. Odd tail is zero-padded.
pub fn pack_f16(x: &[f32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(x.len().div_ceil(2));
    let mut i = 0;
    while i < x.len() {
        let lo = half::f16::from_f32(x[i]).to_bits() as u32;
        let hi = if i + 1 < x.len() {
            half::f16::from_f32(x[i + 1]).to_bits() as u32
        } else {
            0
        };
        out.push(lo | (hi << 16));
        i += 2;
    }
    out
}

/// Round an f32 slice through f16 (to model the precision of f16-weight kernels on the CPU side).
pub fn round_f16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect()
}
