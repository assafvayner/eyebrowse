//! Shared CPU references and helpers for the nn crate's tests.

pub use eyebrowse_gpu::pack_f16;
use eyebrowse_gpu::Tensor;

/// Read a GPU tensor back to host f32.
pub fn read(t: &Tensor) -> Vec<f32> {
    pollster::block_on(t.to_f32()).expect("readback")
}

/// Round each value through f16 (kernels read f16 weights, so CPU references must match).
pub fn round_f16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect()
}

/// Relative L2 error `||a - b|| / ||b||`.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num / den.max(1e-12)).sqrt() as f32
}

/// Deterministic pseudo-random sequence in roughly [-1, 1).
pub fn prng(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        out.push((u * 2.0 - 1.0) as f32);
    }
    out
}

/// CPU reference for a row-major linear: `y[i,o] = sum_k x[i,k] * w[o,k]`.
pub fn cpu_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * out_f];
    for i in 0..m {
        for o in 0..out_f {
            let mut acc = 0.0f32;
            for k in 0..in_f {
                acc += x[i * in_f + k] * w[o * in_f + k];
            }
            y[i * out_f + o] = acc;
        }
    }
    y
}
