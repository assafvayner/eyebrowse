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

/// CPU reference for a Linear layer with HF weight layout `w[out, in]` (no bias):
/// `y[m, o] = sum_k x[m, k] * w[o, k]`.
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

/// CPU reference for RMSNorm over rows of `[rows, dim]`.
pub fn cpu_rmsnorm(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let base = r * dim;
        let mut ss = 0.0f32;
        for i in 0..dim {
            ss += x[base + i] * x[base + i];
        }
        let inv = 1.0 / ((ss / dim as f32) + eps).sqrt();
        for i in 0..dim {
            out[base + i] = x[base + i] * inv * w[i];
        }
    }
    out
}

/// CPU reference for rotary position embedding, matching the kernel's half-split convention.
/// `cos`/`sin` are `[seq, head_dim/2]`. `x` is `[seq, n_heads, head_dim]`.
pub fn cpu_rope(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    n_heads: usize,
    hd: usize,
) -> Vec<f32> {
    let half = hd / 2;
    let mut out = x.to_vec();
    for s in 0..seq {
        for h in 0..n_heads {
            let base = (s * n_heads + h) * hd;
            for p in 0..half {
                let c = cos[s * half + p];
                let sn = sin[s * half + p];
                let x1 = x[base + p];
                let x2 = x[base + half + p];
                out[base + p] = x1 * c - x2 * sn;
                out[base + half + p] = x2 * c + x1 * sn;
            }
        }
    }
    out
}

/// CPU reference for causal multi-head attention with GQA. q `[h,s,hd]`, k/v `[hkv,s,hd]`,
/// returns o `[h,s,hd]`. Query head `hh` uses kv head `hh / (h/hkv)`. scale = 1/sqrt(hd).
pub fn cpu_attn_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h: usize,
    hkv: usize,
    s: usize,
    hd: usize,
) -> Vec<f32> {
    let group = h / hkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; h * s * hd];
    for hh in 0..h {
        let kvh = hh / group;
        for i in 0..s {
            let qbase = (hh * s + i) * hd;
            let mut scores = Vec::with_capacity(i + 1);
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                let kbase = (kvh * s + j) * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[qbase + d] * k[kbase + d];
                }
                dot *= scale;
                scores.push(dot);
                mx = mx.max(dot);
            }
            let mut sum = 0.0f32;
            for sc in &scores {
                sum += (sc - mx).exp();
            }
            for (j, sc) in scores.iter().enumerate() {
                let w = (sc - mx).exp() / sum;
                let vbase = (kvh * s + j) * hd;
                for d in 0..hd {
                    o[qbase + d] += w * v[vbase + d];
                }
            }
        }
    }
    o
}

/// CPU reference for a single decode step: q `[h,hd]` attends keys/values `0..=pos` of a KV
/// cache laid out `[hkv, max_seq, hd]`. Returns o `[h,hd]`.
pub fn cpu_attn_decode(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    h: usize,
    hkv: usize,
    pos: usize,
    hd: usize,
    max_seq: usize,
) -> Vec<f32> {
    let group = h / hkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; h * hd];
    for hh in 0..h {
        let kvh = hh / group;
        let qbase = hh * hd;
        let mut scores = Vec::with_capacity(pos + 1);
        let mut mx = f32::NEG_INFINITY;
        for j in 0..=pos {
            let kbase = (kvh * max_seq + j) * hd;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[qbase + d] * kc[kbase + d];
            }
            dot *= scale;
            scores.push(dot);
            mx = mx.max(dot);
        }
        let mut sum = 0.0f32;
        for sc in &scores {
            sum += (sc - mx).exp();
        }
        for (j, sc) in scores.iter().enumerate() {
            let w = (sc - mx).exp() / sum;
            let vbase = (kvh * max_seq + j) * hd;
            for d in 0..hd {
                o[qbase + d] += w * vc[vbase + d];
            }
        }
    }
    o
}

/// CPU reference for numerically-stable row softmax over `[rows, cols]`.
pub fn cpu_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut m = f32::NEG_INFINITY;
        for i in 0..cols {
            m = m.max(x[base + i]);
        }
        let mut sum = 0.0f32;
        for i in 0..cols {
            sum += (x[base + i] - m).exp();
        }
        for i in 0..cols {
            out[base + i] = (x[base + i] - m).exp() / sum;
        }
    }
    out
}
