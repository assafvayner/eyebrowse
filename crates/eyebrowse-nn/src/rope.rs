use std::sync::Arc;

use eyebrowse_core::DType;
use eyebrowse_gpu::{Device, Recorder, Tensor};
use eyebrowse_kernels::rope;

/// Precomputed rotary-position-embedding tables and the kernel application wrapper.
///
/// `cos`/`sin` are `[max_seq, head_dim/2]` with `angle(s, k) = s * theta^(-2k/head_dim)`.
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl Rope {
    pub fn build(dev: &Arc<Device>, max_seq: usize, head_dim: usize, theta: f32) -> Rope {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];
        let theta = theta as f64;
        let hd = head_dim as f64;
        for s in 0..max_seq {
            for k in 0..half {
                let inv_freq = theta.powf(-(2.0 * k as f64) / hd);
                let angle = s as f64 * inv_freq;
                cos[s * half + k] = angle.cos() as f32;
                sin[s * half + k] = angle.sin() as f32;
            }
        }
        Rope {
            cos: Tensor::from_f32(dev, &[max_seq, half], &cos),
            sin: Tensor::from_f32(dev, &[max_seq, half], &sin),
            head_dim,
            max_seq,
        }
    }

    /// `x` is `[seq, n_heads, head_dim]`; returns the rotated tensor of the same shape.
    /// `base_pos` is the sequence position of the first row.
    pub fn apply(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        seq: usize,
        n_heads: usize,
        base_pos: usize,
    ) -> Tensor {
        let out = Tensor::empty(rec.device(), &[seq, n_heads, self.head_dim], DType::F32);
        rope(
            rec,
            x,
            &self.cos,
            &self.sin,
            &out,
            seq,
            n_heads,
            self.head_dim,
            base_pos,
        );
        out
    }
}
