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
        Rope::build_partial(dev, max_seq, head_dim, head_dim / 2, theta)
    }

    /// Like `build`, but only the first `rope_angles` frequencies are nonzero; the remaining
    /// pairs get cos=1, sin=0 (pass-through). Realizes Gemma 4's partial-rotary global RoPE.
    pub fn build_partial(
        dev: &Arc<Device>,
        max_seq: usize,
        head_dim: usize,
        rope_angles: usize,
        theta: f32,
    ) -> Rope {
        let half = head_dim / 2;
        let mut cos = vec![1.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];
        let theta = theta as f64;
        let hd = head_dim as f64;
        for s in 0..max_seq {
            for k in 0..rope_angles.min(half) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_partial_zeros_high_frequencies() {
        let dev = pollster::block_on(Device::new()).expect("device");
        let (max_seq, head_dim) = (5usize, 8usize);
        let half = head_dim / 2;
        let full = Rope::build(&dev, max_seq, head_dim, 10000.0);
        let partial = Rope::build_partial(&dev, max_seq, head_dim, 2, 10000.0);
        let full_cos = pollster::block_on(full.cos.to_f32()).unwrap();
        let full_sin = pollster::block_on(full.sin.to_f32()).unwrap();
        let part_cos = pollster::block_on(partial.cos.to_f32()).unwrap();
        let part_sin = pollster::block_on(partial.sin.to_f32()).unwrap();
        for s in 0..max_seq {
            for k in 0..2 {
                assert_eq!(part_cos[s * half + k], full_cos[s * half + k]);
                assert_eq!(part_sin[s * half + k], full_sin[s * half + k]);
            }
            for k in 2..half {
                assert_eq!(part_cos[s * half + k], 1.0);
                assert_eq!(part_sin[s * half + k], 0.0);
            }
        }
    }
}
