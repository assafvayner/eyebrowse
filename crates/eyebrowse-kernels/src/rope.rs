//! Rotary position embedding (HF half-split convention) over `[seq, n_heads, head_dim]`.
//! `cos`/`sin` are precomputed `[seq, head_dim/2]` tables. One thread per (seq, head, pair).

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const ROPE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cosb: array<f32>;  // [seq, hd/2]
@group(0) @binding(2) var<storage, read_write> sinb: array<f32>;  // [seq, hd/2]
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> dims: array<u32>;  // [seq, n_heads, hd, base_pos]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = dims[0]; let n_heads = dims[1]; let hd = dims[2]; let base_pos = dims[3];
    let half = hd / 2u;
    let total = seq * n_heads * half;
    let idx = gid.x;
    if (idx >= total) { return; }
    let p = idx % half;
    let tmp = idx / half;
    let head = tmp % n_heads;
    let s = tmp / n_heads;
    let base = (s * n_heads + head) * hd;
    let c = cosb[(s + base_pos) * half + p];
    let sn = sinb[(s + base_pos) * half + p];
    let x1 = x[base + p];
    let x2 = x[base + half + p];
    out[base + p] = x1 * c - x2 * sn;
    out[base + half + p] = x2 * c + x1 * sn;
}
"#;

/// Apply RoPE to `x` (`[seq, n_heads, head_dim]`) using `cos`/`sin` (`[max_seq, head_dim/2]`),
/// into `out`. `base_pos` is the absolute position of the first row (0 for prefill; the current
/// position for a single decode step).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    rec: &mut Recorder,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    out: &Tensor,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    base_pos: usize,
) {
    let dims = Tensor::from_u32(
        rec.device(),
        &[4],
        &[seq as u32, n_heads as u32, head_dim as u32, base_pos as u32],
    );
    let total = (seq * n_heads * (head_dim / 2)) as u32;
    dispatch(
        rec,
        "rope",
        ROPE_WGSL,
        "main",
        &[
            &x.buffer,
            &cos.buffer,
            &sin.buffer,
            &out.buffer,
            &dims.buffer,
        ],
        [total.div_ceil(64), 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{cpu_rope, rel_l2, test_device};
    use eyebrowse_core::DType;

    fn rope_tables(seq: usize, hd: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
        let half = hd / 2;
        let mut cos = vec![0.0f32; seq * half];
        let mut sin = vec![0.0f32; seq * half];
        for s in 0..seq {
            for k in 0..half {
                let freq = (theta as f64).powf(-2.0 * k as f64 / hd as f64);
                let ang = s as f64 * freq;
                cos[s * half + k] = ang.cos() as f32;
                sin[s * half + k] = ang.sin() as f32;
            }
        }
        (cos, sin)
    }

    #[test]
    fn rope_matches_cpu() {
        let d = test_device();
        let (seq, n_heads, hd) = (4usize, 2usize, 8usize);
        let theta = 10000.0f32;
        let (cos, sin) = rope_tables(seq, hd, theta);
        let x: Vec<f32> = (0..seq * n_heads * hd)
            .map(|i| (i as f32) * 0.03 - 0.5)
            .collect();
        let xt = Tensor::from_f32(&d, &[seq, n_heads, hd], &x);
        let ct = Tensor::from_f32(&d, &[seq, hd / 2], &cos);
        let st = Tensor::from_f32(&d, &[seq, hd / 2], &sin);
        let ot = Tensor::empty(&d, &[seq, n_heads, hd], DType::F32);
        let mut rec = Recorder::new(&d);
        rope(&mut rec, &xt, &ct, &st, &ot, seq, n_heads, hd, 0);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want = cpu_rope(&x, &cos, &sin, seq, n_heads, hd);
        assert!(rel_l2(&got, &want) < 1e-5, "rel_l2 too high");
    }
}
