//! Flash-style attention with online softmax, causal masking, and grouped-query attention.
//! `head_dim` is baked into the shader (so the per-thread accumulator is a constant-size array)
//! and the pipeline is cached per head_dim. One thread computes one output row.
//!
//! - `attn_prefill`: full causal attention over q/k/v `[*, S, hd]`, output `[H, S, hd]`.
//! - `attn_decode`: a single query position attending `0..=pos` of a `[Hkv, max_seq, hd]` KV cache.

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const PREFILL_TMPL: &str = r#"
@group(0) @binding(0) var<storage, read_write> q: array<f32>;    // [S, H, HD]
@group(0) @binding(1) var<storage, read_write> k: array<f32>;    // [S, Hkv, HD]
@group(0) @binding(2) var<storage, read_write> v: array<f32>;    // [S, Hkv, HD]
@group(0) @binding(3) var<storage, read_write> o: array<f32>;    // [S, H, HD]
@group(0) @binding(4) var<storage, read_write> dims: array<u32>; // [H, Hkv, S]

const HD: u32 = __HD__u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H = dims[0]; let Hkv = dims[1]; let S = dims[2];
    let idx = gid.x;
    if (idx >= H * S) { return; }
    let i = idx / H;
    let h = idx % H;
    let group = H / Hkv;
    let kvh = h / group;
    let qbase = (i * H + h) * HD;
    let scale = 1.0 / sqrt(f32(HD));

    var acc: array<f32, __HD__>;
    for (var d = 0u; d < HD; d = d + 1u) { acc[d] = 0.0; }
    var m = -3.0e38;
    var l = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        let kbase = (j * Hkv + kvh) * HD;
        var s = 0.0;
        for (var d = 0u; d < HD; d = d + 1u) { s = s + q[qbase + d] * k[kbase + d]; }
        s = s * scale;
        let mnew = max(m, s);
        let corr = exp(m - mnew);
        let p = exp(s - mnew);
        l = l * corr + p;
        for (var d = 0u; d < HD; d = d + 1u) { acc[d] = acc[d] * corr + p * v[kbase + d]; }
        m = mnew;
    }
    let inv = 1.0 / l;
    for (var d = 0u; d < HD; d = d + 1u) { o[qbase + d] = acc[d] * inv; }
}
"#;

const DECODE_TMPL: &str = r#"
@group(0) @binding(0) var<storage, read_write> q: array<f32>;    // [H, HD] (current token)
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;   // [MAXSEQ, Hkv, HD]
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;   // [MAXSEQ, Hkv, HD]
@group(0) @binding(3) var<storage, read_write> o: array<f32>;    // [H, HD]
@group(0) @binding(4) var<storage, read_write> dims: array<u32>; // [H, Hkv, pos, MAXSEQ]

const HD: u32 = __HD__u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H = dims[0]; let Hkv = dims[1]; let pos = dims[2]; let max_seq = dims[3];
    let h = gid.x;
    if (h >= H) { return; }
    let group = H / Hkv;
    let kvh = h / group;
    let qbase = h * HD;
    let scale = 1.0 / sqrt(f32(HD));

    var acc: array<f32, __HD__>;
    for (var d = 0u; d < HD; d = d + 1u) { acc[d] = 0.0; }
    var m = -3.0e38;
    var l = 0.0;
    for (var j = 0u; j <= pos; j = j + 1u) {
        let kbase = (j * Hkv + kvh) * HD;
        var s = 0.0;
        for (var d = 0u; d < HD; d = d + 1u) { s = s + q[qbase + d] * kc[kbase + d]; }
        s = s * scale;
        let mnew = max(m, s);
        let corr = exp(m - mnew);
        let p = exp(s - mnew);
        l = l * corr + p;
        for (var d = 0u; d < HD; d = d + 1u) { acc[d] = acc[d] * corr + p * vc[kbase + d]; }
        m = mnew;
    }
    let inv = 1.0 / l;
    for (var d = 0u; d < HD; d = d + 1u) { o[qbase + d] = acc[d] * inv; }
}
"#;

/// Causal GQA attention over full sequences. q `[h,s,hd]`, k/v `[hkv,s,hd]`, out `[h,s,hd]`.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    rec: &mut Recorder,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    h: usize,
    hkv: usize,
    s: usize,
    hd: usize,
) {
    let dims = Tensor::from_u32(rec.device(), &[3], &[h as u32, hkv as u32, s as u32]);
    let total = (h * s) as u32;
    dispatch(
        rec,
        &format!("attn_prefill_hd{hd}"),
        &PREFILL_TMPL.replace("__HD__", &hd.to_string()),
        "main",
        &[&q.buffer, &k.buffer, &v.buffer, &o.buffer, &dims.buffer],
        [total.div_ceil(64), 1, 1],
    );
}

/// Single decode step. q `[h,hd]` attends `0..=pos` of a KV cache `[hkv,max_seq,hd]`; out `[h,hd]`.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode(
    rec: &mut Recorder,
    q: &Tensor,
    kcache: &Tensor,
    vcache: &Tensor,
    o: &Tensor,
    h: usize,
    hkv: usize,
    pos: usize,
    hd: usize,
    max_seq: usize,
) {
    let dims = Tensor::from_u32(
        rec.device(),
        &[4],
        &[h as u32, hkv as u32, pos as u32, max_seq as u32],
    );
    dispatch(
        rec,
        &format!("attn_decode_hd{hd}"),
        &DECODE_TMPL.replace("__HD__", &hd.to_string()),
        "main",
        &[
            &q.buffer,
            &kcache.buffer,
            &vcache.buffer,
            &o.buffer,
            &dims.buffer,
        ],
        [(h as u32).div_ceil(64), 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{cpu_attn_decode, cpu_attn_prefill, rel_l2, test_device};
    use eyebrowse_core::DType;

    fn ramp(n: usize, off: f32, step: f32) -> Vec<f32> {
        (0..n).map(|i| off + i as f32 * step).collect()
    }

    #[test]
    fn attn_prefill_matches_cpu() {
        let d = test_device();
        let (h, hkv, s, hd) = (2usize, 1usize, 5usize, 8usize);
        let q = ramp(h * s * hd, -0.5, 0.011);
        let k = ramp(hkv * s * hd, -0.3, 0.007);
        let v = ramp(hkv * s * hd, 0.2, 0.013);
        let qt = Tensor::from_f32(&d, &[s, h, hd], &q);
        let kt = Tensor::from_f32(&d, &[s, hkv, hd], &k);
        let vt = Tensor::from_f32(&d, &[s, hkv, hd], &v);
        let ot = Tensor::empty(&d, &[s, h, hd], DType::F32);
        let mut rec = Recorder::new(&d);
        attn_prefill(&mut rec, &qt, &kt, &vt, &ot, h, hkv, s, hd);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want = cpu_attn_prefill(&q, &k, &v, h, hkv, s, hd);
        assert!(rel_l2(&got, &want) < 1e-4, "rel_l2 too high");
    }

    #[test]
    fn attn_decode_matches_cpu() {
        let d = test_device();
        let (h, hkv, hd, max_seq, pos) = (4usize, 2usize, 8usize, 16usize, 5usize);
        let q = ramp(h * hd, -0.4, 0.02);
        let kc = ramp(hkv * max_seq * hd, -0.2, 0.005);
        let vc = ramp(hkv * max_seq * hd, 0.1, 0.009);
        let qt = Tensor::from_f32(&d, &[h, hd], &q);
        let kt = Tensor::from_f32(&d, &[max_seq, hkv, hd], &kc);
        let vt = Tensor::from_f32(&d, &[max_seq, hkv, hd], &vc);
        let ot = Tensor::empty(&d, &[h, hd], DType::F32);
        let mut rec = Recorder::new(&d);
        attn_decode(&mut rec, &qt, &kt, &vt, &ot, h, hkv, pos, hd, max_seq);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want = cpu_attn_decode(&q, &kc, &vc, h, hkv, pos, hd, max_seq);
        assert!(rel_l2(&got, &want) < 1e-4, "rel_l2 too high");
    }
}
