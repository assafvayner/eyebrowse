use std::sync::Arc;

use eyebrowse_gpu::{pack_f16, Device, Recorder, Tensor};

use crate::testutil::{cpu_linear, prng, read, rel_l2, round_f16};
use crate::{Attention, Embedding, KvCache, Linear, Mlp, RmsNorm, Rope};

fn dev() -> Arc<Device> {
    pollster::block_on(Device::new()).expect("device")
}

// ---- CPU references ----

fn cpu_rmsnorm(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ---- tests ----

#[test]
fn linear_matches_cpu() {
    let d = dev();
    let (m, in_f, out_f) = (4usize, 8usize, 6usize);
    let x = prng(m * in_f, 1);
    let w = prng(out_f * in_f, 2);

    let xt = Tensor::from_f32(&d, &[m, in_f], &x);
    let wt = Tensor::from_u32(&d, &[pack_f16(&w).len()], &pack_f16(&w));
    let lin = Linear::new(wt, in_f, out_f);

    let mut rec = Recorder::new(&d);
    let out = lin.forward(&mut rec, &xt, m);
    rec.submit();
    let got = read(&out);

    let want = cpu_linear(&x, &round_f16(&w), m, in_f, out_f);
    assert!(
        rel_l2(&got, &want) < 2e-3,
        "rel_l2 = {}",
        rel_l2(&got, &want)
    );
}

#[test]
fn rmsnorm_matches_cpu() {
    let d = dev();
    let (rows, dim, eps) = (5usize, 16usize, 1e-6f32);
    let x = prng(rows * dim, 3);
    let w = prng(dim, 4);

    let xt = Tensor::from_f32(&d, &[rows, dim], &x);
    let wt = Tensor::from_f32(&d, &[dim], &w);
    let rms = RmsNorm::new(wt, eps, dim);

    let mut rec = Recorder::new(&d);
    let out = rms.forward(&mut rec, &xt, rows);
    rec.submit();
    let got = read(&out);

    let want = cpu_rmsnorm(&x, &w, rows, dim, eps);
    assert!(
        rel_l2(&got, &want) < 1e-4,
        "rel_l2 = {}",
        rel_l2(&got, &want)
    );
}

#[test]
fn embedding_matches_cpu() {
    let d = dev();
    let (vocab, dim, n) = (10usize, 8usize, 4usize);
    let table = prng(vocab * dim, 5);
    let ids = [3u32, 0, 9, 5];

    let table_t = Tensor::from_u32(&d, &[pack_f16(&table).len()], &pack_f16(&table));
    let ids_t = Tensor::from_u32(&d, &[n], &ids);
    let emb = Embedding::new(table_t, vocab, dim);

    let mut rec = Recorder::new(&d);
    let out = emb.forward(&mut rec, &ids_t, n);
    rec.submit();
    let got = read(&out);

    let table16 = round_f16(&table);
    let mut want = vec![0.0f32; n * dim];
    for (row, &id) in ids.iter().enumerate() {
        let src = id as usize * dim;
        want[row * dim..row * dim + dim].copy_from_slice(&table16[src..src + dim]);
    }
    assert!(
        rel_l2(&got, &want) < 2e-3,
        "rel_l2 = {}",
        rel_l2(&got, &want)
    );
}

#[test]
fn mlp_matches_cpu() {
    let d = dev();
    let (rows, hidden, inter) = (3usize, 8usize, 16usize);
    let x = prng(rows * hidden, 6);
    let wg = prng(inter * hidden, 7);
    let wu = prng(inter * hidden, 8);
    let wd = prng(hidden * inter, 9);

    let mk = |w: &[f32], inf, outf| {
        Linear::new(
            Tensor::from_u32(&d, &[pack_f16(w).len()], &pack_f16(w)),
            inf,
            outf,
        )
    };
    let mlp = Mlp::new(
        mk(&wg, hidden, inter),
        mk(&wu, hidden, inter),
        mk(&wd, inter, hidden),
    );

    let xt = Tensor::from_f32(&d, &[rows, hidden], &x);
    let mut rec = Recorder::new(&d);
    let out = mlp.forward(&mut rec, &xt, rows);
    rec.submit();
    let got = read(&out);

    let g = cpu_linear(&x, &round_f16(&wg), rows, hidden, inter);
    let u = cpu_linear(&x, &round_f16(&wu), rows, hidden, inter);
    let h: Vec<f32> = g.iter().zip(u.iter()).map(|(a, b)| silu(*a) * *b).collect();
    let want = cpu_linear(&h, &round_f16(&wd), rows, inter, hidden);
    assert!(
        rel_l2(&got, &want) < 2e-3,
        "rel_l2 = {}",
        rel_l2(&got, &want)
    );
}

#[test]
fn rope_build_tables() {
    let d = dev();
    let (max_seq, hd, theta) = (8usize, 4usize, 10000.0f32);
    let rope = Rope::build(&d, max_seq, hd, theta);
    let half = hd / 2;
    let cos = read(&rope.cos);
    let sin = read(&rope.sin);

    // Row 0: angle == 0 for all k, so cos == 1, sin == 0.
    for k in 0..half {
        assert!((cos[k] - 1.0).abs() < 1e-6, "cos[0,{k}] = {}", cos[k]);
        assert!(sin[k].abs() < 1e-6, "sin[0,{k}] = {}", sin[k]);
    }

    // A nonzero row matches a hand CPU computation of angle = s * theta^(-2k/hd).
    let s = 3usize;
    for k in 0..half {
        let inv_freq = (theta as f64).powf(-(2.0 * k as f64) / hd as f64);
        let angle = s as f64 * inv_freq;
        let want_c = angle.cos() as f32;
        let want_s = angle.sin() as f32;
        assert!((cos[s * half + k] - want_c).abs() < 1e-5);
        assert!((sin[s * half + k] - want_s).abs() < 1e-5);
    }
}

/// The key end-to-end check: decode fed one token at a time must agree with prefill at the
/// last position, proving cache layout + rope base_pos + the decode path match prefill.
#[test]
fn prefill_decode_consistency() {
    let d = dev();
    let hidden = 16usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 4usize;
    let seq = 3usize;
    let max_seq = 8usize;
    let eps = 1e-6f32;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mk_lin = |w: &[f32], inf: usize, outf: usize| {
        Linear::new(
            Tensor::from_u32(&d, &[pack_f16(w).len()], &pack_f16(w)),
            inf,
            outf,
        )
    };
    let mk_norm = |w: Vec<f32>| RmsNorm::new(Tensor::from_f32(&d, &[head_dim], &w), eps, head_dim);

    let attn = Attention {
        q_proj: mk_lin(&prng(q_dim * hidden, 11), hidden, q_dim),
        k_proj: mk_lin(&prng(kv_dim * hidden, 12), hidden, kv_dim),
        v_proj: mk_lin(&prng(kv_dim * hidden, 13), hidden, kv_dim),
        o_proj: mk_lin(&prng(hidden * q_dim, 14), q_dim, hidden),
        q_norm: Some(mk_norm(prng(head_dim, 15))),
        k_norm: Some(mk_norm(prng(head_dim, 16))),
        v_norm: None,
        n_heads,
        n_kv_heads,
        head_dim,
        hidden,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };
    let rope = Rope::build(&d, max_seq, head_dim, 10000.0);

    let x = prng(seq * hidden, 17);
    let xt = Tensor::from_f32(&d, &[seq, hidden], &x);

    // Prefill.
    let mut kv_p = KvCache::new(&d, 1, max_seq, n_kv_heads, head_dim);
    let mut rec = Recorder::new(&d);
    let pre = attn.prefill(&mut rec, &xt, &rope, &mut kv_p, 0, seq);
    rec.submit();
    let pre_out = read(&pre);
    let o_pre_last = &pre_out[(seq - 1) * hidden..seq * hidden];

    // Decode the same rows one at a time through a fresh cache.
    let mut kv_d = KvCache::new(&d, 1, max_seq, n_kv_heads, head_dim);
    let mut last = vec![0.0f32; hidden];
    for pos in 0..seq {
        let row = &x[pos * hidden..(pos + 1) * hidden];
        let row_t = Tensor::from_f32(&d, &[1, hidden], row);
        let mut rec = Recorder::new(&d);
        let out = attn.decode(&mut rec, &row_t, &rope, &mut kv_d, 0, pos);
        rec.submit();
        last = read(&out);
    }

    let r = rel_l2(&last, o_pre_last);
    assert!(r < 2e-3, "prefill/decode rel_l2 = {r}");
}
