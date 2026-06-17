//! Small elementwise / data-movement kernels used to assemble transformer layers:
//! SwiGLU activation, f16 embedding gather, and KV-cache writes.

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> params: array<u32>; // [n]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params[0];
    let i = gid.x;
    if (i >= n) { return; }
    let g = gate[i];
    let silu = g / (1.0 + exp(-g));
    out[i] = silu * up[i];
}
"#;

const EMBED_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> ids: array<u32>;    // [n]
@group(0) @binding(1) var<storage, read_write> table: array<u32>;  // packed f16 [vocab*dim/2]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;    // [n*dim]
@group(0) @binding(3) var<storage, read_write> params: array<u32>; // [n, dim]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params[0]; let dim = params[1];
    let idx = gid.x;
    if (idx >= n * dim) { return; }
    let t = idx / dim;
    let j = idx % dim;
    let flat = ids[t] * dim + j;
    let pair = unpack2x16float(table[flat >> 1u]);
    if ((flat & 1u) == 0u) { out[idx] = pair.x; } else { out[idx] = pair.y; }
}
"#;

const KV_WRITE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> cache: array<f32>;  // [max_seq, Hkv, hd]
@group(0) @binding(1) var<storage, read_write> src: array<f32>;    // [count, Hkv, hd]
@group(0) @binding(2) var<storage, read_write> params: array<u32>; // [hkv, count, hd, max_seq, dst_start]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hkv = params[0]; let count = params[1]; let hd = params[2];
    let max_seq = params[3]; let dst = params[4];
    let idx = gid.x;
    if (idx >= hkv * count * hd) { return; }
    let d = idx % hd;
    let tmp = idx / hd;
    let head = tmp % hkv;
    let c = tmp / hkv;
    let cache_idx = ((dst + c) * hkv + head) * hd + d;
    cache[cache_idx] = src[idx];
}
"#;

/// SwiGLU combine: `out[i] = silu(gate[i]) * up[i]`, where `silu(x) = x * sigmoid(x)`.
pub fn swiglu(rec: &mut Recorder, gate: &Tensor, up: &Tensor, out: &Tensor, n: usize) {
    let params = Tensor::from_u32(rec.device(), &[1], &[n as u32]);
    dispatch(
        rec,
        "swiglu",
        SWIGLU_WGSL,
        "main",
        &[&gate.buffer, &up.buffer, &out.buffer, &params.buffer],
        [(n as u32).div_ceil(64), 1, 1],
    );
}

/// Gather `n` rows of width `dim` from a packed-u32 f16 embedding table into f32 `out` (`[n,dim]`).
pub fn embedding_f16(
    rec: &mut Recorder,
    ids: &Tensor,
    table: &Tensor,
    out: &Tensor,
    n: usize,
    dim: usize,
) {
    let params = Tensor::from_u32(rec.device(), &[2], &[n as u32, dim as u32]);
    dispatch(
        rec,
        "embedding_f16",
        EMBED_WGSL,
        "main",
        &[&ids.buffer, &table.buffer, &out.buffer, &params.buffer],
        [((n * dim) as u32).div_ceil(64), 1, 1],
    );
}

/// Write `src` (`[hkv, count, hd]`) into `cache` (`[hkv, max_seq, hd]`) at sequence offset `dst_start`.
#[allow(clippy::too_many_arguments)]
pub fn kv_write(
    rec: &mut Recorder,
    cache: &Tensor,
    src: &Tensor,
    hkv: usize,
    count: usize,
    hd: usize,
    max_seq: usize,
    dst_start: usize,
) {
    let params = Tensor::from_u32(
        rec.device(),
        &[5],
        &[
            hkv as u32,
            count as u32,
            hd as u32,
            max_seq as u32,
            dst_start as u32,
        ],
    );
    dispatch(
        rec,
        "kv_write",
        KV_WRITE_WGSL,
        "main",
        &[&cache.buffer, &src.buffer, &params.buffer],
        [((hkv * count * hd) as u32).div_ceil(64), 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{pack_f16, rel_l2, round_f16, test_device};
    use eyebrowse_core::DType;

    #[test]
    fn swiglu_matches_cpu() {
        let d = test_device();
        let n = 100usize;
        let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - 2.0).collect();
        let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.02 - 1.0).collect();
        let gt = Tensor::from_f32(&d, &[n], &gate);
        let ut = Tensor::from_f32(&d, &[n], &up);
        let ot = Tensor::empty(&d, &[n], DType::F32);
        let mut rec = Recorder::new(&d);
        swiglu(&mut rec, &gt, &ut, &ot, n);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want: Vec<f32> = (0..n)
            .map(|i| {
                let g = gate[i];
                (g / (1.0 + (-g).exp())) * up[i]
            })
            .collect();
        assert!(rel_l2(&got, &want) < 1e-5);
    }

    #[test]
    fn embedding_gathers_rows() {
        let d = test_device();
        let (vocab, dim) = (4usize, 6usize);
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1 - 1.0).collect();
        let packed = pack_f16(&table);
        let ids = [2u32, 0u32, 3u32];
        let n = ids.len();
        let idt = Tensor::from_u32(&d, &[n], &ids);
        let tt = Tensor::from_u32(&d, &[packed.len()], &packed);
        let ot = Tensor::empty(&d, &[n, dim], DType::F32);
        let mut rec = Recorder::new(&d);
        embedding_f16(&mut rec, &idt, &tt, &ot, n, dim);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let table16 = round_f16(&table);
        let mut want = vec![0.0f32; n * dim];
        for (t, &id) in ids.iter().enumerate() {
            for j in 0..dim {
                want[t * dim + j] = table16[id as usize * dim + j];
            }
        }
        assert!(rel_l2(&got, &want) < 1e-6);
    }

    #[test]
    fn kv_write_places_rows() {
        let d = test_device();
        let (hkv, hd, max_seq) = (2usize, 4usize, 8usize);
        let cache = vec![0.0f32; max_seq * hkv * hd];
        let ct = Tensor::from_f32(&d, &[max_seq, hkv, hd], &cache);
        // write 1 position (seq-major src [1, hkv, hd]) at dst=3
        let src: Vec<f32> = (0..hkv * hd).map(|i| i as f32 + 1.0).collect();
        let st = Tensor::from_f32(&d, &[1, hkv, hd], &src);
        let mut rec = Recorder::new(&d);
        kv_write(&mut rec, &ct, &st, hkv, 1, hd, max_seq, 3);
        rec.submit();
        let got = pollster::block_on(ct.to_f32()).unwrap();
        // cache position 3 (for both heads) should equal src; position 0 stays zero.
        for head in 0..hkv {
            for d_ in 0..hd {
                let at = (3 * hkv + head) * hd + d_;
                assert_eq!(got[at], src[head * hd + d_]);
            }
        }
        assert_eq!(got[(0 * hkv + 0) * hd + 0], 0.0);
    }
}
