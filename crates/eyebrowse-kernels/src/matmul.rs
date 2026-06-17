//! Tiled GEMM kernels. `matmul` is the f32 reference; `matmul_f16w` reads a packed-u32 f16
//! weight (unpacked in-kernel) with an f32 activation and f32 accumulation — the v1 path.
//!
//! Matrix dims are passed via a small storage buffer (`dims`) rather than push constants,
//! since WebGPU has no push constants; this keeps the dispatch helper uniform.

use eyebrowse_gpu::{dispatch_with_uniform, uniform_u32, Recorder, Tensor};

const TILE: u32 = 16;

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;     // [M,K]
@group(0) @binding(1) var<storage, read_write> b: array<f32>;     // [K,N]
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // [M,N]
@group(0) @binding(3) var<uniform> dims: vec4<u32>;               // [M,K,N,_]

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = dims.x; let K = dims.y; let N = dims.z;
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let n_tiles = (K + 15u) / 16u;
    for (var t = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * 16u + lid.x;
        let b_row = t * 16u + lid.y;
        As[lid.y * 16u + lid.x] = select(0.0, a[row * K + a_col], row < M && a_col < K);
        Bs[lid.y * 16u + lid.x] = select(0.0, b[b_row * N + col], b_row < K && col < N);
        workgroupBarrier();
        for (var p = 0u; p < 16u; p = p + 1u) {
            acc = acc + As[lid.y * 16u + p] * Bs[p * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (row < M && col < N) { c[row * N + col] = acc; }
}
"#;

const MATMUL_F16W_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;     // [M,K] activations
@group(0) @binding(1) var<storage, read_write> b: array<u32>;     // [K,N] f16 packed 2/word, row-major flatten
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // [M,N]
@group(0) @binding(3) var<uniform> dims: vec4<u32>;               // [M,K,N,_]

// Read logical f16 weight element at flat index `idx` (row-major over [K,N]) from the packed u32 buffer.
fn wld(idx: u32) -> f32 {
    let word = b[idx >> 1u];
    let pair = unpack2x16float(word);
    if ((idx & 1u) == 0u) { return pair.x; } else { return pair.y; }
}

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = dims.x; let K = dims.y; let N = dims.z;
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let n_tiles = (K + 15u) / 16u;
    for (var t = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * 16u + lid.x;
        let b_row = t * 16u + lid.y;
        As[lid.y * 16u + lid.x] = select(0.0, a[row * K + a_col], row < M && a_col < K);
        Bs[lid.y * 16u + lid.x] = select(0.0, wld(b_row * N + col), b_row < K && col < N);
        workgroupBarrier();
        for (var p = 0u; p < 16u; p = p + 1u) {
            acc = acc + As[lid.y * 16u + p] * Bs[p * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (row < M && col < N) { c[row * N + col] = acc; }
}
"#;

// Like matmul_f16w but the weight is in HF Linear layout [N=out, K=in] (row-major), so
// `y[m,o] = sum_k a[m,k] * w[o,k]` — no transpose needed at load time.
const LINEAR_F16W_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;     // [M,K] activations
@group(0) @binding(1) var<storage, read_write> b: array<u32>;     // [N,K] f16 packed 2/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // [M,N]
@group(0) @binding(3) var<uniform> dims: vec4<u32>;               // [M,K,N,_]

fn wld(idx: u32) -> f32 {
    let pair = unpack2x16float(b[idx >> 1u]);
    if ((idx & 1u) == 0u) { return pair.x; } else { return pair.y; }
}

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = dims.x; let K = dims.y; let N = dims.z;
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let n_tiles = (K + 15u) / 16u;
    for (var t = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * 16u + lid.x;
        let b_row = t * 16u + lid.y;
        As[lid.y * 16u + lid.x] = select(0.0, a[row * K + a_col], row < M && a_col < K);
        Bs[lid.y * 16u + lid.x] = select(0.0, wld(col * K + b_row), col < N && b_row < K);
        workgroupBarrier();
        for (var p = 0u; p < 16u; p = p + 1u) {
            acc = acc + As[lid.y * 16u + p] * Bs[p * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (row < M && col < N) { c[row * N + col] = acc; }
}
"#;

/// Linear layer GEMM with HF weight layout: weight `w` is packed-u32 f16 of logical `[out, in]`,
/// computing `c[m, out] = a[m, in] * w[out, in]^T`. f32 activations/accumulation.
pub fn linear_f16w(
    rec: &mut Recorder,
    a: &Tensor,
    w: &Tensor,
    c: &Tensor,
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    let dims = uniform_u32(rec.device(), &[m as u32, in_f as u32, out_f as u32, 0]);
    let gx = (out_f as u32).div_ceil(TILE);
    let gy = (m as u32).div_ceil(TILE);
    dispatch_with_uniform(
        rec,
        "linear_f16w",
        LINEAR_F16W_WGSL,
        "main",
        &[&a.buffer, &w.buffer, &c.buffer],
        &[&dims],
        [gx, gy, 1],
    );
}

/// f32 GEMM: `c[m,n] = a[m,k] * b[k,n]` (all row-major). Records into `rec`.
pub fn matmul(
    rec: &mut Recorder,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let dims = uniform_u32(rec.device(), &[m as u32, k as u32, n as u32, 0]);
    let gx = (n as u32).div_ceil(TILE);
    let gy = (m as u32).div_ceil(TILE);
    dispatch_with_uniform(
        rec,
        "matmul_f32",
        MATMUL_WGSL,
        "main",
        &[&a.buffer, &b.buffer, &c.buffer],
        &[&dims],
        [gx, gy, 1],
    );
}

/// GEMM with a packed-u32 f16 weight `b` (logical `[k,n]`): `c[m,n] = a[m,k] * b[k,n]`,
/// f32 activations/accumulation. `b.buffer` holds `ceil(k*n/2)` u32 words (see `pack_f16`).
pub fn matmul_f16w(
    rec: &mut Recorder,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let dims = uniform_u32(rec.device(), &[m as u32, k as u32, n as u32, 0]);
    let gx = (n as u32).div_ceil(TILE);
    let gy = (m as u32).div_ceil(TILE);
    dispatch_with_uniform(
        rec,
        "matmul_f16w",
        MATMUL_F16W_WGSL,
        "main",
        &[&a.buffer, &b.buffer, &c.buffer],
        &[&dims],
        [gx, gy, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{cpu_linear, cpu_matmul, pack_f16, rel_l2, round_f16, test_device};
    use eyebrowse_core::DType;

    #[test]
    fn matmul_f32_matches_cpu() {
        let d = test_device();
        let (m, k, n) = (3usize, 4usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.07 - 0.5).collect();
        let at = Tensor::from_f32(&d, &[m, k], &a);
        let bt = Tensor::from_f32(&d, &[k, n], &b);
        let ct = Tensor::empty(&d, &[m, n], DType::F32);
        let mut rec = Recorder::new(&d);
        matmul(&mut rec, &at, &bt, &ct, m, k, n);
        rec.submit();
        let got = pollster::block_on(ct.to_f32()).unwrap();
        let want = cpu_matmul(&a, &b, m, k, n);
        assert!(rel_l2(&got, &want) < 1e-4, "got {got:?} want {want:?}");
    }

    #[test]
    fn matmul_f32_non_multiple_dims() {
        let d = test_device();
        let (m, k, n) = (17usize, 33usize, 19usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 * 0.03).collect();
        let at = Tensor::from_f32(&d, &[m, k], &a);
        let bt = Tensor::from_f32(&d, &[k, n], &b);
        let ct = Tensor::empty(&d, &[m, n], DType::F32);
        let mut rec = Recorder::new(&d);
        matmul(&mut rec, &at, &bt, &ct, m, k, n);
        rec.submit();
        let got = pollster::block_on(ct.to_f32()).unwrap();
        let want = cpu_matmul(&a, &b, m, k, n);
        assert!(rel_l2(&got, &want) < 1e-4);
    }

    #[test]
    fn matmul_f16w_matches_cpu() {
        let d = test_device();
        let (m, k, n) = (5usize, 8usize, 6usize);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.02 - 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.03 - 0.2).collect();
        let at = Tensor::from_f32(&d, &[m, k], &a);
        let packed = pack_f16(&b);
        let bt = Tensor::from_u32(&d, &[packed.len()], &packed);
        let ct = Tensor::empty(&d, &[m, n], DType::F32);
        let mut rec = Recorder::new(&d);
        matmul_f16w(&mut rec, &at, &bt, &ct, m, k, n);
        rec.submit();
        let got = pollster::block_on(ct.to_f32()).unwrap();
        // Reference multiplies with f16-rounded weights (kernel reads f16).
        let want = cpu_matmul(&a, &round_f16(&b), m, k, n);
        assert!(rel_l2(&got, &want) < 2e-3, "got {got:?} want {want:?}");
    }

    #[test]
    fn linear_f16w_matches_cpu() {
        let d = test_device();
        let (m, in_f, out_f) = (5usize, 33usize, 7usize); // in not a multiple of 16
        let x: Vec<f32> = (0..m * in_f).map(|i| i as f32 * 0.01 - 0.2).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| i as f32 * 0.005 - 0.3).collect();
        let xt = Tensor::from_f32(&d, &[m, in_f], &x);
        let packed = pack_f16(&w);
        let wt = Tensor::from_u32(&d, &[packed.len()], &packed);
        let ct = Tensor::empty(&d, &[m, out_f], DType::F32);
        let mut rec = Recorder::new(&d);
        linear_f16w(&mut rec, &xt, &wt, &ct, m, in_f, out_f);
        rec.submit();
        let got = pollster::block_on(ct.to_f32()).unwrap();
        let want = cpu_linear(&x, &round_f16(&w), m, in_f, out_f);
        assert!(rel_l2(&got, &want) < 2e-3, "got {got:?} want {want:?}");
    }
}
