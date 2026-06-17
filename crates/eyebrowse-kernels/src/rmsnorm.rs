//! RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`, per row of `[rows, dim]`.
//! One workgroup per row; sum-of-squares via shared-memory reduction.

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> params: array<u32>; // [rows, dim, eps_bits]

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let dim = params[1];
    let eps = bitcast<f32>(params[2]);
    let base = wid.x * dim;
    var s = 0.0;
    for (var i = lid.x; i < dim; i = i + 256u) {
        let v = x[base + i];
        s = s + v * v;
    }
    red[lid.x] = s;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = 1.0 / sqrt(red[0] / f32(dim) + eps);
    for (var i = lid.x; i < dim; i = i + 256u) {
        out[base + i] = x[base + i] * inv * weight[i];
    }
}
"#;

/// RMSNorm over rows of `x` (`[rows, dim]`), scaled by `weight` (`[dim]`), into `out`.
pub fn rmsnorm(
    rec: &mut Recorder,
    x: &Tensor,
    weight: &Tensor,
    out: &Tensor,
    rows: usize,
    dim: usize,
    eps: f32,
) {
    let params = Tensor::from_u32(
        rec.device(),
        &[3],
        &[rows as u32, dim as u32, eps.to_bits()],
    );
    dispatch(
        rec,
        "rmsnorm",
        RMSNORM_WGSL,
        "main",
        &[&x.buffer, &weight.buffer, &out.buffer, &params.buffer],
        [rows as u32, 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{cpu_rmsnorm, rel_l2, test_device};
    use eyebrowse_core::DType;

    #[test]
    fn rmsnorm_matches_cpu() {
        let d = test_device();
        let (rows, dim) = (3usize, 320usize); // dim > 256 to exercise the strided reduction
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
            .collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let xt = Tensor::from_f32(&d, &[rows, dim], &x);
        let wt = Tensor::from_f32(&d, &[dim], &w);
        let ot = Tensor::empty(&d, &[rows, dim], DType::F32);
        let mut rec = Recorder::new(&d);
        rmsnorm(&mut rec, &xt, &wt, &ot, rows, dim, eps);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want = cpu_rmsnorm(&x, &w, rows, dim, eps);
        assert!(rel_l2(&got, &want) < 1e-4, "rel_l2 too high");
    }
}
