//! Numerically-stable row softmax over `[rows, cols]`. One workgroup per row; two
//! shared-memory reductions (row max, then sum of exp).

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>; // [rows, cols]

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = params[1];
    let base = wid.x * cols;

    var m = -3.0e38;
    for (var i = lid.x; i < cols; i = i + 256u) { m = max(m, x[base + i]); }
    red[lid.x] = m;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = max(red[lid.x], red[lid.x + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = red[0];
    workgroupBarrier();

    var s = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) { s = s + exp(x[base + i] - row_max); }
    red[lid.x] = s;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = 1.0 / red[0];
    for (var i = lid.x; i < cols; i = i + 256u) {
        out[base + i] = exp(x[base + i] - row_max) * inv;
    }
}
"#;

/// Row softmax of `x` (`[rows, cols]`) into `out`.
pub fn softmax(rec: &mut Recorder, x: &Tensor, out: &Tensor, rows: usize, cols: usize) {
    let params = Tensor::from_u32(rec.device(), &[2], &[rows as u32, cols as u32]);
    dispatch(
        rec,
        "softmax",
        SOFTMAX_WGSL,
        "main",
        &[&x.buffer, &out.buffer, &params.buffer],
        [rows as u32, 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{cpu_softmax, rel_l2, test_device};
    use eyebrowse_core::DType;

    #[test]
    fn softmax_matches_cpu() {
        let d = test_device();
        let (rows, cols) = (4usize, 300usize); // cols > 256 to exercise the strided reduction
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i % 23) as f32) * 0.1 - 1.0).collect();
        let xt = Tensor::from_f32(&d, &[rows, cols], &x);
        let ot = Tensor::empty(&d, &[rows, cols], DType::F32);
        let mut rec = Recorder::new(&d);
        softmax(&mut rec, &xt, &ot, rows, cols);
        rec.submit();
        let got = pollster::block_on(ot.to_f32()).unwrap();
        let want = cpu_softmax(&x, rows, cols);
        assert!(rel_l2(&got, &want) < 1e-5, "rel_l2 too high");
    }
}
