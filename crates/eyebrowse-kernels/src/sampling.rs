//! Greedy sampling: GPU argmax over a `[vocab]` logits vector, writing the winning index
//! to a `u32` output. One workgroup; shared-memory argmax reduction.

use eyebrowse_gpu::{dispatch, Recorder, Tensor};

const ARGMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>; // [vocab]

var<workgroup> vmax: array<f32, 256>;
var<workgroup> imax: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let vocab = params[0];
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = lid.x; i < vocab; i = i + 256u) {
        let val = logits[i];
        if (val > bv) { bv = val; bi = i; }
    }
    vmax[lid.x] = bv;
    imax[lid.x] = bi;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            if (vmax[lid.x + stride] > vmax[lid.x]) {
                vmax[lid.x] = vmax[lid.x + stride];
                imax[lid.x] = imax[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid.x == 0u) { out_idx[0] = imax[0]; }
}
"#;

/// Write `argmax(logits)` (a single token id) into `out_idx[0]`. `out_idx` is a `[1]` u32 tensor.
pub fn argmax(rec: &mut Recorder, logits: &Tensor, out_idx: &Tensor, vocab: usize) {
    let params = Tensor::from_u32(rec.device(), &[1], &[vocab as u32]);
    dispatch(
        rec,
        "argmax",
        ARGMAX_WGSL,
        "main",
        &[&logits.buffer, &out_idx.buffer, &params.buffer],
        [1, 1, 1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_device;
    use eyebrowse_core::DType;

    #[test]
    fn argmax_finds_max() {
        let d = test_device();
        let vocab = 1000usize;
        let mut logits = vec![0.0f32; vocab];
        for (i, x) in logits.iter_mut().enumerate() {
            *x = (i as f32 * 0.001).sin();
        }
        logits[731] = 5.0; // unique max
        let lt = Tensor::from_f32(&d, &[vocab], &logits);
        let ot = Tensor::empty(&d, &[1], DType::U32);
        let mut rec = Recorder::new(&d);
        argmax(&mut rec, &lt, &ot, vocab);
        rec.submit();
        let bytes = pollster::block_on(ot.read_bytes()).unwrap();
        let idx = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(idx, 731);
    }
}
