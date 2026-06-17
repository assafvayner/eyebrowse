use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::swiglu;

use crate::Linear;

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Mlp { gate, up, down }
    }

    /// `x` is `[rows, hidden]` f32; returns `[rows, hidden]` f32.
    pub fn forward(&self, rec: &mut Recorder, x: &Tensor, rows: usize) -> Tensor {
        let inter = self.gate.out_f;
        let g = self.gate.forward(rec, x, rows);
        let u = self.up.forward(rec, x, rows);
        let h = Tensor::empty(rec.device(), &[rows, inter], DType::F32);
        swiglu(rec, &g, &u, &h, rows * inter);
        self.down.forward(rec, &h, rows)
    }
}
