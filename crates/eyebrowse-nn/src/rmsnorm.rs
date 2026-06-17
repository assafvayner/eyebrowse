use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::rmsnorm;

/// RMSNorm over the last dimension (`dim`) of a `[rows, dim]` activation.
pub struct RmsNorm {
    pub w: Tensor,
    pub eps: f32,
    pub dim: usize,
}

impl RmsNorm {
    pub fn new(w: Tensor, eps: f32, dim: usize) -> Self {
        RmsNorm { w, eps, dim }
    }

    /// `x` is `[rows, dim]` f32; returns `[rows, dim]` f32.
    pub fn forward(&self, rec: &mut Recorder, x: &Tensor, rows: usize) -> Tensor {
        let out = Tensor::empty(rec.device(), &[rows, self.dim], DType::F32);
        rmsnorm(rec, x, &self.w, &out, rows, self.dim, self.eps);
        out
    }
}
