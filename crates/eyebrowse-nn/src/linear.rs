use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::linear_f16w;

/// A bias-free linear layer with packed-f16 weights `[out_f, in_f]` (HF layout).
pub struct Linear {
    pub w: Tensor,
    pub in_f: usize,
    pub out_f: usize,
}

impl Linear {
    pub fn new(w: Tensor, in_f: usize, out_f: usize) -> Self {
        Linear { w, in_f, out_f }
    }

    /// `x` is `[rows, in_f]` f32; returns `[rows, out_f]` f32.
    pub fn forward(&self, rec: &mut Recorder, x: &Tensor, rows: usize) -> Tensor {
        let out = Tensor::empty(rec.device(), &[rows, self.out_f], DType::F32);
        linear_f16w(rec, x, &self.w, &out, rows, self.in_f, self.out_f);
        out
    }
}
