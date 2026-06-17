use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::embedding_f16;

/// Token embedding lookup over a packed-f16 table `[vocab, dim]`.
pub struct Embedding {
    pub table: Tensor,
    pub vocab: usize,
    pub dim: usize,
}

impl Embedding {
    pub fn new(table: Tensor, vocab: usize, dim: usize) -> Self {
        Embedding { table, vocab, dim }
    }

    /// `ids` is u32 `[n]`; returns f32 `[n, dim]`.
    pub fn forward(&self, rec: &mut Recorder, ids: &Tensor, n: usize) -> Tensor {
        let out = Tensor::empty(rec.device(), &[n, self.dim], DType::F32);
        embedding_f16(rec, ids, &self.table, &out, n, self.dim);
        out
    }
}
