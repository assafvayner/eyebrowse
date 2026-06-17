use std::sync::Arc;

use eyebrowse_core::DType;
use eyebrowse_gpu::{Device, Tensor};

/// Per-layer key/value caches, each laid out seq-major `[max_seq, n_kv_heads, head_dim]`.
pub struct KvCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub max_seq: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(
        dev: &Arc<Device>,
        n_layers: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let shape = [max_seq, n_kv_heads, head_dim];
        let k = (0..n_layers)
            .map(|_| Tensor::empty(dev, &shape, DType::F32))
            .collect();
        let v = (0..n_layers)
            .map(|_| Tensor::empty(dev, &shape, DType::F32))
            .collect();
        KvCache {
            k,
            v,
            max_seq,
            n_kv_heads,
            head_dim,
        }
    }
}
