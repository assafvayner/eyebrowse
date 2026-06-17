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
        KvCache::new_per_layer(dev, &vec![head_dim; n_layers], max_seq, n_kv_heads)
    }

    /// Allocate per-layer caches where layer `l` uses `head_dims[l]`. The uniform `head_dim`
    /// field holds `head_dims[0]` (or 0 when empty); per-layer consumers read tensor shapes.
    pub fn new_per_layer(
        dev: &Arc<Device>,
        head_dims: &[usize],
        max_seq: usize,
        n_kv_heads: usize,
    ) -> Self {
        let k = head_dims
            .iter()
            .map(|&hd| Tensor::empty(dev, &[max_seq, n_kv_heads, hd], DType::F32))
            .collect();
        let v = head_dims
            .iter()
            .map(|&hd| Tensor::empty(dev, &[max_seq, n_kv_heads, hd], DType::F32))
            .collect();
        KvCache {
            k,
            v,
            max_seq,
            n_kv_heads,
            head_dim: head_dims.first().copied().unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_per_layer_sizes_each_layer() {
        let dev = pollster::block_on(Device::new()).expect("device");
        let cache = KvCache::new_per_layer(&dev, &[16, 32], 8, 2);
        assert_eq!(cache.k[0].shape, vec![8, 2, 16]);
        assert_eq!(cache.k[1].shape, vec![8, 2, 32]);
        assert_eq!(cache.v[0].shape, vec![8, 2, 16]);
        assert_eq!(cache.v[1].shape, vec![8, 2, 32]);
    }
}
