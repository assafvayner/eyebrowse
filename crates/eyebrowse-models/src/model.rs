//! The architecture-agnostic decoder driver.
//!
//! A [`LanguageModel`] owns the embedding table, a stack of [`Block`]s, the final norm, and the LM
//! head, and implements the shared prefill/decode skeleton exactly once. Per-architecture modules
//! ([`crate::decoder`], [`crate::gemma4`]) build the block list and set the few hooks that differ
//! (embedding scale, final-logit softcap). Adding an architecture means writing a `Block` and a
//! loader — no change to this driver.

use std::sync::Arc;

use eyebrowse_core::{DType, Result};
use eyebrowse_gpu::{copy_range, Device, Recorder, Tensor};
use eyebrowse_kernels::{argmax, embedding_f16, linear_f16w, mul_scalar};
use eyebrowse_load::Config;
use eyebrowse_nn::{KvCache, RmsNorm};

/// One decoder block: the attention and feed-forward sub-blocks with their norms and residuals.
/// Implementations own everything they need (their RoPE table, norms, activation), so the driver
/// can treat every architecture uniformly.
pub trait Block {
    /// Update the hidden states `x` (`[rows, hidden]`) for this layer, returning the new states.
    /// `base_pos` is the absolute position of the first row (0 for prefill, `pos` for decode).
    fn forward(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        kv: &mut KvCache,
        layer: usize,
        rows: usize,
        base_pos: usize,
    ) -> Tensor;
}

/// A loaded decoder-only language model of any supported architecture.
pub struct LanguageModel {
    pub(crate) dev: Arc<Device>,
    /// Packed-f16 embedding table `[vocab, hidden]`, also the tied LM-head weight.
    pub(crate) embed_w: Tensor,
    /// Separate LM-head weight when the model is untied (`None` => use `embed_w`).
    pub(crate) lm_head_w: Option<Tensor>,
    pub(crate) blocks: Vec<Box<dyn Block>>,
    pub(crate) norm: RmsNorm,
    /// Embedding scale (Gemma multiplies embeddings by `√hidden`); `None` => no scaling.
    pub(crate) embed_scale: Option<f32>,
    /// Final-logit softcap `cap * tanh(x / cap)` (Gemma); `None` => no softcap. Monotonic, so it is
    /// skipped on the argmax path.
    pub(crate) logit_softcap: Option<f32>,
    /// Per-layer KV head_dim (uniform for most archs; per-layer for Gemma's local/global layers).
    pub(crate) head_dims: Vec<usize>,
    pub(crate) n_kv_heads: usize,
    pub cfg: Config,
}

impl LanguageModel {
    /// Build a KV cache sized for this model and the given max sequence length.
    pub fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        KvCache::new_per_layer(&self.dev, &self.head_dims, max_seq, self.n_kv_heads)
    }

    fn lm_head_weight(&self) -> &Tensor {
        self.lm_head_w.as_ref().unwrap_or(&self.embed_w)
    }

    /// Embed `ids` (`[rows]`) and apply the optional embedding scale, returning `[rows, hidden]`.
    fn embed(&self, rec: &mut Recorder, ids: &[u32], rows: usize) -> Tensor {
        let hidden = self.cfg.hidden;
        let ids_t = Tensor::from_u32(&self.dev, &[rows], ids);
        let emb = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
        embedding_f16(rec, &ids_t, &self.embed_w, &emb, rows, hidden);
        match self.embed_scale {
            Some(s) => {
                let scaled = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
                mul_scalar(rec, &emb, &scaled, rows * hidden, s);
                scaled
            }
            None => emb,
        }
    }

    /// Run every block over already-embedded `x`, apply the final norm, and return the LAST row's
    /// hidden state as `[1, hidden]` (the only row the LM head needs).
    fn run_blocks_last(
        &self,
        rec: &mut Recorder,
        x: Tensor,
        kv: &mut KvCache,
        rows: usize,
        base_pos: usize,
    ) -> Tensor {
        let hidden = self.cfg.hidden;
        let mut x = x;
        for (l, block) in self.blocks.iter().enumerate() {
            x = block.forward(rec, &x, kv, l, rows, base_pos);
        }
        let xn = self.norm.forward(rec, &x, rows);
        if rows == 1 {
            return xn;
        }
        let last = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        copy_range(rec, &xn, &last, (rows - 1) * hidden, 0, hidden);
        last
    }

    /// Record embed -> blocks -> final norm -> LM head into `rec`, returning the (on-GPU) logits
    /// tensor `[1, vocab]` for the last position. Prefill (`base_pos == 0`, many ids) and decode
    /// (`base_pos == pos`, one id) flow through the same path; `Attention::forward` picks the
    /// batched or single-step kernel from `rows`/`base_pos`.
    fn record_logits(
        &self,
        rec: &mut Recorder,
        ids: &[u32],
        kv: &mut KvCache,
        base_pos: usize,
    ) -> Tensor {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        let rows = ids.len();
        let x = self.embed(rec, ids, rows);
        let last = self.run_blocks_last(rec, x, kv, rows, base_pos);
        let logits = Tensor::empty(&self.dev, &[1, vocab], DType::F32);
        linear_f16w(rec, &last, self.lm_head_weight(), &logits, 1, hidden, vocab);
        logits
    }

    fn softcap(&self, logits: &mut [f32]) {
        if let Some(cap) = self.logit_softcap {
            for v in logits.iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }
    }

    async fn logits_for(&self, ids: &[u32], base_pos: usize, kv: &mut KvCache) -> Result<Vec<f32>> {
        let mut rec = Recorder::new(&self.dev);
        let logits = self.record_logits(&mut rec, ids, kv, base_pos);
        rec.submit();
        let mut out = logits.to_f32().await?;
        self.softcap(&mut out);
        Ok(out)
    }

    /// Prefill over `ids`, returning the full logits (`[vocab]`) for the last position.
    pub async fn forward_prefill(&self, ids: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        self.logits_for(ids, 0, kv).await
    }

    /// Decode one `token` at absolute position `pos`, returning the full logits (`[vocab]`).
    pub async fn forward_decode(
        &self,
        token: u32,
        pos: usize,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>> {
        self.logits_for(&[token], pos, kv).await
    }

    async fn argmax_for(&self, ids: &[u32], base_pos: usize, kv: &mut KvCache) -> Result<u32> {
        let mut rec = Recorder::new(&self.dev);
        let logits = self.record_logits(&mut rec, ids, kv, base_pos);
        let out_idx = Tensor::empty(&self.dev, &[1], DType::U32);
        argmax(&mut rec, &logits, &out_idx, self.cfg.vocab);
        rec.submit();
        let bytes = out_idx.read_bytes().await?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Prefill over `ids` and return the greedy (argmax) next token, reading back a single `u32`
    /// instead of the whole vocab. The softcap is monotonic, so it does not change the argmax.
    pub async fn prefill_argmax(&self, ids: &[u32], kv: &mut KvCache) -> Result<u32> {
        self.argmax_for(ids, 0, kv).await
    }

    /// Decode one `token` at position `pos` and return the greedy (argmax) next token.
    pub async fn decode_argmax(&self, token: u32, pos: usize, kv: &mut KvCache) -> Result<u32> {
        self.argmax_for(&[token], pos, kv).await
    }
}
