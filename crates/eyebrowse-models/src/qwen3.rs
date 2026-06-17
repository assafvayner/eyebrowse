//! Qwen3 (dense) assembled from the shared `eyebrowse-nn` primitives. Adding another decoder-only
//! architecture (Llama, Mistral, …) is a sibling module reusing the same primitives + this pattern.
//!
//! Per-layer: RMSNorm -> attention (GQA + QK-RMSNorm + RoPE) -> residual -> RMSNorm -> SwiGLU MLP
//! -> residual. Final RMSNorm then the LM head (tied to the embedding for Qwen3-0.6B). The whole
//! forward records into one `Recorder` and submits once.

use std::sync::Arc;

use eyebrowse_core::{DType, Result};
use eyebrowse_gpu::{add, copy_range, Device, Recorder, Tensor};
use eyebrowse_kernels::{embedding_f16, linear_f16w};
use eyebrowse_load::{Config, WeightSource};
use eyebrowse_nn::{Attention, KvCache, Linear, Mlp, RmsNorm, Rope};

use crate::upload::{upload_f16, upload_f32};

struct Layer {
    ln1: RmsNorm,
    attn: Attention,
    ln2: RmsNorm,
    mlp: Mlp,
}

pub struct Qwen3Model {
    dev: Arc<Device>,
    /// Packed-f16 embedding table `[vocab, hidden]`, also used as the (tied) LM-head weight.
    embed_w: Tensor,
    /// Separate LM-head weight when the model is not tied (`None` => use `embed_w`).
    lm_head_w: Option<Tensor>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope: Rope,
    pub cfg: Config,
}

impl Qwen3Model {
    /// Build a KV cache sized for this model and the given max sequence length.
    pub fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        KvCache::new(
            &self.dev,
            self.cfg.n_layers,
            max_seq,
            self.cfg.n_kv_heads,
            self.cfg.head_dim,
        )
    }

    /// Load all weights from a `WeightSource` and precompute RoPE tables up to `max_seq`.
    pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<Self> {
        let cfg = src.config().clone();
        let (hidden, head_dim, n_heads, n_kv, inter, eps) = (
            cfg.hidden,
            cfg.head_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.intermediate,
            cfg.rms_eps,
        );
        let f16 = |name: &str| -> Result<Tensor> { upload_f16(dev, &src.raw(name)?) };
        let f32t = |name: &str| -> Result<Tensor> { upload_f32(dev, &src.raw(name)?) };

        let embed_w = f16("model.embed_tokens.weight")?;
        let names = src.tensor_names();
        let lm_head_w = if names.iter().any(|n| n == "lm_head.weight") {
            Some(f16("lm_head.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            let ln1 = RmsNorm::new(f32t(&format!("{p}.input_layernorm.weight"))?, eps, hidden);
            let attn = Attention {
                q_proj: Linear::new(
                    f16(&format!("{p}.self_attn.q_proj.weight"))?,
                    hidden,
                    n_heads * head_dim,
                ),
                k_proj: Linear::new(
                    f16(&format!("{p}.self_attn.k_proj.weight"))?,
                    hidden,
                    n_kv * head_dim,
                ),
                v_proj: Linear::new(
                    f16(&format!("{p}.self_attn.v_proj.weight"))?,
                    hidden,
                    n_kv * head_dim,
                ),
                o_proj: Linear::new(
                    f16(&format!("{p}.self_attn.o_proj.weight"))?,
                    n_heads * head_dim,
                    hidden,
                ),
                q_norm: RmsNorm::new(
                    f32t(&format!("{p}.self_attn.q_norm.weight"))?,
                    eps,
                    head_dim,
                ),
                k_norm: RmsNorm::new(
                    f32t(&format!("{p}.self_attn.k_norm.weight"))?,
                    eps,
                    head_dim,
                ),
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                hidden,
            };
            let ln2 = RmsNorm::new(
                f32t(&format!("{p}.post_attention_layernorm.weight"))?,
                eps,
                hidden,
            );
            let mlp = Mlp::new(
                Linear::new(f16(&format!("{p}.mlp.gate_proj.weight"))?, hidden, inter),
                Linear::new(f16(&format!("{p}.mlp.up_proj.weight"))?, hidden, inter),
                Linear::new(f16(&format!("{p}.mlp.down_proj.weight"))?, inter, hidden),
            );
            layers.push(Layer {
                ln1,
                attn,
                ln2,
                mlp,
            });
        }
        let norm = RmsNorm::new(f32t("model.norm.weight")?, eps, hidden);
        let rope = Rope::build(dev, max_seq, head_dim, cfg.rope_theta);

        Ok(Qwen3Model {
            dev: dev.clone(),
            embed_w,
            lm_head_w,
            layers,
            norm,
            rope,
            cfg,
        })
    }

    fn lm_head_weight(&self) -> &Tensor {
        self.lm_head_w.as_ref().unwrap_or(&self.embed_w)
    }

    /// Run every decoder block over `x` (`[rows, hidden]`) in place, returning the post-final-norm
    /// hidden states. `positions` is the absolute position of each row's first token for the cache
    /// (prefill: rows are 0..rows at base 0; decode: one row at `base_pos`).
    fn blocks(
        &self,
        rec: &mut Recorder,
        mut x: Tensor,
        kv: &mut KvCache,
        rows: usize,
        base_pos: usize,
    ) -> Tensor {
        let hidden = self.cfg.hidden;
        for (l, layer) in self.layers.iter().enumerate() {
            let h = layer.ln1.forward(rec, &x, rows);
            let attn = if base_pos == 0 && rows > 1 {
                layer.attn.prefill(rec, &h, &self.rope, kv, l, rows)
            } else {
                layer.attn.decode(rec, &h, &self.rope, kv, l, base_pos)
            };
            let x2 = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
            add(rec, &x, &attn, &x2);
            let h2 = layer.ln2.forward(rec, &x2, rows);
            let m = layer.mlp.forward(rec, &h2, rows);
            let x3 = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
            add(rec, &x2, &m, &x3);
            x = x3;
        }
        self.norm.forward(rec, &x, rows)
    }

    /// Prefill: embed `ids`, run all blocks (filling the KV cache for positions `0..len`), and
    /// return the logits (`[vocab]`) for the LAST position.
    pub async fn forward_prefill(&self, ids: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        let n = ids.len();
        let mut rec = Recorder::new(&self.dev);
        let ids_t = Tensor::from_u32(&self.dev, &[n], ids);
        let x = Tensor::empty(&self.dev, &[n, hidden], DType::F32);
        embedding_f16(&mut rec, &ids_t, &self.embed_w, &x, n, hidden);
        let xn = self.blocks(&mut rec, x, kv, n, 0);

        let last = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        copy_range(&mut rec, &xn, &last, (n - 1) * hidden, 0, hidden);
        let logits = Tensor::empty(&self.dev, &[1, vocab], DType::F32);
        linear_f16w(
            &mut rec,
            &last,
            self.lm_head_weight(),
            &logits,
            1,
            hidden,
            vocab,
        );
        rec.submit();
        logits.to_f32().await
    }

    /// Decode one token `token` at absolute position `pos`, returning logits (`[vocab]`).
    pub async fn forward_decode(
        &self,
        token: u32,
        pos: usize,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        let mut rec = Recorder::new(&self.dev);
        let ids_t = Tensor::from_u32(&self.dev, &[1], &[token]);
        let x = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        embedding_f16(&mut rec, &ids_t, &self.embed_w, &x, 1, hidden);
        let xn = self.blocks(&mut rec, x, kv, 1, pos);

        let logits = Tensor::empty(&self.dev, &[1, vocab], DType::F32);
        linear_f16w(
            &mut rec,
            &xn,
            self.lm_head_weight(),
            &logits,
            1,
            hidden,
            vocab,
        );
        rec.submit();
        logits.to_f32().await
    }
}
