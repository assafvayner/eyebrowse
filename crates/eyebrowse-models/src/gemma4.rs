//! Gemma 4 dense decoder.
//!
//! Differs from the shared [`crate::Decoder`] (Qwen3/Llama/Mistral) in several Gemma-specific ways:
//! per-layer head_dim (local "sliding_attention" layers use `head_dim`; global "full_attention"
//! layers use `global_head_dim`), attention `scale = 1.0`, per-head Q/K-RMSNorm plus a *weightless*
//! V-RMSNorm, two RoPE instances (local full-rotary + global partial-rotary "proportional"),
//! GeGLU MLP, sandwich norms around each sub-block, a per-layer residual `layer_scalar`, an
//! embedding `×√hidden` scale, and a CPU `final_logit_softcapping` on the LM head output.

use std::sync::Arc;

use eyebrowse_core::{DType, EyebrowseError, Result};
use eyebrowse_gpu::{add, copy_range, Device, Recorder, Tensor};
use eyebrowse_kernels::{embedding_f16, linear_f16w, mul_scalar};
use eyebrowse_load::{Config, WeightSource};
use eyebrowse_nn::{Attention, KvCache, Linear, Mlp, RmsNorm, Rope};

use crate::upload::{raw_to_f32, upload_f16, upload_f32};

/// Which RoPE/head-dim regime a layer belongs to.
#[derive(Clone, Copy, PartialEq)]
enum LayerKind {
    Local,
    Global,
}

struct Layer {
    input_ln: RmsNorm,
    attn: Attention,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
    mlp: Mlp,
    /// Residual scale read from this layer's `layer_scalar [1]` tensor.
    layer_scalar: f32,
    kind: LayerKind,
}

pub struct Gemma4 {
    dev: Arc<Device>,
    /// Packed-f16 embedding table `[vocab, hidden]`, also the (tied) LM-head weight.
    embed_w: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope_local: Rope,
    rope_global: Rope,
    /// Per-layer KV head_dim, for sizing the KV cache.
    head_dims: Vec<usize>,
    embed_scale: f32,
    logit_cap: f32,
    pub cfg: Config,
}

/// Free-function loader matching the `qwen3::load` / `mistral::load` convention used by the
/// `load_model` selector.
pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<Gemma4> {
    Gemma4::load(dev, src, max_seq)
}

fn cfg_u64(extra: &serde_json::Value, key: &str) -> Option<usize> {
    extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

impl Gemma4 {
    /// Build a KV cache sized for this model and the given max sequence length, honoring the
    /// per-layer head_dim (global layers are wider than local ones).
    pub fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        KvCache::new_per_layer(&self.dev, &self.head_dims, max_seq, self.cfg.n_kv_heads)
    }

    /// Load all weights from a `WeightSource` and precompute both RoPE tables up to `max_seq`.
    pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<Self> {
        let cfg = src.config().clone();
        let extra = &cfg.extra;

        // Reject the configurations this dense loader does not implement.
        if cfg_u64(extra, "hidden_size_per_layer_input").unwrap_or(0) != 0 {
            return Err(EyebrowseError::UnsupportedConfig(
                "gemma4 hidden_size_per_layer_input != 0 (per-layer input embeddings)".to_string(),
            ));
        }
        if extra
            .get("num_experts")
            .map(|v| !v.is_null())
            .unwrap_or(false)
            && cfg_u64(extra, "num_experts").unwrap_or(0) != 0
        {
            return Err(EyebrowseError::UnsupportedConfig(
                "gemma4 MoE (num_experts) is not supported".to_string(),
            ));
        }
        if extra
            .get("enable_moe_block")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Err(EyebrowseError::UnsupportedConfig(
                "gemma4 MoE (enable_moe_block) is not supported".to_string(),
            ));
        }

        let (hidden, n_heads, n_kv, inter, eps) = (
            cfg.hidden,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.intermediate,
            cfg.rms_eps,
        );
        let local_hd = cfg.head_dim;
        let global_hd = cfg_u64(extra, "global_head_dim").ok_or_else(|| {
            EyebrowseError::Load("gemma4 config missing global_head_dim".to_string())
        })?;

        let layer_types = extra
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| EyebrowseError::Load("gemma4 config missing layer_types".to_string()))?;

        let f16 = |name: &str| -> Result<Tensor> { upload_f16(dev, &src.raw(name)?) };
        let f32t = |name: &str| -> Result<Tensor> { upload_f32(dev, &src.raw(name)?) };
        // Decode the `[1]` residual scale on the CPU; no GPU round-trip needed at load time.
        let scalar = |name: &str| -> Result<f32> { Ok(raw_to_f32(&src.raw(name)?)?[0]) };

        let embed_w = f16("model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut head_dims = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let is_global = layer_types
                .get(l)
                .and_then(|v| v.as_str())
                .map(|s| s == "full_attention")
                .unwrap_or(false);
            let (kind, hd) = if is_global {
                (LayerKind::Global, global_hd)
            } else {
                (LayerKind::Local, local_hd)
            };
            head_dims.push(hd);

            let p = format!("model.layers.{l}");
            let q_proj = Linear::new(
                f16(&format!("{p}.self_attn.q_proj.weight"))?,
                hidden,
                n_heads * hd,
            );
            let k_proj = Linear::new(
                f16(&format!("{p}.self_attn.k_proj.weight"))?,
                hidden,
                n_kv * hd,
            );
            let v_proj = Linear::new(
                f16(&format!("{p}.self_attn.v_proj.weight"))?,
                hidden,
                n_kv * hd,
            );
            let o_proj = Linear::new(
                f16(&format!("{p}.self_attn.o_proj.weight"))?,
                n_heads * hd,
                hidden,
            );
            let q_norm = RmsNorm::new(f32t(&format!("{p}.self_attn.q_norm.weight"))?, eps, hd);
            let k_norm = RmsNorm::new(f32t(&format!("{p}.self_attn.k_norm.weight"))?, eps, hd);
            // Gemma 4 has no v_norm tensor: the V-RMSNorm is weightless (unit-weight vector).
            let v_norm = RmsNorm::new(Tensor::from_f32(dev, &[hd], &vec![1.0; hd]), eps, hd);

            let attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Some(q_norm),
                k_norm: Some(k_norm),
                v_norm: Some(v_norm),
                n_heads,
                n_kv_heads: n_kv,
                head_dim: hd,
                hidden,
                scale: 1.0,
            };

            let mlp = Mlp::geglu(
                Linear::new(f16(&format!("{p}.mlp.gate_proj.weight"))?, hidden, inter),
                Linear::new(f16(&format!("{p}.mlp.up_proj.weight"))?, hidden, inter),
                Linear::new(f16(&format!("{p}.mlp.down_proj.weight"))?, inter, hidden),
            );

            layers.push(Layer {
                input_ln: RmsNorm::new(f32t(&format!("{p}.input_layernorm.weight"))?, eps, hidden),
                attn,
                post_attn_ln: RmsNorm::new(
                    f32t(&format!("{p}.post_attention_layernorm.weight"))?,
                    eps,
                    hidden,
                ),
                pre_ff_ln: RmsNorm::new(
                    f32t(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                    eps,
                    hidden,
                ),
                post_ff_ln: RmsNorm::new(
                    f32t(&format!("{p}.post_feedforward_layernorm.weight"))?,
                    eps,
                    hidden,
                ),
                mlp,
                layer_scalar: scalar(&format!("{p}.layer_scalar"))?,
                kind,
            });
        }

        let norm = RmsNorm::new(f32t("model.norm.weight")?, eps, hidden);

        // RoPE: local layers use full rotary at the local head_dim; global layers use a partial
        // ("proportional") rotary at the global head_dim where only the first `rope_angles`
        // frequency pairs are active.
        let rp = &extra["rope_parameters"];
        let theta_local = rp["sliding_attention"]["rope_theta"]
            .as_f64()
            .ok_or_else(|| {
                EyebrowseError::Load(
                    "gemma4 missing rope_parameters.sliding_attention.rope_theta".to_string(),
                )
            })? as f32;
        let theta_global = rp["full_attention"]["rope_theta"].as_f64().ok_or_else(|| {
            EyebrowseError::Load(
                "gemma4 missing rope_parameters.full_attention.rope_theta".to_string(),
            )
        })? as f32;
        let partial_rotary_factor = rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .ok_or_else(|| {
                EyebrowseError::Load(
                    "gemma4 missing rope_parameters.full_attention.partial_rotary_factor"
                        .to_string(),
                )
            })?;
        let rope_angles = ((partial_rotary_factor * global_hd as f64) / 2.0).floor() as usize;

        let rope_local = Rope::build(dev, max_seq, local_hd, theta_local);
        let rope_global = Rope::build_partial(dev, max_seq, global_hd, rope_angles, theta_global);

        let embed_scale = (hidden as f32).sqrt();
        let logit_cap = extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        Ok(Gemma4 {
            dev: dev.clone(),
            embed_w,
            layers,
            norm,
            rope_local,
            rope_global,
            head_dims,
            embed_scale,
            logit_cap,
            cfg,
        })
    }

    fn rope_for(&self, kind: LayerKind) -> &Rope {
        match kind {
            LayerKind::Local => &self.rope_local,
            LayerKind::Global => &self.rope_global,
        }
    }

    /// Run every decoder block over `x` (`[rows, hidden]`), returning post-final-norm hidden states.
    /// `base_pos` is the absolute position of the first row (prefill: 0; decode: `pos`).
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
            let rope = self.rope_for(layer.kind);

            // Attention sub-block: input_ln -> attn -> post_attn_ln -> residual add.
            let h = layer.input_ln.forward(rec, &x, rows);
            let a = if base_pos == 0 && rows > 1 {
                layer.attn.prefill(rec, &h, rope, kv, l, rows)
            } else {
                layer.attn.decode(rec, &h, rope, kv, l, base_pos)
            };
            let a = layer.post_attn_ln.forward(rec, &a, rows);
            let x2 = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
            add(rec, &x, &a, &x2);

            // Feed-forward sub-block: pre_ff_ln -> mlp -> post_ff_ln -> residual add.
            let h2 = layer.pre_ff_ln.forward(rec, &x2, rows);
            let m = layer.mlp.forward(rec, &h2, rows);
            let m = layer.post_ff_ln.forward(rec, &m, rows);
            let x3 = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
            add(rec, &x2, &m, &x3);

            // Per-layer residual scale.
            let xs = Tensor::empty(&self.dev, &[rows, hidden], DType::F32);
            mul_scalar(rec, &x3, &xs, rows * hidden, layer.layer_scalar);
            x = xs;
        }
        self.norm.forward(rec, &x, rows)
    }

    /// Apply Gemma's CPU softcap to logits in place (no-op when `logit_cap <= 0`).
    fn softcap(&self, logits: &mut [f32]) {
        if self.logit_cap > 0.0 {
            let cap = self.logit_cap;
            for v in logits.iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }
    }

    /// Prefill: embed `ids`, run all blocks, and return softcapped logits (`[vocab]`) for the
    /// LAST position.
    pub async fn forward_prefill(&self, ids: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        let n = ids.len();
        let mut rec = Recorder::new(&self.dev);
        let ids_t = Tensor::from_u32(&self.dev, &[n], ids);
        let emb = Tensor::empty(&self.dev, &[n, hidden], DType::F32);
        embedding_f16(&mut rec, &ids_t, &self.embed_w, &emb, n, hidden);
        let x = Tensor::empty(&self.dev, &[n, hidden], DType::F32);
        mul_scalar(&mut rec, &emb, &x, n * hidden, self.embed_scale);

        let xn = self.blocks(&mut rec, x, kv, n, 0);

        let last = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        copy_range(&mut rec, &xn, &last, (n - 1) * hidden, 0, hidden);
        let logits = Tensor::empty(&self.dev, &[1, vocab], DType::F32);
        linear_f16w(&mut rec, &last, &self.embed_w, &logits, 1, hidden, vocab);
        rec.submit();
        let mut out = logits.to_f32().await?;
        self.softcap(&mut out);
        Ok(out)
    }

    /// Decode one token `token` at absolute position `pos`, returning softcapped logits (`[vocab]`).
    pub async fn forward_decode(
        &self,
        token: u32,
        pos: usize,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        let mut rec = Recorder::new(&self.dev);
        let ids_t = Tensor::from_u32(&self.dev, &[1], &[token]);
        let emb = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        embedding_f16(&mut rec, &ids_t, &self.embed_w, &emb, 1, hidden);
        let x = Tensor::empty(&self.dev, &[1, hidden], DType::F32);
        mul_scalar(&mut rec, &emb, &x, hidden, self.embed_scale);

        let xn = self.blocks(&mut rec, x, kv, 1, pos);

        let logits = Tensor::empty(&self.dev, &[1, vocab], DType::F32);
        linear_f16w(&mut rec, &xn, &self.embed_w, &logits, 1, hidden, vocab);
        rec.submit();
        let mut out = logits.to_f32().await?;
        self.softcap(&mut out);
        Ok(out)
    }
}
