//! Gemma 4 dense decoder block and loader.
//!
//! Differs from the standard block ([`crate::decoder`]) in several Gemma-specific ways: per-layer
//! head_dim (local "sliding_attention" layers use `head_dim`; global "full_attention" layers use
//! `global_head_dim`), attention `scale = 1.0`, per-head Q/K-RMSNorm plus a *weightless* V-RMSNorm,
//! two RoPE instances (local full-rotary + global partial-rotary "proportional"), a GeGLU MLP,
//! sandwich norms around each sub-block, and a per-layer residual `layer_scalar`. The embedding
//! `×√hidden` scale and the CPU `final_logit_softcapping` are set as hooks on the [`LanguageModel`].

use std::sync::Arc;

use eyebrowse_core::{DType, EyebrowseError, Result};
use eyebrowse_gpu::{add, Device, Recorder, Tensor};
use eyebrowse_kernels::mul_scalar;
use eyebrowse_load::WeightSource;
use eyebrowse_nn::{Attention, KvCache, Linear, Mlp, RmsNorm, Rope};
use serde::Deserialize;

use crate::model::{Block, LanguageModel};
use crate::upload::{raw_to_f32, upload_f16, upload_f32};

/// Gemma 4's architecture-specific config fields, parsed once from `Config.extra` (the raw
/// `config.json`). Anything not in the normalized [`Config`] lives here; serde ignores the many
/// other keys in the file.
#[derive(Deserialize)]
struct Gemma4Config {
    /// Q/KV head_dim for global ("full_attention") layers; local layers use `Config.head_dim`.
    global_head_dim: usize,
    /// Per-layer attention regime: `"full_attention"` => global, anything else => local.
    layer_types: Vec<String>,
    rope_parameters: Gemma4Rope,
    /// `cap * tanh(x / cap)` applied to LM-head logits; `None`/`0` disables.
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    /// Per-layer input-embedding width (E2B/E4B variants). Nonzero => unsupported by this loader.
    #[serde(default)]
    hidden_size_per_layer_input: usize,
    /// MoE controls; either set => unsupported by this dense-only loader.
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    enable_moe_block: bool,
}

#[derive(Deserialize)]
struct Gemma4Rope {
    sliding_attention: Gemma4RopeEntry,
    full_attention: Gemma4RopeEntry,
}

#[derive(Deserialize)]
struct Gemma4RopeEntry {
    rope_theta: f64,
    /// Fraction of head_dim that is rotated (global "proportional" RoPE); required for global.
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

/// A Gemma 4 block: sandwich norms around attention and GeGLU MLP, with a per-layer residual scale
/// and a per-layer (local or global) RoPE table.
struct GemmaBlock {
    input_ln: RmsNorm,
    attn: Attention,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
    mlp: Mlp,
    /// Residual scale read from this layer's `layer_scalar [1]` tensor.
    layer_scalar: f32,
    rope: Arc<Rope>,
}

impl Block for GemmaBlock {
    fn forward(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        kv: &mut KvCache,
        layer: usize,
        rows: usize,
        base_pos: usize,
    ) -> Tensor {
        let hidden = self.attn.hidden;

        // Attention sub-block: input_ln -> attn -> post_attn_ln -> residual add.
        let h = self.input_ln.forward(rec, x, rows);
        let a = self
            .attn
            .forward(rec, &h, &self.rope, kv, layer, rows, base_pos);
        let a = self.post_attn_ln.forward(rec, &a, rows);
        let x2 = Tensor::empty(x.device(), &[rows, hidden], DType::F32);
        add(rec, x, &a, &x2);

        // Feed-forward sub-block: pre_ff_ln -> mlp -> post_ff_ln -> residual add.
        let h2 = self.pre_ff_ln.forward(rec, &x2, rows);
        let m = self.mlp.forward(rec, &h2, rows);
        let m = self.post_ff_ln.forward(rec, &m, rows);
        let x3 = Tensor::empty(x.device(), &[rows, hidden], DType::F32);
        add(rec, &x2, &m, &x3);

        // Per-layer residual scale.
        let xs = Tensor::empty(x.device(), &[rows, hidden], DType::F32);
        mul_scalar(rec, &x3, &xs, rows * hidden, self.layer_scalar);
        xs
    }
}

/// Load a Gemma 4 dense model and precompute both RoPE tables up to `max_seq`.
pub fn load(dev: &Arc<Device>, src: &dyn WeightSource, max_seq: usize) -> Result<LanguageModel> {
    let cfg = src.config().clone();
    let gc: Gemma4Config = serde_json::from_value(cfg.extra.clone())
        .map_err(|e| EyebrowseError::Load(format!("gemma4 config: {e}")))?;

    // Reject the configurations this dense loader does not implement.
    if gc.hidden_size_per_layer_input != 0 {
        return Err(EyebrowseError::UnsupportedConfig(
            "gemma4 hidden_size_per_layer_input != 0 (per-layer input embeddings)".to_string(),
        ));
    }
    if gc.num_experts.unwrap_or(0) != 0 || gc.enable_moe_block {
        return Err(EyebrowseError::UnsupportedConfig(
            "gemma4 MoE is not supported".to_string(),
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
    let global_hd = gc.global_head_dim;

    let f16 = |name: &str| -> Result<Tensor> { upload_f16(dev, &src.raw(name)?) };
    let f32t = |name: &str| -> Result<Tensor> { upload_f32(dev, &src.raw(name)?) };
    // Decode the `[1]` residual scale on the CPU; no GPU round-trip needed at load time.
    let scalar = |name: &str| -> Result<f32> { Ok(raw_to_f32(&src.raw(name)?)?[0]) };

    let embed_w = f16("model.embed_tokens.weight")?;

    // RoPE: local layers use full rotary at the local head_dim; global layers use a partial
    // ("proportional") rotary at the global head_dim where only the first `rope_angles` frequency
    // pairs are active.
    let theta_local = gc.rope_parameters.sliding_attention.rope_theta as f32;
    let theta_global = gc.rope_parameters.full_attention.rope_theta as f32;
    let partial_rotary_factor = gc
        .rope_parameters
        .full_attention
        .partial_rotary_factor
        .ok_or_else(|| {
            EyebrowseError::Load(
                "gemma4 missing rope_parameters.full_attention.partial_rotary_factor".to_string(),
            )
        })?;
    let rope_angles = ((partial_rotary_factor * global_hd as f64) / 2.0).floor() as usize;
    let rope_local = Arc::new(Rope::build(dev, max_seq, local_hd, theta_local));
    let rope_global = Arc::new(Rope::build_partial(
        dev,
        max_seq,
        global_hd,
        rope_angles,
        theta_global,
    ));

    let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(cfg.n_layers);
    let mut head_dims = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let is_global = gc
            .layer_types
            .get(l)
            .map(|s| s == "full_attention")
            .unwrap_or(false);
        let (hd, rope) = if is_global {
            (global_hd, rope_global.clone())
        } else {
            (local_hd, rope_local.clone())
        };
        head_dims.push(hd);

        let p = format!("model.layers.{l}");
        let q_norm = RmsNorm::new(f32t(&format!("{p}.self_attn.q_norm.weight"))?, eps, hd);
        let k_norm = RmsNorm::new(f32t(&format!("{p}.self_attn.k_norm.weight"))?, eps, hd);
        // Gemma 4 has no v_norm tensor: the V-RMSNorm is weightless (unit-weight vector).
        let v_norm = RmsNorm::new(Tensor::from_f32(dev, &[hd], &vec![1.0; hd]), eps, hd);

        let attn = Attention {
            q_proj: Linear::new(
                f16(&format!("{p}.self_attn.q_proj.weight"))?,
                hidden,
                n_heads * hd,
            ),
            k_proj: Linear::new(
                f16(&format!("{p}.self_attn.k_proj.weight"))?,
                hidden,
                n_kv * hd,
            ),
            v_proj: Linear::new(
                f16(&format!("{p}.self_attn.v_proj.weight"))?,
                hidden,
                n_kv * hd,
            ),
            o_proj: Linear::new(
                f16(&format!("{p}.self_attn.o_proj.weight"))?,
                n_heads * hd,
                hidden,
            ),
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

        blocks.push(Box::new(GemmaBlock {
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
            rope,
        }));
    }

    let norm = RmsNorm::new(f32t("model.norm.weight")?, eps, hidden);

    let logit_cap = gc.final_logit_softcapping.unwrap_or(0.0) as f32;
    Ok(LanguageModel {
        dev: dev.clone(),
        embed_w,
        lm_head_w: None,
        blocks,
        norm,
        embed_scale: Some((hidden as f32).sqrt()),
        logit_softcap: (logit_cap > 0.0).then_some(logit_cap),
        head_dims,
        n_kv_heads: n_kv,
        cfg,
    })
}
