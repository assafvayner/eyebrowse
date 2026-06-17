//! The standard decoder block (Qwen3 / Llama / Mistral) and its loader.
//!
//! Per block: RMSNorm -> attention (GQA + optional QK-norm + RoPE) -> residual -> RMSNorm ->
//! SwiGLU MLP -> residual. The per-architecture modules (`qwen3`, `mistral`) call [`load`] and
//! differ only in whether per-head QK-RMSNorm is attached. The driver in [`crate::model`] runs the
//! resulting [`LanguageModel`].

use std::sync::Arc;

use eyebrowse_core::{DType, Result};
use eyebrowse_gpu::{add, Device, Recorder, Tensor};
use eyebrowse_load::WeightSource;
use eyebrowse_nn::{Attention, KvCache, Linear, Mlp, RmsNorm, Rope};

use crate::model::{Block, LanguageModel};
use crate::upload::{upload_f16, upload_f32};

/// Per-architecture knobs for [`load`].
pub struct DecoderOpts {
    /// Whether the architecture has per-head QK-RMSNorm (Qwen3 yes; Llama/Mistral no).
    pub has_qk_norm: bool,
}

/// A standard pre-norm transformer block with a SwiGLU MLP and a shared RoPE table.
struct StandardBlock {
    ln1: RmsNorm,
    attn: Attention,
    ln2: RmsNorm,
    mlp: Mlp,
    rope: Arc<Rope>,
}

impl Block for StandardBlock {
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
        let h = self.ln1.forward(rec, x, rows);
        let a = self
            .attn
            .forward(rec, &h, &self.rope, kv, layer, rows, base_pos);
        let x2 = Tensor::empty(x.device(), &[rows, hidden], DType::F32);
        add(rec, x, &a, &x2);
        let h2 = self.ln2.forward(rec, &x2, rows);
        let m = self.mlp.forward(rec, &h2, rows);
        let x3 = Tensor::empty(x.device(), &[rows, hidden], DType::F32);
        add(rec, &x2, &m, &x3);
        x3
    }
}

/// Load a Qwen3/Llama/Mistral-family model and precompute RoPE up to `max_seq`.
pub fn load(
    dev: &Arc<Device>,
    src: &dyn WeightSource,
    max_seq: usize,
    opts: DecoderOpts,
) -> Result<LanguageModel> {
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

    let rope = Arc::new(Rope::build(dev, max_seq, head_dim, cfg.rope_theta));

    let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        let ln1 = RmsNorm::new(f32t(&format!("{p}.input_layernorm.weight"))?, eps, hidden);
        let (q_norm, k_norm) = if opts.has_qk_norm {
            (
                Some(RmsNorm::new(
                    f32t(&format!("{p}.self_attn.q_norm.weight"))?,
                    eps,
                    head_dim,
                )),
                Some(RmsNorm::new(
                    f32t(&format!("{p}.self_attn.k_norm.weight"))?,
                    eps,
                    head_dim,
                )),
            )
        } else {
            (None, None)
        };
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
            q_norm,
            k_norm,
            v_norm: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            hidden,
            scale: 1.0 / (head_dim as f32).sqrt(),
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
        blocks.push(Box::new(StandardBlock {
            ln1,
            attn,
            ln2,
            mlp,
            rope: rope.clone(),
        }));
    }
    let norm = RmsNorm::new(f32t("model.norm.weight")?, eps, hidden);

    Ok(LanguageModel {
        dev: dev.clone(),
        embed_w,
        lm_head_w,
        blocks,
        norm,
        embed_scale: None,
        logit_softcap: None,
        head_dims: vec![head_dim; cfg.n_layers],
        n_kv_heads: n_kv,
        cfg,
    })
}
