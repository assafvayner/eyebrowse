use eyebrowse_core::DType;
use eyebrowse_gpu::{Recorder, Tensor};
use eyebrowse_kernels::{attn_decode, attn_prefill, kv_write};

use crate::{KvCache, Linear, RmsNorm, Rope};

/// Grouped-query attention with RoPE and *optional* per-head norms. Qwen3 sets `q_norm`/`k_norm`
/// to `Some(..)`; Llama/Mistral leave them `None`. Gemma 4 additionally sets `v_norm` (a weightless
/// RMSNorm) and `scale = 1.0`; Qwen3/Mistral use `scale = 1/√head_dim`.
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub v_norm: Option<RmsNorm>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub scale: f32,
}

impl Attention {
    /// Run attention for `rows` rows whose first row is at absolute position `base_pos`, dispatching
    /// to the batched prefill path for a fresh multi-row prompt and to the single-step decode path
    /// otherwise (incl. a one-token prompt at position 0).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        rope: &Rope,
        kv: &mut KvCache,
        layer: usize,
        rows: usize,
        base_pos: usize,
    ) -> Tensor {
        if base_pos == 0 && rows > 1 {
            self.prefill(rec, x, rope, kv, layer, rows)
        } else {
            self.decode(rec, x, rope, kv, layer, base_pos)
        }
    }

    /// Prefill over a whole prompt. `x` is `[seq, hidden]`; returns `[seq, hidden]`.
    /// Writes the layer's keys/values into `kv` starting at sequence position 0.
    pub fn prefill(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        rope: &Rope,
        kv: &mut KvCache,
        layer: usize,
        seq: usize,
    ) -> Tensor {
        let q = self.q_proj.forward(rec, x, seq);
        let k = self.k_proj.forward(rec, x, seq);
        let v = self.v_proj.forward(rec, x, seq);

        let qn = match &self.q_norm {
            Some(n) => n.forward(rec, &q, seq * self.n_heads),
            None => q,
        };
        let kn = match &self.k_norm {
            Some(n) => n.forward(rec, &k, seq * self.n_kv_heads),
            None => k,
        };
        let v = match &self.v_norm {
            Some(n) => n.forward(rec, &v, seq * self.n_kv_heads),
            None => v,
        };

        let qr = rope.apply(rec, &qn, seq, self.n_heads, 0);
        let kr = rope.apply(rec, &kn, seq, self.n_kv_heads, 0);

        kv_write(
            rec,
            &kv.k[layer],
            &kr,
            self.n_kv_heads,
            seq,
            self.head_dim,
            kv.max_seq,
            0,
        );
        kv_write(
            rec,
            &kv.v[layer],
            &v,
            self.n_kv_heads,
            seq,
            self.head_dim,
            kv.max_seq,
            0,
        );

        let o = Tensor::empty(
            rec.device(),
            &[seq, self.n_heads * self.head_dim],
            DType::F32,
        );
        attn_prefill(
            rec,
            &qr,
            &kr,
            &v,
            &o,
            self.n_heads,
            self.n_kv_heads,
            seq,
            self.head_dim,
            self.scale,
        );
        self.o_proj.forward(rec, &o, seq)
    }

    /// Single decode step at sequence position `pos`. `x` is `[1, hidden]`; returns `[1, hidden]`.
    pub fn decode(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        rope: &Rope,
        kv: &mut KvCache,
        layer: usize,
        pos: usize,
    ) -> Tensor {
        let q = self.q_proj.forward(rec, x, 1);
        let k = self.k_proj.forward(rec, x, 1);
        let v = self.v_proj.forward(rec, x, 1);

        let qn = match &self.q_norm {
            Some(n) => n.forward(rec, &q, self.n_heads),
            None => q,
        };
        let kn = match &self.k_norm {
            Some(n) => n.forward(rec, &k, self.n_kv_heads),
            None => k,
        };
        let v = match &self.v_norm {
            Some(n) => n.forward(rec, &v, self.n_kv_heads),
            None => v,
        };

        let qr = rope.apply(rec, &qn, 1, self.n_heads, pos);
        let kr = rope.apply(rec, &kn, 1, self.n_kv_heads, pos);

        kv_write(
            rec,
            &kv.k[layer],
            &kr,
            self.n_kv_heads,
            1,
            self.head_dim,
            kv.max_seq,
            pos,
        );
        kv_write(
            rec,
            &kv.v[layer],
            &v,
            self.n_kv_heads,
            1,
            self.head_dim,
            kv.max_seq,
            pos,
        );

        let o = Tensor::empty(rec.device(), &[1, self.n_heads * self.head_dim], DType::F32);
        attn_decode(
            rec,
            &qr,
            &kv.k[layer],
            &kv.v[layer],
            &o,
            self.n_heads,
            self.n_kv_heads,
            pos,
            self.head_dim,
            kv.max_seq,
            self.scale,
        );
        self.o_proj.forward(rec, &o, 1)
    }
}
