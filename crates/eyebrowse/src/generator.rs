use std::path::Path;

use eyebrowse_core::Result;
use eyebrowse_gpu::Device;
use eyebrowse_load::{decode, encode, load_tokenizer, SafeTensorsSource};
use eyebrowse_models::Qwen3Model;
use tokenizers::Tokenizer;

/// A text-generation engine: a loaded model + tokenizer + a fixed KV-cache capacity. Greedy decode.
pub struct Generator {
    model: Qwen3Model,
    tok: Tokenizer,
    max_seq: usize,
}

impl Generator {
    /// Load a model directory (HF `config.json` + `model.safetensors` + `tokenizer.json`) natively.
    /// `max_seq` bounds prompt + generated tokens (sizes the KV cache and RoPE tables).
    pub async fn load(model_dir: &str, max_seq: usize) -> Result<Self> {
        let dir = Path::new(model_dir);
        let src = SafeTensorsSource::from_dir(dir)?;
        let dev = Device::new().await?;
        let model = Qwen3Model::load(&dev, &src, max_seq)?;
        let tok = load_tokenizer(&dir.join("tokenizer.json"))?;
        Ok(Generator { model, tok, max_seq })
    }

    /// Greedy-generate `max_new` tokens after `input_ids`, returning the generated ids.
    pub async fn generate_ids(&self, input_ids: &[u32], max_new: usize) -> Result<Vec<u32>> {
        assert!(
            input_ids.len() + max_new <= self.max_seq,
            "prompt ({}) + max_new ({}) exceeds max_seq ({})",
            input_ids.len(),
            max_new,
            self.max_seq
        );
        let mut kv = self.model.new_kv_cache(self.max_seq);
        let logits = self.model.forward_prefill(input_ids, &mut kv).await?;
        let mut next = argmax(&logits);
        let mut out = vec![next];
        let mut pos = input_ids.len();
        for _ in 1..max_new {
            let logits = self.model.forward_decode(next, pos, &mut kv).await?;
            pos += 1;
            next = argmax(&logits);
            out.push(next);
        }
        Ok(out)
    }

    /// Tokenize `prompt`, greedy-generate `max_new` tokens, and detokenize the continuation.
    pub async fn generate(&self, prompt: &str, max_new: usize) -> Result<String> {
        let ids = encode(&self.tok, prompt)?;
        let out = self.generate_ids(&ids, max_new).await?;
        decode(&self.tok, &out)
    }
}

/// Index of the maximum logit (greedy next token).
fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}
