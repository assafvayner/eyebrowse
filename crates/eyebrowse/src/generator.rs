use std::path::Path;

use eyebrowse_core::Result;
use eyebrowse_gpu::Device;
use eyebrowse_load::{decode, encode, load_tokenizer, SafeTensorsSource};
use eyebrowse_models::{load_model, LanguageModel};
use tokenizers::Tokenizer;

use crate::decode::greedy_generate;

/// A native text-generation engine: a loaded model + tokenizer + a fixed KV-cache capacity.
pub struct Generator {
    model: LanguageModel,
    tok: Tokenizer,
    max_seq: usize,
}

impl Generator {
    /// Load a model directory (HF `config.json` + `model.safetensors` + `tokenizer.json`).
    /// `max_seq` bounds prompt + generated tokens (sizes the KV cache and RoPE tables).
    pub async fn load(model_dir: &str, max_seq: usize) -> Result<Self> {
        let dir = Path::new(model_dir);
        let src = SafeTensorsSource::from_dir(dir)?;
        let dev = Device::new().await?;
        let model = load_model(&dev, &src, max_seq)?;
        let tok = load_tokenizer(&dir.join("tokenizer.json"))?;
        Ok(Generator {
            model,
            tok,
            max_seq,
        })
    }

    /// Greedy-generate `max_new` tokens after `input_ids`, returning the generated ids.
    pub async fn generate_ids(&self, input_ids: &[u32], max_new: usize) -> Result<Vec<u32>> {
        greedy_generate(&self.model, input_ids, max_new, self.max_seq).await
    }

    /// Tokenize `prompt`, greedy-generate `max_new` tokens, and detokenize the continuation.
    pub async fn generate(&self, prompt: &str, max_new: usize) -> Result<String> {
        let ids = encode(&self.tok, prompt)?;
        let out = self.generate_ids(&ids, max_new).await?;
        decode(&self.tok, &out)
    }
}
