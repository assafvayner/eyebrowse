//! Browser entry point. The host (JS) fetches `config.json` + `model.safetensors` and the input
//! token ids, hands them in, and gets back the greedily generated ids. Tokenization stays on the
//! JS side (the `tokenizers` onig dep does not build on wasm), keeping this path id-in / id-out.

use wasm_bindgen::prelude::*;

use eyebrowse_gpu::Device;
use eyebrowse_load::SafeTensorsSource;
use eyebrowse_models::Qwen3Model;

use crate::decode::greedy_generate;

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Build the model from in-memory bytes and greedily generate `max_new` tokens after `input_ids`.
/// `weights` is the raw `model.safetensors` blob; it is freed before generation to bound wasm heap.
#[wasm_bindgen]
pub async fn generate_ids(
    config_json: String,
    weights: Vec<u8>,
    input_ids: Vec<u32>,
    max_new: usize,
    max_seq: usize,
) -> std::result::Result<Vec<u32>, JsValue> {
    console_error_panic_hook::set_once();
    let src = SafeTensorsSource::from_bytes(&config_json, weights).map_err(err)?;
    let dev = Device::new().await.map_err(err)?;
    let model = Qwen3Model::load(&dev, &src, max_seq).map_err(err)?;
    // Weights are now resident on the GPU; free the ~GB of safetensors bytes from the wasm heap.
    drop(src);
    greedy_generate(&model, &input_ids, max_new, max_seq)
        .await
        .map_err(err)
}
