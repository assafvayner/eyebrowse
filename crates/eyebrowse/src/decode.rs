//! The greedy autoregressive loop, shared by the native `Generator` and the wasm binding.
//! Tokenizer-free (operates on token ids) so it compiles on both targets.

use eyebrowse_core::Result;
use eyebrowse_models::Qwen3Model;

/// Greedy-generate `max_new` tokens after `input_ids`, returning the generated ids. Allocates a
/// fresh KV cache sized to `max_seq` (must be >= `input_ids.len() + max_new`).
pub async fn greedy_generate(
    model: &Qwen3Model,
    input_ids: &[u32],
    max_new: usize,
    max_seq: usize,
) -> Result<Vec<u32>> {
    assert!(
        input_ids.len() + max_new <= max_seq,
        "prompt ({}) + max_new ({}) exceeds max_seq ({})",
        input_ids.len(),
        max_new,
        max_seq
    );
    let mut kv = model.new_kv_cache(max_seq);
    let logits = model.forward_prefill(input_ids, &mut kv).await?;
    let mut next = argmax(&logits);
    let mut out = vec![next];
    let mut pos = input_ids.len();
    for _ in 1..max_new {
        let logits = model.forward_decode(next, pos, &mut kv).await?;
        pos += 1;
        next = argmax(&logits);
        out.push(next);
    }
    Ok(out)
}

/// Index of the maximum logit (greedy next token).
pub(crate) fn argmax(logits: &[f32]) -> u32 {
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
