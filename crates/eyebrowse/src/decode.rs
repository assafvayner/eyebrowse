//! The greedy autoregressive loop used by the `Generator`. Tokenizer-free: it operates on
//! token ids, leaving encode/decode to the caller.
//!
//! Greedy selection runs on the GPU (`*_argmax`): each step reads back a single token id rather
//! than the full vocab logits, so the per-token GPU→CPU transfer is constant instead of `O(vocab)`.

use eyebrowse_core::Result;
use eyebrowse_models::LanguageModel;

/// Greedy-generate `max_new` tokens after `input_ids`, returning the generated ids. Allocates a
/// fresh KV cache sized to `max_seq` (must be >= `input_ids.len() + max_new`).
pub async fn greedy_generate(
    model: &LanguageModel,
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
    let mut next = model.prefill_argmax(input_ids, &mut kv).await?;
    let mut out = vec![next];
    for step in 1..max_new {
        // The just-produced token sits at absolute position (prompt_len + step - 1).
        let pos = input_ids.len() + step - 1;
        next = model.decode_argmax(next, pos, &mut kv).await?;
        out.push(next);
    }
    Ok(out)
}
