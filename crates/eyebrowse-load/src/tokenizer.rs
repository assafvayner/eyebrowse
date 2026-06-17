//! Thin wrappers over the `tokenizers` crate.

use std::path::Path;

use eyebrowse_core::{EyebrowseError, Result};
use tokenizers::Tokenizer;

/// Load a tokenizer from a `tokenizer.json` file.
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| EyebrowseError::Load(format!("loading tokenizer {}: {e}", path.display())))
}

/// Encode `text` into token ids (no special tokens added).
pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tok
        .encode(text, false)
        .map_err(|e| EyebrowseError::Load(format!("encode: {e}")))?;
    Ok(encoding.get_ids().to_vec())
}

/// Decode token ids back into a string (special tokens skipped).
pub fn decode(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    tok.decode(ids, true)
        .map_err(|e| EyebrowseError::Load(format!("decode: {e}")))
}
