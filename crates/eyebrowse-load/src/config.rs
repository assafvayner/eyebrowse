//! Normalized model configuration, decoupled from any single source format.

use eyebrowse_core::{EyebrowseError, Result};
use serde::Deserialize;

/// Architecture-independent description of a transformer model.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub arch: String,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub max_seq: usize,
    /// The full source `config.json` as a JSON value, for architecture-specific fields not
    /// captured by the typed fields above (e.g. Gemma 4's `layer_types`, `rope_parameters`).
    /// `serde_json::Value::Null` when the source has no JSON config (e.g. GGUF).
    pub extra: serde_json::Value,
}

/// transformers v5 nests RoPE settings here instead of using a top-level `rope_theta`.
#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: Option<f64>,
}

#[derive(Deserialize)]
struct HfConfig {
    model_type: Option<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    // Old configs use a top-level `rope_theta`; transformers v5 nests it under `rope_parameters`.
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    tie_word_embeddings: bool,
    max_position_embeddings: usize,
}

/// Parse a Hugging Face `config.json` into a normalized [`Config`].
pub fn config_from_hf_json(s: &str) -> Result<Config> {
    let hf: HfConfig =
        serde_json::from_str(s).map_err(|e| EyebrowseError::Load(format!("config.json: {e}")))?;
    let extra: serde_json::Value =
        serde_json::from_str(s).map_err(|e| EyebrowseError::Load(format!("config.json: {e}")))?;

    let arch = hf.model_type.unwrap_or_else(|| "unknown".to_string());
    let n_heads = hf.num_attention_heads;
    let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / n_heads);
    // Resolve RoPE theta from either the top-level field or the v5 `rope_parameters` block.
    let rope_theta = hf
        .rope_theta
        .or(hf.rope_parameters.and_then(|r| r.rope_theta))
        .unwrap_or(10000.0);

    Ok(Config {
        arch,
        hidden: hf.hidden_size,
        n_layers: hf.num_hidden_layers,
        n_heads,
        n_kv_heads: hf.num_key_value_heads.unwrap_or(n_heads),
        head_dim,
        intermediate: hf.intermediate_size,
        vocab: hf.vocab_size,
        rms_eps: hf.rms_norm_eps as f32,
        rope_theta: rope_theta as f32,
        tie_word_embeddings: hf.tie_word_embeddings,
        max_seq: hf.max_position_embeddings,
        extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_defaults_to_hidden_over_heads() {
        let json = r#"{
            "model_type": "test",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 1024,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "max_position_embeddings": 2048
        }"#;
        let cfg = config_from_hf_json(json).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.n_kv_heads, 8);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn tolerates_float_eps_and_integer_theta() {
        let json = r#"{
            "model_type": "test",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 1024,
            "vocab_size": 100,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 2048
        }"#;
        let cfg = config_from_hf_json(json).unwrap();
        assert_eq!(cfg.n_kv_heads, 2);
        assert!(cfg.tie_word_embeddings);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn reads_rope_theta_from_v5_rope_parameters() {
        // transformers v5 nests rope_theta under rope_parameters and omits the top-level field.
        let json = r#"{
            "model_type": "mistral",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "vocab_size": 320,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
            "max_position_embeddings": 128
        }"#;
        let cfg = config_from_hf_json(json).unwrap();
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(cfg.arch, "mistral");
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(config_from_hf_json("{ not json").is_err());
    }
}
