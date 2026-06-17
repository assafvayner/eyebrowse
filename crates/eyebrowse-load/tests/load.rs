use std::collections::HashMap;
use std::path::PathBuf;

use eyebrowse_load::{
    config_from_hf_json, decode, encode, load_tokenizer, RawDType, SafeTensorsSource, WeightSource,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn parses_qwen3_config_fixture() {
    let path = fixtures_dir().join("qwen3-0.6b-config.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent (run scripts/fetch-fixtures.sh)", path.display());
        return;
    }
    let json = std::fs::read_to_string(&path).unwrap();
    let cfg = config_from_hf_json(&json).unwrap();

    assert_eq!(cfg.arch, "qwen3");
    assert_eq!(cfg.hidden, 1024);
    assert_eq!(cfg.n_layers, 28);
    assert_eq!(cfg.n_heads, 16);
    assert_eq!(cfg.n_kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate, 3072);
    assert_eq!(cfg.vocab, 151936);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.max_seq, 40960);
    assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    assert!((cfg.rms_eps - 1e-6).abs() < 1e-9);
}

#[test]
fn safetensors_round_trip() {
    let dir = std::env::temp_dir().join(format!("eyebrowse-load-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let config = r#"{
        "model_type": "test",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 8,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "max_position_embeddings": 128
    }"#;
    std::fs::write(dir.join("config.json"), config).unwrap();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![-1.5, 0.5];
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();

    let va = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &a_bytes)
        .unwrap();
    let vb =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &b_bytes).unwrap();

    let mut tensors: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    tensors.insert("alpha".to_string(), va);
    tensors.insert("beta".to_string(), vb);
    let serialized = safetensors::serialize(&tensors, &None).unwrap();
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();

    let source = SafeTensorsSource::from_dir(&dir).unwrap();

    let mut names = source.tensor_names();
    names.sort();
    assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);

    let ra = source.raw("alpha").unwrap();
    assert_eq!(ra.dtype, RawDType::F32);
    assert_eq!(ra.shape, vec![2, 3]);
    assert_eq!(ra.bytes, a_bytes);

    let rb = source.raw("beta").unwrap();
    assert_eq!(rb.dtype, RawDType::F32);
    assert_eq!(rb.shape, vec![2]);
    assert_eq!(rb.bytes, b_bytes);

    assert_eq!(source.config().hidden, 4);
    assert!(source.raw("missing").is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tokenizer_round_trip() {
    let path = fixtures_dir().join("qwen3-0.6b-tokenizer.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent (run scripts/fetch-fixtures.sh)", path.display());
        return;
    }
    let tok = load_tokenizer(&path).unwrap();

    let text = "Hello, world!";
    let ids = encode(&tok, text).unwrap();
    assert!(!ids.is_empty());

    let decoded = decode(&tok, &ids).unwrap();
    assert!(
        decoded.contains("Hello, world!"),
        "decoded {decoded:?} should contain {text:?}; ids = {ids:?}"
    );

    eprintln!("tokenizer ids for {text:?}: {ids:?}");
}
