//! GGUF end-to-end validation: a Q8_0 GGUF of Qwen3-0.6B must generate (near-)identically to the
//! safetensors path (matching the committed HF golden), and a Q4_K_M GGUF must at least reproduce
//! the high-margin first token. Skips if the fixtures are absent (run scripts/make-gguf-fixtures.sh).

use std::path::Path;

fn repo(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn ids(v: &serde_json::Value, key: &str) -> Vec<u32> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn golden() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(repo("golden/qwen3-golden.json")).unwrap()).unwrap()
}

#[test]
fn gguf_q8_0_matches_safetensors_golden() {
    use eyebrowse_load::WeightSource;
    let gguf = repo("models/gguf/qwen3-0.6b-q8_0.gguf");
    if !gguf.exists() {
        eprintln!("SKIP: run scripts/make-gguf-fixtures.sh");
        return;
    }
    let g = golden();
    let input_ids = ids(&g, "input_ids");
    let top1 = g["first_logits_top10_ids"][0].as_u64().unwrap() as u32;
    let cont = ids(&g, "greedy_continuation_ids");

    let dev = pollster::block_on(eyebrowse_gpu::Device::new()).unwrap();
    let src = eyebrowse_load::GgufSource::from_path(&gguf).unwrap();

    // The GGUF config must map to the same structural fields as the safetensors config.
    let st = eyebrowse_load::SafeTensorsSource::from_dir(&repo("models/qwen3-0.6b")).unwrap();
    let (a, b) = (src.config(), st.config());
    assert_eq!(a.arch, "qwen3");
    assert_eq!(
        (a.hidden, a.n_layers, a.n_heads, a.n_kv_heads, a.head_dim, a.intermediate, a.vocab),
        (b.hidden, b.n_layers, b.n_heads, b.n_kv_heads, b.head_dim, b.intermediate, b.vocab)
    );

    let model = eyebrowse_models::load_model(&dev, &src, 256).unwrap();
    let out = pollster::block_on(eyebrowse::greedy_generate(&model, &input_ids, cont.len(), 256)).unwrap();
    assert_eq!(out[0], top1, "q8_0 first token");
    let matched = out.iter().zip(&cont).take_while(|(x, y)| x == y).count();
    println!("q8_0 continuation match {}/{}: {out:?}", matched, cont.len());
    assert!(matched >= cont.len() - 2, "q8_0 diverged: {matched}/{}", cont.len());
}

#[test]
fn gguf_q4_k_m_smoke() {
    let gguf = repo("models/gguf/qwen3-0.6b-q4_k_m.gguf");
    if !gguf.exists() {
        eprintln!("SKIP: run scripts/make-gguf-fixtures.sh");
        return;
    }
    let g = golden();
    let input_ids = ids(&g, "input_ids");
    let top1 = g["first_logits_top10_ids"][0].as_u64().unwrap() as u32;
    let cont = ids(&g, "greedy_continuation_ids");

    let dev = pollster::block_on(eyebrowse_gpu::Device::new()).unwrap();
    let src = eyebrowse_load::GgufSource::from_path(&gguf).unwrap();
    let model = eyebrowse_models::load_model(&dev, &src, 256).unwrap();
    let out = pollster::block_on(eyebrowse::greedy_generate(&model, &input_ids, cont.len(), 256)).unwrap();
    let matched = out.iter().zip(&cont).take_while(|(x, y)| x == y).count();
    println!("q4_k_m continuation match {}/{}: {out:?}", matched, cont.len());
    // Q4_K_M is lossy; the high-margin first token ("Paris") should still survive quantization.
    assert_eq!(out[0], top1, "q4_k_m first token");
}
