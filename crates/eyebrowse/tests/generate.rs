//! Integration test: native Qwen3-0.6B greedy generation must match the HuggingFace `transformers`
//! golden (committed in `golden/qwen3-golden.json`). Skips gracefully if weights are absent.

use std::path::Path;

fn repo_path(rel: &str) -> std::path::PathBuf {
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

#[test]
fn qwen3_matches_hf_golden() {
    let model_dir = repo_path("models/qwen3-0.6b");
    if !model_dir.join("model.safetensors").exists() {
        eprintln!("SKIP: weights not present at {}", model_dir.display());
        return;
    }
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(repo_path("golden/qwen3-golden.json")).unwrap())
            .unwrap();
    let input_ids = ids(&golden, "input_ids");
    let top1 = golden["first_logits_top10_ids"][0].as_u64().unwrap() as u32;
    let cont = ids(&golden, "greedy_continuation_ids");

    let gen = pollster::block_on(eyebrowse::Generator::load(model_dir.to_str().unwrap(), 256)).unwrap();
    let out = pollster::block_on(gen.generate_ids(&input_ids, cont.len())).unwrap();

    // The first generated token has a huge logit margin (Paris); it must match exactly.
    assert_eq!(out[0], top1, "first-token argmax mismatch: got {}, want {}", out[0], top1);

    let matched = out.iter().zip(&cont).take_while(|(a, b)| a == b).count();
    println!("continuation prefix match {}/{}", matched, cont.len());
    println!("golden: {cont:?}");
    println!("ours:   {out:?}");
    // f16 weights vs HF f32: high-margin factual tokens should reproduce; allow minor late drift.
    assert!(
        matched >= cont.len() - 2,
        "continuation diverged: only {}/{} tokens matched",
        matched,
        cont.len()
    );
}
