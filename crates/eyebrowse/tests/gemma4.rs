//! Validates the Gemma 4 dense path: our last-position logits (post-softcap) for a tiny synthetic
//! Gemma 4 must match the HuggingFace reference within rel-L2. Skips if the fixture is absent.

use std::path::Path;

fn repo(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

#[test]
fn gemma4_tiny_logits_match_hf() {
    let dir = repo("models/gemma4-tiny");
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: generate models/gemma4-tiny to run this test");
        return;
    }
    let g: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo("golden/gemma4-tiny-golden.json")).unwrap(),
    )
    .unwrap();
    let ids: Vec<u32> = g["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<f32> = g["last_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    use eyebrowse_load::WeightSource;
    let dev = pollster::block_on(eyebrowse_gpu::Device::new()).unwrap();
    let src = eyebrowse_load::SafeTensorsSource::from_dir(&dir).unwrap();
    assert!(
        src.config().arch.starts_with("gemma4"),
        "fixture should be a gemma4 model, got {}",
        src.config().arch
    );
    let model = eyebrowse_models::load_model(&dev, &src, 64).unwrap();
    let mut kv = model.new_kv_cache(64);
    let got = pollster::block_on(model.forward_prefill(&ids, &mut kv)).unwrap();

    assert_eq!(got.len(), want.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in got.iter().zip(&want) {
        let d = (*a - *b) as f64;
        num += d * d;
        den += (*b as f64) * (*b as f64);
    }
    let rel = (num.sqrt() / (den.sqrt() + 1e-12)) as f32;
    println!("gemma4 tiny logits rel-L2 = {rel}");
    assert!(rel < 2e-2, "rel-L2 {rel} too high");
}
