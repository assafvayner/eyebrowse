//! Validates the Mistral (optional-QK-norm) path: our last-position logits for a tiny synthetic
//! Mistral must match the HuggingFace reference within rel-L2. Skips if the fixture is absent
//! (generate via `golden/gen_mistral_golden.py`).

use std::path::Path;

fn repo(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
}

#[test]
fn mistral_tiny_logits_match_hf() {
    let dir = repo("models/mistral-tiny");
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: run golden/gen_mistral_golden.py to create models/mistral-tiny");
        return;
    }
    let g: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(repo("golden/mistral-tiny-golden.json")).unwrap())
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
    assert_eq!(src.config().arch, "mistral", "fixture should be a mistral model");
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
    println!("mistral tiny logits rel-L2 = {rel}");
    assert!(rel < 2e-2, "rel-L2 {rel} too high");
}
