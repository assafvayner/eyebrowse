//! Native greedy generation demo.
//! Usage: cargo run -p eyebrowse --release --example generate -- [model_dir] [prompt] [max_new]

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/qwen3-0.6b").to_string()
    });
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".to_string());
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let gen = pollster::block_on(eyebrowse::Generator::load(&dir, 512)).expect("load");
    let text = pollster::block_on(gen.generate(&prompt, max_new)).expect("generate");
    println!("PROMPT:  {prompt}");
    println!("OUTPUT: {text}");
}
