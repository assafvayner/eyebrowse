# eyebrowse

A lean, **extensible** model runtime written in Rust that runs neural networks on **WebGPU** via
[`wgpu`](https://github.com/gfx-rs/wgpu) (Metal/Vulkan/DX12). Kernels are hand-written WGSL; the
design favors a small set of composable primitives so adding a new model architecture is a
self-contained module, not a rewrite.

It runs **Qwen3** and **Mistral / Llama**-family text models, loaded from **safetensors or GGUF**
(incl. Q8_0 / Q4_K_M quantization). Generation matches HuggingFace `transformers` token-for-token.

```
$ cargo run -p eyebrowse --release --example generate
PROMPT:  The capital of France is
OUTPUT:  Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China
```

## Status

Implemented and validated (native, Metal):

- ✅ **Qwen3-0.6B** — matches the HF greedy golden 20/20.
- ✅ **Mistral / Llama** family (QK-norm is optional; one shared `Decoder`, arch-selected loader) — logits match HF (rel-L2 ~4e-4).
- ✅ **GGUF loading** (Q8_0 / Q4_K / Q6_K / F16 / F32, dequantized on the CPU into the f16 upload path) — a Q8_0 GGUF of Qwen3-0.6B matches the safetensors golden 20/20.

## Architecture

A Cargo workspace, layered bottom-up. Each crate has one responsibility:

| Crate | Responsibility |
|---|---|
| `eyebrowse-core` | Shared `DType` + the crate-wide error type. |
| `eyebrowse-gpu` | `wgpu` device, `Tensor` (a handle over a GPU buffer), a command `Recorder`, and the kernel-dispatch helper. |
| `eyebrowse-kernels` | Hand-written WGSL compute kernels (GEMM, RMSNorm, RoPE, flash-attention, SwiGLU, embedding, KV-cache write, argmax) + native correctness tests. |
| `eyebrowse-nn` | Composable primitives: `Linear`, `RmsNorm`, `Rope`, `Attention` (GQA + optional QK-RMSNorm + KV cache), `Mlp` (SwiGLU), `Embedding`. |
| `eyebrowse-load` | `WeightSource` trait + **safetensors and GGUF** loaders (GGUF dequant for Q8_0/Q4_K/Q6_K), normalized HF `config.json`, and tokenizer. |
| `eyebrowse-models` | A shared `Decoder` + thin per-architecture loaders (`qwen3`, `mistral`) selected by config arch via `load_model`. |
| `eyebrowse` | The generation runtime: greedy prefill/decode loop and the `Generator`. |

### Key design points

- **Eager execution with batched submit.** A model's `forward` is plain Rust calling kernel
  functions; a whole step records its dispatches into one command buffer and submits once,
  attacking the per-dispatch overhead that dominates WebGPU inference.
- **f16 weights, f32 compute.** Weights are stored as packed-`u32` f16 and unpacked in-kernel with
  `unpack2x16float` (no `shader-f16` feature needed → portable across WebGPU backends).
- **Fixed KV cache.** Allocated up front (no mid-decode growth), seq-major `[max_seq, kv_heads, head_dim]`.
- **Native-first testing.** Every kernel and primitive is unit-tested on the native GPU against a
  CPU reference; the model is validated token-by-token against HF `transformers`.

## Build & test

Requirements: Rust (see `rust-toolchain.toml`) and a WebGPU-capable GPU.

```bash
# Optional: fetch the large tokenizer fixture (the loader's tokenizer test skips without it)
bash scripts/fetch-fixtures.sh

# Unit tests (native GPU): kernels, primitives, loaders
cargo test -p eyebrowse-core -p eyebrowse-gpu -p eyebrowse-kernels -p eyebrowse-nn

# Download Qwen3-0.6B weights + regenerate the HF golden (uses uv + transformers)
hf download Qwen/Qwen3-0.6B --local-dir models/qwen3-0.6b
python golden/gen_golden.py     # writes golden/qwen3-golden.json

# Qwen3 end-to-end vs HF golden
cargo test -p eyebrowse --release --test generate -- --nocapture
cargo run  -p eyebrowse --release --example generate

# Mistral path: synthetic tiny model + HF logits golden
uv run --with torch --with transformers --with safetensors golden/gen_mistral_golden.py
cargo test -p eyebrowse --release --test mistral -- --nocapture

# GGUF: build Q8_0 + Q4_K_M fixtures of Qwen3-0.6B, then test end-to-end vs the golden
bash scripts/make-gguf-fixtures.sh
cargo test -p eyebrowse --release --test gguf -- --nocapture
```

## Roadmap

- **Gemma 4** (its own effort: sandwich norms, per-layer head_dim, partial-rotary RoPE, GeGLU, V-norm, logit softcap).
- Native quantized matmul kernels (GGUF weights currently dequantize to f16 at load; computing on packed quants would cut memory + bandwidth).
- More GGUF quant types (Q2_K/Q3_K/Q5_K, legacy Q4_0/Q5_0) and GGUF tokenizer extraction.
- A second modality: an image-generation pipeline on the same runtime.
- Performance: kernel fusion, subgroups, fewer transient allocations.

## License

Apache-2.0.
