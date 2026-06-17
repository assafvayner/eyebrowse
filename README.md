# eyebrowse

A lean, **extensible** model runtime written in Rust that runs neural networks on **WebGPU** via
[`wgpu`](https://github.com/gfx-rs/wgpu) (Metal/Vulkan/DX12). Kernels are hand-written WGSL; the
design favors a small set of composable primitives so adding a new model architecture is a
self-contained module, not a rewrite.

The first proof model is **Qwen3-0.6B** text generation. It generates correct text matching
HuggingFace `transformers` token-for-token.

```
$ cargo run -p eyebrowse --release --example generate
PROMPT:  The capital of France is
OUTPUT:  Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China
```

## Status

v1 (text generation) is implemented and validated:

- ✅ Native generation on Apple Silicon (Metal) — matches the HF greedy golden 20/20.

## Architecture

A Cargo workspace, layered bottom-up. Each crate has one responsibility:

| Crate | Responsibility |
|---|---|
| `eyebrowse-core` | Shared `DType` + the crate-wide error type. |
| `eyebrowse-gpu` | `wgpu` device, `Tensor` (a handle over a GPU buffer), a command `Recorder`, and the kernel-dispatch helper. |
| `eyebrowse-kernels` | Hand-written WGSL compute kernels (GEMM, RMSNorm, RoPE, flash-attention, SwiGLU, embedding, KV-cache write, argmax) + native correctness tests. |
| `eyebrowse-nn` | Composable primitives: `Linear`, `RmsNorm`, `Rope`, `Attention` (GQA + QK-RMSNorm + KV cache), `Mlp` (SwiGLU), `Embedding`. |
| `eyebrowse-load` | `WeightSource` trait + safetensors loader, normalized HF `config.json`, and tokenizer. |
| `eyebrowse-models` | Per-architecture modules. `qwen3` assembles `eyebrowse-nn` primitives from a config. |
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

# End-to-end vs HF golden
cargo test -p eyebrowse --release --test generate -- --nocapture
cargo run  -p eyebrowse --release --example generate
```

## Roadmap

- A second weight format (GGUF) to exercise the loader trait against a second source.
- Quantization (q8 → q4), reusing the packed-weight kernel pattern.
- A second modality: an image-generation pipeline on the same runtime.
- Performance: kernel fusion, subgroups, fewer transient allocations.

## License

Apache-2.0.
