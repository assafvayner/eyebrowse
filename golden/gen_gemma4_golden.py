#!/usr/bin/env python3
"""Generate a tiny *dense* Gemma 4 text model + a HuggingFace logits golden.

Run with:
    uv run --with torch --with transformers --with safetensors \
        python golden/gen_gemma4_golden.py

Builds a small dense (no PLE, no MoE) Gemma 4 text model that exercises both a
local (sliding) and a global (full) attention layer, saves it under
models/gemma4-tiny/ (gitignored), and writes the last-token logits (already
including the final softcap) to golden/gemma4-tiny-golden.json.
"""

import json
import os

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "models", "gemma4-tiny")
GOLDEN_PATH = os.path.join(REPO_ROOT, "golden", "gemma4-tiny-golden.json")


def main() -> None:
    torch.manual_seed(0)

    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        # Dense only: disable Per-Layer Embeddings and KV sharing, no MoE.
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        attention_k_eq_v=False,
        num_global_key_value_heads=None,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        # One local (sliding) + global (full) layer; last layer must be full.
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=512,
        # Per-layer-type RoPE: local default theta=1e4, global proportional theta=1e6.
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        torch_dtype="float32",
    )

    model = Gemma4ForCausalLM(config)
    model = model.to(torch.float32)
    model.eval()

    model.save_pretrained(MODEL_DIR, safe_serialization=True)

    input_ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float().tolist()

    golden = {"input_ids": input_ids[0].tolist(), "last_logits": logits}
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)

    # Report the saved tensor names/shapes for verification.
    from safetensors import safe_open

    st_path = os.path.join(MODEL_DIR, "model.safetensors")
    print("=== model.safetensors tensors ===")
    with safe_open(st_path, framework="pt") as st:
        for k in sorted(st.keys()):
            print(f"{k}\t{list(st.get_slice(k).get_shape())}")

    print("\n=== saved config.json key values ===")
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    for k in sorted(cfg.keys()):
        print(f"{k}: {cfg[k]}")

    print("\n=== golden ===")
    print(json.dumps(golden, indent=2))
    print(f"\nWrote {GOLDEN_PATH}")
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
