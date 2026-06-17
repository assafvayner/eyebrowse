"""Generate a HuggingFace transformers golden reference for the Qwen3-0.6B Rust runtime.

Run with: uv run --with torch --with transformers --with safetensors python golden/gen_golden.py
CPU, float32, deterministic.
"""

import json
import os

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = "/Users/assafvayner/eyebrowse/models/qwen3-0.6b"
OUT_PATH = "/Users/assafvayner/eyebrowse/golden/qwen3-golden.json"
PROMPT = "The capital of France is"

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


def embed_dtype_from_safetensors() -> str:
    st_path = os.path.join(LOCAL_DIR, "model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        keys = list(f.keys())
        key = "model.embed_tokens.weight"
        if key not in keys:
            # fall back to first matching embed key
            cand = [k for k in keys if "embed_tokens.weight" in k]
            key = cand[0] if cand else keys[0]
        slc = f.get_slice(key)
        dtype = slc.get_dtype()  # e.g. "BF16", "F16", "F32"
    mapping = {"BF16": "bf16", "F16": "f16", "F32": "f32"}
    return mapping.get(dtype, dtype.lower())


def main() -> None:
    embed_dtype = embed_dtype_from_safetensors()

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, torch_dtype=torch.float32)
    model.eval()

    input_ids = tokenizer(
        PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    input_id_list = input_ids[0].tolist()

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
        topk = torch.topk(logits, 10)
        top10_ids = topk.indices.tolist()
        top10_values = topk.values.tolist()

        gen = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            min_new_tokens=20,
            max_new_tokens=20,
        )
    continuation_ids = gen[0, input_ids.shape[1]:].tolist()
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=False)

    eos_token_id = model.config.eos_token_id

    golden = {
        "prompt": PROMPT,
        "input_ids": input_id_list,
        "first_logits_top10_ids": top10_ids,
        "first_logits_top10_values": top10_values,
        "greedy_continuation_ids": continuation_ids,
        "greedy_continuation_text": continuation_text,
        "embed_dtype": embed_dtype,
        "eos_token_id": eos_token_id,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print("=== GOLDEN JSON ===")
    print(json.dumps(golden, indent=2, ensure_ascii=False))
    print("=== END GOLDEN JSON ===")


if __name__ == "__main__":
    main()
