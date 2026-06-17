# Build a tiny random Mistral, save weights, and record HF last-position logits as a golden.
# Run: uv run --with torch --with transformers --with safetensors golden/gen_mistral_golden.py
import torch, json, os
from transformers import MistralConfig, MistralForCausalLM

torch.manual_seed(0)
cfg = MistralConfig(
    vocab_size=320, hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, max_position_embeddings=128, rms_norm_eps=1e-6,
    rope_theta=1000000.0, tie_word_embeddings=False,
)
m = MistralForCausalLM(cfg).eval()

here = os.path.dirname(__file__)
out_dir = os.path.join(here, "..", "models", "mistral-tiny")
os.makedirs(out_dir, exist_ok=True)
m.save_pretrained(out_dir, safe_serialization=True)  # config.json + model.safetensors

ids = [1, 5, 9, 13, 17]
with torch.no_grad():
    logits = m(torch.tensor([ids])).logits[0, -1].float().tolist()

json.dump(
    {"input_ids": ids, "last_logits": logits},
    open(os.path.join(here, "mistral-tiny-golden.json"), "w"),
)
print(f"wrote mistral-tiny ({cfg.num_hidden_layers}L hidden={cfg.hidden_size} vocab={cfg.vocab_size})")
