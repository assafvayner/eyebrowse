#!/usr/bin/env bash
# Fetch test fixtures that are too large to commit (the Qwen3-0.6B tokenizer, ~11 MB).
# The tiny config.json fixture IS committed; only the tokenizer is fetched here.
set -euo pipefail

dir="$(cd "$(dirname "$0")/.." && pwd)/crates/eyebrowse-load/tests/fixtures"
mkdir -p "$dir"

url="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
out="$dir/qwen3-0.6b-tokenizer.json"

if [ -f "$out" ]; then
  echo "already present: $out"
else
  echo "downloading tokenizer.json -> $out"
  curl -sL "$url" -o "$out"
  echo "done"
fi
