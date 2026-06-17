#!/usr/bin/env bash
# Fetch test fixtures that are not committed (the Qwen3-0.6B config + tokenizer).
# Run before `cargo test -p eyebrowse-load` if you want the loader tests to exercise the
# real fixtures (they skip gracefully when absent).
set -euo pipefail

dir="$(cd "$(dirname "$0")/.." && pwd)/crates/eyebrowse-load/tests/fixtures"
mkdir -p "$dir"

base="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"

fetch() {
  local remote="$1" out="$dir/$2"
  if [ -f "$out" ]; then
    echo "already present: $out"
  else
    echo "downloading $remote -> $out"
    curl -sL "$base/$remote" -o "$out"
  fi
}

fetch config.json qwen3-0.6b-config.json
fetch tokenizer.json qwen3-0.6b-tokenizer.json
echo "done"
