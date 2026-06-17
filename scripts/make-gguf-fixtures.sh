#!/usr/bin/env bash
#
# make-gguf-fixtures.sh
#
# Regenerate the GGUF fixtures used to validate the Rust GGUF loader.
#
# Produces, from the HF safetensors model in models/qwen3-0.6b/:
#   models/gguf/qwen3-0.6b-q8_0.gguf    uniform Q8_0 quantization
#   models/gguf/qwen3-0.6b-q4_k_m.gguf  Q4_K_M (mix of Q4_K + Q6_K, F32 norms)
#
# It clones + builds llama.cpp in a throwaway, gitignored location, uses the
# HF->GGUF converter for Q8_0 (direct) and for an F16 intermediate, then runs
# llama-quantize to produce Q4_K_M. The F16 intermediate is deleted at the end.
#
# Requirements (already present in this environment):
#   uv, git, cmake, a C/C++ toolchain, Python 3.12
#
# Usage:
#   ./scripts/make-gguf-fixtures.sh
#
set -euo pipefail

# --- paths -------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_MODEL="${REPO_ROOT}/models/qwen3-0.6b"
OUT_DIR="${REPO_ROOT}/models/gguf"

# Throwaway llama.cpp checkout/build. Lives under models/ (gitignored).
# Override with LLAMA_CPP_DIR=/tmp/llama.cpp if you prefer.
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/models/.llama.cpp-build}"
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp"

Q8_0_OUT="${OUT_DIR}/qwen3-0.6b-q8_0.gguf"
F16_OUT="${OUT_DIR}/qwen3-0.6b-f16.gguf"     # intermediate, removed at the end
Q4KM_OUT="${OUT_DIR}/qwen3-0.6b-q4_k_m.gguf"

# Python deps for the converter, pulled ephemerally by `uv run --with ...`.
UV_DEPS=(--with numpy --with torch --with transformers --with sentencepiece --with safetensors --with gguf)

# --- sanity checks -----------------------------------------------------------
if [[ ! -f "${SRC_MODEL}/config.json" || ! -f "${SRC_MODEL}/model.safetensors" ]]; then
  echo "error: expected HF model at ${SRC_MODEL} (config.json + model.safetensors)" >&2
  exit 1
fi

for tool in uv git cmake; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "error: '${tool}' not found in PATH" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"

# --- 1. clone llama.cpp (shallow) -------------------------------------------
if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
  echo ">> cloning llama.cpp (shallow) into ${LLAMA_CPP_DIR}"
  rm -rf "${LLAMA_CPP_DIR}"
  git clone --depth 1 "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
else
  echo ">> reusing existing llama.cpp checkout at ${LLAMA_CPP_DIR}"
fi

cd "${LLAMA_CPP_DIR}"

# --- 2. Q8_0: converter emits it directly -----------------------------------
echo ">> converting ${SRC_MODEL} -> Q8_0"
uv run "${UV_DEPS[@]}" python convert_hf_to_gguf.py \
  "${SRC_MODEL}" \
  --outfile "${Q8_0_OUT}" \
  --outtype q8_0

# --- 3. F16 intermediate for the K-quant path -------------------------------
echo ">> converting ${SRC_MODEL} -> F16 (intermediate for Q4_K_M)"
uv run "${UV_DEPS[@]}" python convert_hf_to_gguf.py \
  "${SRC_MODEL}" \
  --outfile "${F16_OUT}" \
  --outtype f16

# --- 4. build llama-quantize ------------------------------------------------
echo ">> building llama-quantize"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=OFF
cmake --build build --target llama-quantize -j

# llama.cpp historically placed the binary at ./build/bin/llama-quantize;
# fall back to ./build/llama-quantize for older layouts.
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
[[ -x "${QUANTIZE_BIN}" ]] || QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/llama-quantize"

# --- 5. Q4_K_M from the F16 intermediate ------------------------------------
echo ">> quantizing F16 -> Q4_K_M"
"${QUANTIZE_BIN}" "${F16_OUT}" "${Q4KM_OUT}" Q4_K_M

# --- 6. drop the F16 intermediate -------------------------------------------
rm -f "${F16_OUT}"

echo
echo ">> done. fixtures:"
ls -lh "${Q8_0_OUT}" "${Q4KM_OUT}"
