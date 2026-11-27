#!/usr/bin/env bash
# Build the pinned llama-frankenstein + Inkling 2 Mini / DSpark llama-server binary for RDNA4 (gfx1201).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${VENDOR_DIR:-$REPO_ROOT/vendor}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1201}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="${JOBS:-$(nproc)}"

LLAMA_COMMIT="919794500"
RDNA_BOOSTS_TAG="v18-919794500-r359"
RDNA_BOOSTS_COMMIT="a0e7e62b9dc748ecfe1902101084c46c244748fe"
OVERDOING_COMMIT="1ec4b8c4858052afc4803aaabaf2fde69c35688c"

mkdir -p "$VENDOR_DIR"

echo "==> [1/5] Cloning pinned upstream repositories into $VENDOR_DIR"
if [[ ! -d "$VENDOR_DIR/llama-cpp-rdna-boosts/.git" ]]; then
  git clone https://github.com/stew675/llama-cpp-rdna-boosts.git "$VENDOR_DIR/llama-cpp-rdna-boosts"
fi
git -C "$VENDOR_DIR/llama-cpp-rdna-boosts" fetch --tags --quiet
git -C "$VENDOR_DIR/llama-cpp-rdna-boosts" checkout "$RDNA_BOOSTS_COMMIT"

if [[ ! -d "$VENDOR_DIR/llama-cpp-overdoing/.git" ]]; then
  git clone https://github.com/overdoingism/llama-cpp-overdoing.git "$VENDOR_DIR/llama-cpp-overdoing"
fi
git -C "$VENDOR_DIR/llama-cpp-overdoing" fetch --quiet
git -C "$VENDOR_DIR/llama-cpp-overdoing" checkout "$OVERDOING_COMMIT"

SRC_DIR="$VENDOR_DIR/llama-frankenstein"
if [[ -d "$SRC_DIR" ]]; then
  echo "Existing $SRC_DIR found. Remove it first (rm -rf '$SRC_DIR') for a clean rebuild." >&2
  exit 1
fi

git clone https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
git -C "$SRC_DIR" checkout "$LLAMA_COMMIT"

echo "==> [2/5] Applying stew675/llama-cpp-rdna-boosts ($RDNA_BOOSTS_TAG)"
(cd "$SRC_DIR" && bash "$VENDOR_DIR/llama-cpp-rdna-boosts/apply-all.sh")

echo "==> [3/5] Applying overdoingism/llama-cpp-overdoing (J1-J7 + J2 part b)"
PDIR="$VENDOR_DIR/llama-cpp-overdoing/patches"
for p in \
  J1-919-attention-v5.patch \
  J2-919-prefill-a-v15.patch \
  J3-919-decode-v5.3.patch \
  J4-919-sampling-v1.patch \
  J5-919-speculative-v3.1.patch \
  J6-919-step-v10.1.patch \
  J7-919-state-v3.patch \
  J2-919-prefill-b-v15.patch; do
  git -C "$SRC_DIR" apply "$PDIR/$p"
done

echo "==> [4/5] Applying Inkling 2 Mini + DSpark + F16 in-place kq_b patches"
git -C "$SRC_DIR" apply "$REPO_ROOT/patches/0001-inkling2-and-dspark.patch"
git -C "$SRC_DIR" apply "$REPO_ROOT/patches/0002-inkling2-f16-kqb-inplace.patch"

echo "==> [5/5] Configuring & compiling llama-server for $AMDGPU_TARGETS (JOBS=$JOBS)"
HIP_CXX="$ROCM_PATH/lib/llvm/bin/clang++"
if [[ ! -x "$HIP_CXX" ]]; then
  HIP_CXX="$(command -v hipcc || true)"
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$HIP_CXX" \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DGPU_TARGETS="$AMDGPU_TARGETS" \
  -DGGML_HIP_NO_VMM=ON \
  -DFRANK_HIPBLASLT=ON \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF

cmake --build "$BUILD_DIR" -j"$JOBS" --target llama-server

echo ""
echo "Build complete: $BUILD_DIR/bin/llama-server"
