#!/usr/bin/env bash
# set -euo pipefail

# OUT_DIR="${1:-./unstructed_navsim}"
# mkdir -p "$OUT_DIR"
# cd "$OUT_DIR"

# REPO="aaaaaap/unstructed"
# REV="main"
# BASE_URL="https://huggingface.co/datasets/${REPO}/resolve/${REV}/navsim"

# # 如果需要 token（私有/门控仓库），取消下一行注释并填入
# # HF_TOKEN="hf_xxx"
# # AUTH_HEADER=(--header="Authorization: Bearer ${HF_TOKEN}")

# AUTH_HEADER=()

# download_one () {
#   local url="$1"
#   local out="$2"
#   aria2c -c -x 4 -s 4 -k 1M \
#     "${AUTH_HEADER[@]}" \
#     -o "$out" \
#     "${url}?download=true"
# }

# train: 0000..0037
for i in $(seq -w 0000 0037); do
  # fname="navsim_train_shard_${i}.tar"
  # url="${BASE_URL}/${fname}"
  # echo "Downloading ${fname}"
  # download_one "$url" "$fname"
  echo "Extracting file navsim_train_shard_${i}.tar"
  tar -xf navsim_train_shard_${i}.tar
done

# val: 0000..0009
for i in $(seq -w 0000 0009); do
  # fname="navsim_val_shard_${i}.tar"
  # url="${BASE_URL}/${fname}"
  # echo "Downloading ${fname}"
  # download_one "$url" "$fname"
  echo "Extracting file navsim_val_shard_${i}.tar"
  tar -xf navsim_val_shard_${i}.tar
done

echo "All done. Files saved to: ${OUT_DIR}"

