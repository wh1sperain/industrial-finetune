#!/usr/bin/env bash
# ModelScope 下载基座模型，并写入 configs/models.yaml 的 model_path（模型路径唯一来源）
# 用法：bash scripts/download_model.sh [model_id]
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-7B-Instruct}"
MODELS_CONFIG="configs/models.yaml"
CACHE_DIR="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope/hub}"

if ! python -c "import modelscope" 2>/dev/null; then
  echo "[download_model] modelscope 未安装，请先执行: pip install modelscope" >&2
  exit 1
fi

echo "[download_model] 下载 $MODEL_ID -> $CACHE_DIR ..."
MODEL_PATH=$(python - "$MODEL_ID" "$CACHE_DIR" <<'EOF'
import sys
from modelscope import snapshot_download
print(snapshot_download(sys.argv[1], cache_dir=sys.argv[2]))
EOF
)
echo "[download_model] 完成: $MODEL_PATH"

python - "$MODELS_CONFIG" "$MODEL_PATH" <<'EOF'
import re
import sys

cfg_path, model_path = sys.argv[1], sys.argv[2]
with open(cfg_path, encoding="utf-8") as f:
    content = f.read()
if f"model_path: {model_path}" in content:
    print(f"[download_model] model_path 已是最新: {cfg_path}")
else:
    new = re.sub(r"^model_path:.*$", f"model_path: {model_path}", content, count=1, flags=re.MULTILINE)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"[download_model] model_path 已更新: {cfg_path}")
EOF

echo "[download_model] 下一步: bash scripts/train.sh configs/train/lora_qwen7b_industrial.yaml"
