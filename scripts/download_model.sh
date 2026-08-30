#!/usr/bin/env bash
# 用 ModelScope 下载基座模型到本地（国内服务器推荐：免 HF 许可审核、直连快）
# 用法：bash scripts/download_model.sh [model_id]
# 默认模型：Qwen/Qwen2.5-7B-Instruct
# 下载完成后自动把 configs/train/lora_qwen7b_industrial.yaml 的 model_name_or_path 指向本地路径
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="configs/train/lora_qwen7b_industrial.yaml"
CACHE_DIR="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope/hub}"

# 1. 检查 modelscope
if ! python -c "import modelscope" 2>/dev/null; then
  echo "[download_model] modelscope 未安装，请先执行: pip install modelscope" >&2
  exit 1
fi

# 2. 下载模型（国内速度分钟级，约 15GB）
echo "[download_model] 下载 $MODEL_ID -> $CACHE_DIR ..."
MODEL_PATH=$(python - "$MODEL_ID" "$CACHE_DIR" <<'EOF'
import sys
from modelscope import snapshot_download
print(snapshot_download(sys.argv[1], cache_dir=sys.argv[2]))
EOF
)
echo "[download_model] 完成: $MODEL_PATH"

# 3. 自动更新 config 指向本地路径（幂等，可重复执行）
python - "$CONFIG" "$MODEL_PATH" <<'EOF'
import re
import sys

cfg_path, model_path = sys.argv[1], sys.argv[2]
with open(cfg_path, encoding="utf-8") as f:
    content = f.read()
if model_path in content:
    print(f"[download_model] config 已是本地路径，无需更新: {cfg_path}")
else:
    new = re.sub(r"model_name_or_path:.*", f"model_name_or_path: {model_path}", content, count=1)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"[download_model] config 已更新: {cfg_path}")
EOF

echo "[download_model] 下一步: bash scripts/train.sh configs/train/lora_qwen7b_industrial.yaml"
