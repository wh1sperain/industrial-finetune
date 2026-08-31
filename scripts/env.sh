#!/usr/bin/env bash
# 环境初始化：设置数据盘缓存/网络/密钥环境变量 + 环境自检
# 用法：
#   source scripts/env.sh        # 当前会话生效
#   bash scripts/env.sh          # 只打印检查报告
# 处理项：HF Xet 401（HF_HUB_DISABLE_XET=1）、缓存指向数据盘、
#         .env 密钥加载、torch/trl 版本与模型路径自检
# ============================================================================
# 是否被 source（source 时不设置 set -euo pipefail——
# 否则 set -e 会污染调用者 shell，导致任何失败命令/Ctrl+C 直接退出终端）
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  SOURCED=0
  set -euo pipefail
else
  SOURCED=1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================================
# 1. 数据盘检测与缓存路径（AutoDL: /root/autodl-tmp；其他机器 fallback 到 $HOME）
# ============================================================================
if [[ -d /root/autodl-tmp ]]; then
  DATA_DISK=/root/autodl-tmp
else
  DATA_DISK="$HOME"
fi
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_DISK/modelscope}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
echo "[env] 数据盘: $DATA_DISK"
echo "[env] 模型缓存: $MODELSCOPE_CACHE"
echo "[env] HF 缓存: $HF_HOME"

# ============================================================================
# 2. HF 网络（Xet 401 bug 强制禁用；国内镜像按需开关）
# ============================================================================
export HF_HUB_DISABLE_XET=1
# 国内服务器直连 HF 仍慢时，取消下一行注释启用镜像（与 HF_ENDPOINT 默认值互斥）：
# export HF_ENDPOINT=https://hf-mirror.com
echo "[env] HF_HUB_DISABLE_XET=1（Xet 401 bug 已禁用）"

# ============================================================================
# 3. 密钥（.env 存在则加载；HF_TOKEN 走 ModelScope 后非必需，wandb 需要）
# ============================================================================
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
  echo "[env] 已加载 $REPO_DIR/.env"
else
  echo "[env] 警告: 未找到 .env（训练用 wandb 前请 cp .env.example .env 并填 WANDB_API_KEY）"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ============================================================================
# 4. 环境自检
# ============================================================================
echo ""
echo "===== 环境自检 ====="

PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[✗] python 不在 PATH（先 source .venv/bin/activate）"
else
  echo "[✓] python: $PYTHON_BIN"
fi

# 4.1 Python / torch / CUDA / trl 版本
python - <<'EOF' 2>&1 || true
import sys
ok = True
print(f"[✓] Python: {sys.version.split()[0]}")
try:
    import torch
    print(f"[✓] torch: {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available() and torch.cuda.get_device_name(0):
        ok = ok and True
    major, minor = int(torch.__version__.split(".")[0]), int(torch.__version__.split(".")[1])
    if (major, minor) < (2, 7):
        print("[✗] torch < 2.7：Blackwell (sm_120) 不支持，需装 cu128/cu130 版")
        ok = False
except ImportError:
    print("[✗] torch 未安装")
    ok = False
try:
    import trl
    v = tuple(int(x) for x in trl.__version__.split(".")[:2])
    print(f"[✓] trl: {trl.__version__}")
    if not (1, 12) <= v < (2, 0):
        print("[✗] trl 版本不在 [1.12, 2.0)：SFTConfig API 可能不兼容（pip install 'trl>=1.12,<2.0'）")
        ok = False
except ImportError:
    print("[✗] trl 未安装")
    ok = False
sys.exit(0 if ok else 1)
EOF

# 4.2 模型路径（configs/models.yaml 的 local_paths 映射，用户自填，检查本地存在性）
LOCAL_PATHS=$(grep -A20 '^local_paths:' "$REPO_DIR/configs/models.yaml" 2>/dev/null | grep -E '^\s+\S+:\s*\S+' | sed 's/^\s*//' || true)
if [[ -n "$LOCAL_PATHS" ]]; then
  echo "[✓] 模型本地映射:"
  echo "$LOCAL_PATHS" | sed 's/^/    /'
  MISSING=$(echo "$LOCAL_PATHS" | awk -F': ' '{if ($2 != "" && system("test -d \"" $2 "\"") != 0) print "     [✗] 路径不存在: " $2}')
  if [[ -n "$MISSING" ]]; then
    echo "$MISSING"
    echo "     → 检查 configs/models.yaml 中该路径是否正确"
  fi
else
  echo "[!] configs/models.yaml 无 local_paths 映射（模型将在线加载，或自行填写本地路径）"
fi

# 4.3 wandb 密钥
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "[!] WANDB_API_KEY 为空：wandb 上报会失败；可填 .env 或训练时 report_to 改 none"
else
  echo "[✓] WANDB_API_KEY 已设置"
fi

echo ""
echo "===== 自检完成 ====="
if [[ "$SOURCED" == "0" ]]; then
  echo "（直接运行模式：以上为检查报告，未修改当前会话；"
  echo "  请用 source scripts/env.sh 使环境变量生效）"
fi
