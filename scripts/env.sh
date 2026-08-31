#!/usr/bin/env bash
# ============================================================================
# 环境初始化脚本：集中设置所有环境变量 + 环境自检（历史踩坑一键处理）
#
# 用法：
#   source scripts/env.sh        # 当前会话生效（推荐：训练/评测前执行一次）
#   bash scripts/env.sh          # 只打印环境检查报告（不改变当前会话）
#   echo 'source ~/autodl-tmp/industrial-finetune/scripts/env.sh' >> ~/.bashrc
#                                # 登录自动生效（可选）
#
# 解决的历史问题：
#   1. HF Xet 下载协议 401 bug        → HF_HUB_DISABLE_XET=1
#   2. 国内服务器直连 HF 慢/卡死       → HF_ENDPOINT 镜像（可开关）
#   3. 模型/数据集占满系统盘           → MODELSCOPE_CACHE / HF_HOME 指向数据盘
#   4. gated 模型 401 Unauthorized    → HF_TOKEN（从 .env 加载；走 ModelScope 后可选）
#   5. wandb 密钥不生效（代码不读.env）→ WANDB_API_KEY（从 .env 加载并 export）
#   6. torch 被装成 CPU 版            → 自检 torch>=2.7 + cuda.is_available()
#   7. TRL 版本 API 不兼容            → 自检 1.12 <= trl < 2.0
#   8. 模型路径被 git pull 覆盖回 HF ID → 自检 config 中 model_name_or_path 存在
# ============================================================================
set -euo pipefail

# 是否被 source（source 时不允许 exit）
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  SOURCED=0
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

# 4.2 模型路径（从 configs/models.yaml 单一事实源读取，检查本地存在性）
MODEL_PATH="$(grep -E '^model_path:' "$REPO_DIR/configs/models.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' || true)"
BASE_MODEL="$(grep -E '^base_model:' "$REPO_DIR/configs/models.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' || true)"
if [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]]; then
  echo "[✓] 模型本地路径存在: $MODEL_PATH"
elif [[ -n "$MODEL_PATH" ]]; then
  echo "[✗] configs/models.yaml 的 model_path 不存在: $MODEL_PATH"
  echo "     → 执行 bash scripts/download_model.sh 重新下载/修正"
else
  echo "[!] model_path 为空，将使用在线模型 ID（$BASE_MODEL）"
  echo "     → 建议先执行 bash scripts/download_model.sh 下载到本地"
fi

# 4.3 评测数据集缓存
for ds in "acnul___mining-engineering-sft" "acnul___mining-engineering-eval" "acnul___mining-engineering-probe"; do
  if [[ -d "$HF_HOME/datasets/$ds" || -d "$HF_HOME/hub/datasets--$ds" ]]; then
    echo "[✓] 数据集缓存: $ds"
  else
    echo "[!] 数据集未缓存: $ds（可执行 bash scripts/download_data.sh / eval 前预下载）"
  fi
done

# 4.4 wandb 密钥
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
