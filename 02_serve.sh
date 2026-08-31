#!/usr/bin/env bash
# ============================================================
# 步驟 2：啟動 vLLM
# 用法：bash 02_serve.sh <模型路徑> <標籤>
# 範例：bash 02_serve.sh /workspace/models/Qwen3-8B qwen3-8b-fp16
#
# 這支會把啟動訊息存檔，因為記憶體數字就藏在裡面。
# ============================================================
set -e

WORKDIR=${WORKDIR:-/workspace}
MODEL_PATH=$1
TAG=$2

if [ -z "$MODEL_PATH" ] || [ -z "$TAG" ]; then
  echo "用法：bash 02_serve.sh <模型路徑> <標籤>"
  exit 1
fi

LOG="$WORKDIR/logs/serve_${TAG}.log"
mkdir -p "$WORKDIR/logs"

# ---------- 實驗參數：這些就是你的研究變因 ----------
GPU_UTIL=${GPU_UTIL:-0.85}        # 最多用幾成顯卡記憶體。先保守，成功再往上調
MAX_LEN=${MAX_LEN:-16384}         # 單次最長 context。一定要設！見下方說明
MAX_SEQS=${MAX_SEQS:-8}           # 同時最多處理幾個請求
PORT=${PORT:-8000}
# ---------------------------------------------------

# 為什麼一定要設 MAX_LEN：
#   Qwen3-4B 原生 context 是 262,144。如果不設，vLLM 會照那個長度
#   去配置 KV cache，需要約 37 GB，連 5090 都不夠，會直接啟動失敗。

echo "=============================================="
echo "啟動設定（這些論文要記錄）"
echo "=============================================="
echo "  模型      : $MODEL_PATH"
echo "  gpu_util  : $GPU_UTIL"
echo "  max_len   : $MAX_LEN"
echo "  max_seqs  : $MAX_SEQS"
echo "  log 存到  : $LOG"
echo ""

# 把設定也寫進 log 開頭，之後回頭看才知道當時用什麼跑的
{
  echo "===== 啟動設定 ====="
  echo "時間     : $(date -Iseconds)"
  echo "模型     : $MODEL_PATH"
  echo "gpu_util : $GPU_UTIL"
  echo "max_len  : $MAX_LEN"
  echo "max_seqs : $MAX_SEQS"
  echo "===================="
} > "$LOG"

# 啟動。tee -a 是「畫面上顯示，同時附加到檔案」
vllm serve "$MODEL_PATH" \
  --served-model-name "$TAG" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee -a "$LOG"

# ---------- 給 xLAM 用的版本（要多兩個參數）----------
# 先下載 parser（整個 xLAM 系列共用一個，只要下載一次）：
#   wget https://huggingface.co/Salesforce/xLAM-2-1b-fc-r/raw/main/xlam_tool_call_parser.py
#
# 然後把上面的 --tool-call-parser hermes 換成：
#   --tool-parser-plugin ./xlam_tool_call_parser.py \
#   --tool-call-parser xlam
#
# ---------- 給 Nanbeige 用的版本 ----------
# 要多加：--trust-remote-code
# 這個模型用了自訂程式碼，vLLM 不一定支援，第一次跑要特別注意有沒有報錯
