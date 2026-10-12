#!/usr/bin/env bash
# ============================================================
# 步驟 7：建立量化專用的獨立 venv（只有要跑 05_quantize.py 時才需要）
# 用法：bash 07_setup_quant_env.sh
#
# 為什麼要獨立一個 venv，而且不併進 00_setup.sh：
#   1. 主環境的 numpy 被 bfcl-eval 釘死在 1.26.4（見 00_setup.sh 的註解），
#      但 llmcompressor 透過 datasets → pyarrow 26 需要 numpy 2。pyarrow 沒有在
#      metadata 宣告這個需求，pip check 抓不到，要到 import 才噴：
#        ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
#      兩邊不可能在同一個環境裡同時滿足。
#   2. venv 要自己裝一份 torch（約數 GB），大部分上機不做量化，
#      併進 00_setup.sh 會讓每次重建都白裝一次。
#   3. 完全隔離（不用 --system-site-packages）：venv 的 numpy 2 不會去
#      蓋主環境那些對 numpy 1.x 建置的套件，主環境的 BFCL 基準不受影響。
#
# 量化一律用 venv 的 python 執行：
#   /workspace/venv-quant/bin/python 05_quantize.py ...
# vLLM 與 BFCL 照舊用主環境，不要在 activate 過 venv 的 shell 裡跑。
# ============================================================
set -e

WORKDIR=${WORKDIR:-/workspace}
VENV=${VENV:-$WORKDIR/venv-quant}

# 版本跟主環境對齊：
#   torch 2.13.0（cu130）= vLLM 0.28.0 釘的版本，驅動需求相同
#   llmcompressor 0.14.0 會帶進 compressed-tensors 0.19.0，跟主環境 vLLM 端
#   實際在用的 ct 版本一致，量化出來的 checkpoint 格式兩邊對得上
TORCH_VER=2.13.0
LLMC_VER=0.14.0

if [ -e "$VENV" ]; then
  echo "  $VENV 已存在，不重建。要重建請先手動處理舊的那份。"
else
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install "torch==$TORCH_VER" "llmcompressor==$LLMC_VER"
fi

echo ""
echo "=============================================="
echo "【檢查】venv 內的版本與相依"
echo "=============================================="
"$VENV/bin/pip" check
# 在 /tmp 執行並加 -I，避免目前目錄裡的同名檔案被當成模組匯入
(cd /tmp && "$VENV/bin/python" -I - <<'PY'
import numpy, torch, compressed_tensors, transformers, llmcompressor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
import datasets
print(f"  numpy          = {numpy.__version__}")
print(f"  torch          = {torch.__version__}")
print(f"  compressed-tensors = {compressed_tensors.__version__}")
print(f"  transformers   = {transformers.__version__}")
print(f"  llmcompressor  = {llmcompressor.__version__}")
print(f"  CUDA 可用      = {torch.cuda.is_available()}")
assert numpy.__version__.startswith("2."), "venv 的 numpy 應該是 2.x"
print("  OK llmcompressor / GPTQModifier / oneshot / datasets 都能匯入")
PY
)

echo ""
echo "=============================================="
echo "【檢查】主環境沒有被動到"
echo "=============================================="
(cd /tmp && python3 -I -c "import numpy; v=numpy.__version__; print(f'  主環境 numpy = {v}'); assert v == '1.26.4', '主環境 numpy 應維持 1.26.4（bfcl-eval 釘的）'")
echo "  OK"
