#!/usr/bin/env bash
# ============================================================
# 步驟 0：環境檢查與安裝
# 用法：bash 00_setup.sh
# ============================================================
set -e   # 任何一行失敗就停下來，不要繼續跑

echo "=============================================="
echo "【檢查 1】顯卡是什麼、CUDA 版本多少"
echo "=============================================="
nvidia-smi
echo ""
echo ">>> 要確認兩件事："
echo "    1. GPU 名稱有沒有寫 RTX 5090"
echo "    2. 右上角 CUDA Version 是不是 12.8 以上"
echo "    如果 CUDA 低於 12.8，5090 可能跑不起來，要換 RunPod 樣板"
echo ""
# 加 || true：腳本開頭有 set -e，若用 `bash 00_setup.sh < /dev/null` 餵空輸入，
# read 讀到 EOF 會回傳非零值，整支腳本會在這裡直接中斷、一個套件都沒裝。
read -p "確認無誤按 Enter 繼續，有問題按 Ctrl+C 中斷..." _ || true

echo ""
echo "=============================================="
echo "【安裝】主要套件"
echo "=============================================="

# 工作目錄。如果你有掛 Network Volume，改成 /workspace
WORKDIR=${WORKDIR:-/workspace}
mkdir -p "$WORKDIR"/{models,logs,results}
cd "$WORKDIR"

pip install --upgrade pip

# vLLM：釘 0.28.0（相依 torch 2.13.0+cu130），需要驅動 580 以上。
#
# 為什麼要釘版本、又為什麼是這一版：
#   1. 不釘版本則不可重現，pip 每次可能裝到不同版，記憶體與效能數字無從對照。
#   2. vllm <= 0.19.1 相依 torch <= 2.10.0 (cu128)，能配舊驅動，但它打包的
#      FlashAttention 只含 sm_80 / sm_90a 的 cubin，沒有 sm_120。RTX 5090
#      上只能靠驅動 JIT 編譯 sm_80 的 PTX，而那份 PTX 是 ISA 8.8 (CUDA 12.9)，
#      驅動 570 最高只吃 ISA 8.7，會噴 cudaErrorUnsupportedPtxVersion。
#   3. 所以 5090 要用 FlashAttention，就必須配新驅動 + 新 vLLM。
pip install "vllm==0.28.0"

# 壓縮工具。一定要排在 bfcl-eval 前面：llmcompressor 會把 numpy 拉到 2.x，
# 而 bfcl-eval 2025.12.17 釘死 numpy==1.26.4。後裝的才是最終生效的版本，
# 所以 bfcl-eval 放最後，確保基準分數跑在它預期的 numpy 上。
pip install llmcompressor

# BFCL：版本必須釘死，這是可重現性的關鍵
pip install "bfcl-eval==2025.12.17"

# 下載模型用
pip install "huggingface_hub[cli]"

# qwen-agent（bfcl-eval 的相依）程式碼裡直接 import soundfile，但它的套件
# metadata 沒有宣告這個相依，不補裝的話 bfcl CLI 一啟動就 ModuleNotFoundError，
# 整個評測跑不了。這是上游打包的漏洞。
pip install soundfile

echo ""
echo "=============================================="
echo "【檢查 2】裝了什麼版本（這些數字論文要寫）"
echo "=============================================="
python3 - <<'PY'
import importlib.metadata as md
for pkg in ["vllm", "bfcl-eval", "llmcompressor", "torch", "transformers"]:
    try:
        print(f"  {pkg:<16} = {md.version(pkg)}")
    except Exception:
        print(f"  {pkg:<16} = (未安裝)")

import torch
print(f"\n  torch CUDA 版本    = {torch.version.cuda}")
print(f"  CUDA 可用          = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 名稱           = {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  Compute Capability = sm_{cap[0]}{cap[1]}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  總記憶體           = {total:.2f} GB")
PY

echo ""
echo "=============================================="
echo "【閘門】環境不對就停在這裡，不要往下做"
echo "=============================================="
# 硬性檢查。之前踩過的坑：套件裝完看起來一切正常，要到第 3 步啟動 vLLM
# 才發現驅動與 kernel 不相容，白花了一個多小時。寧可在這裡就擋下來。
python3 - <<'GATE'
import sys, torch
ok = True
if not torch.cuda.is_available():
    print(f"  X torch 看不到 GPU。torch 編譯的 CUDA = {torch.version.cuda}，"
          "通常是驅動比 torch 要求的舊。")
    ok = False
else:
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name(0)
    if sm != "sm_120":
        print(f"  X Compute Capability 是 {sm}，預期 sm_120（RTX 5090）。實際 GPU：{name}")
        ok = False
    else:
        print(f"  OK {name} / {sm} / torch CUDA {torch.version.cuda}")
if not ok:
    print("")
    print("  停止。請先處理驅動或樣板問題，不要繼續往下跑。")
    sys.exit(1)
print("  OK 環境檢查通過")
GATE

echo ""
echo ">>> 檢查重點："
echo "    Compute Capability 要是 sm_120（5090 的架構）"
echo "    總記憶體要接近 32 GB"
echo "    torch CUDA 版本要 12.8 以上"
echo ""
echo "把上面整段輸出複製起來存檔，論文的實驗環境章節要用。"
