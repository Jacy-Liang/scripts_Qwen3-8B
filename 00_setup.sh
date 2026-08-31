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
read -p "確認無誤按 Enter 繼續，有問題按 Ctrl+C 中斷..." _

echo ""
echo "=============================================="
echo "【安裝】主要套件"
echo "=============================================="

# 工作目錄。如果你有掛 Network Volume，改成 /workspace
WORKDIR=${WORKDIR:-/workspace}
mkdir -p "$WORKDIR"/{models,logs,results}
cd "$WORKDIR"

pip install --upgrade pip

# vLLM：不指定版本，讓它裝最新的（Blackwell 支援還在演進中）
pip install vllm

# BFCL：版本必須釘死，這是可重現性的關鍵
pip install "bfcl-eval==2025.12.17"

# 壓縮工具
pip install llmcompressor

# 下載模型用
pip install "huggingface_hub[cli]"

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
echo ">>> 檢查重點："
echo "    Compute Capability 要是 sm_120（5090 的架構）"
echo "    總記憶體要接近 32 GB"
echo "    torch CUDA 版本要 12.8 以上"
echo ""
echo "把上面整段輸出複製起來存檔，論文的實驗環境章節要用。"
