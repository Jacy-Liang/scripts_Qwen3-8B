#!/usr/bin/env bash
# ============================================================
# 步驟 1：下載模型
# 用法：bash 01_download.sh
#
# 只先下載第一個要測的（Qwen3-8B），其他等驗證過再下載。
# 理由：先確認流程對，再花時間下載其他的。
# ============================================================
set -e

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

# 下載會很久，開一個能斷線續傳的加速器
pip install -q hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 模型存放位置（如果有 Network Volume，關機重開不用重下載）
export HF_HOME="$WORKDIR/hf_cache"

echo "=============================================="
echo "下載 Qwen3-8B 原版（約 16 GB，會跑一陣子）"
echo "=============================================="
hf download Qwen/Qwen3-8B --local-dir "$WORKDIR/models/Qwen3-8B"

echo ""
echo "=============================================="
echo "檢查：看實際大小跟預期對不對"
echo "=============================================="
du -sh "$WORKDIR/models/Qwen3-8B"
echo ""
echo ">>> 應該接近 16 GB（8B 參數 × 2 bytes）"
echo "    差太多代表沒下載完，重跑一次（會續傳）"

echo ""
echo "=============================================="
echo "把架構參數印出來（算 KV cache 要用）"
echo "=============================================="
python3 - <<PY
import json, os
p = os.path.join("$WORKDIR", "models", "Qwen3-8B", "config.json")
c = json.load(open(p))
L  = c["num_hidden_layers"]
H  = c["hidden_size"]
QH = c["num_attention_heads"]
KV = c["num_key_value_heads"]
hd = c.get("head_dim", H // QH)

print(f"  層數 (num_hidden_layers)      = {L}")
print(f"  hidden_size                   = {H}")
print(f"  Q head 數                     = {QH}")
print(f"  KV head 數                    = {KV}   ← GQA，這個決定 KV cache 大小")
print(f"  head_dim                      = {hd}")
print(f"  原生最長 context              = {c['max_position_embeddings']:,}")

# KV cache 每個 token 佔多少（FP16 = 2 bytes）
per_tok = 2 * L * KV * hd * 2
print(f"\n  每個 token 的 KV cache        = {per_tok/1024:.1f} KB")
print(f"  16K context 單一 agent        = {per_tok*16384/1024**3:.2f} GB")
print(f"  16K context × 8 個 agent      = {per_tok*16384*8/1024**3:.2f} GB")
PY

echo ""
echo ">>> 把這些數字抄進你的可用性表"
