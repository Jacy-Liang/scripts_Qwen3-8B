#!/usr/bin/env bash
# ============================================================
# 步驟 4：跑 BFCL，對照 leaderboard
# 用法：bash 04_bfcl.sh
#
# ⚠️ 這一步是閘門。對不上就停下來修，不要往下做。
# ============================================================
set -e

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

# BFCL 的模型 ID。這個字串要跟 BFCL 支援清單上寫的一模一樣
BFCL_MODEL=${BFCL_MODEL:-"Qwen/Qwen3-8B-FC"}

echo "=============================================="
echo "【檢查】確認 bfcl-eval 版本正確"
echo "=============================================="
python3 -c "
import importlib.metadata as md
v = md.version('bfcl-eval')
print(f'  bfcl-eval = {v}')
assert v == '2025.12.17', f'❌ 版本錯誤！要 2025.12.17，現在是 {v}'
print('  ✓ 版本正確')
"

echo ""
echo "=============================================="
echo "【檢查】題目檔在不在"
echo "=============================================="
python3 - <<'PY'
import bfcl_eval, os
root = os.path.dirname(bfcl_eval.__file__)
q = os.path.join(root, "data", "BFCL_v4_multi_turn_base.json")
a = os.path.join(root, "data", "possible_answer", "BFCL_v4_multi_turn_base.json")
for label, p in [("題目", q), ("標準答案", a)]:
    if os.path.exists(p):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  ✓ {label}：{n} 題  ({p})")
    else:
        print(f"  ✗ 找不到{label}：{p}")
PY

echo ""
echo "=============================================="
echo "【設定】讓 BFCL 連到我自己開的 vLLM"
echo "=============================================="
# 這一步很重要：預設 BFCL 會自己去啟動 vLLM，但那樣它會用它自己的參數。
# 你的研究就是在調參數，所以必須用你自己開的那個。
# 已用 `bfcl generate --help` 與原始碼確認（bfcl-eval 2025.12.17）：
#   正確名稱是 LOCAL_SERVER_ENDPOINT / LOCAL_SERVER_PORT，不是 VLLM_ENDPOINT / VLLM_PORT。
#   出處：bfcl_eval/model_handler/local_inference/base_oss_handler.py:42-43
#   而且光設環境變數還不夠，一定要加 --skip-server-setup，否則它會自己再開一個 vLLM。
export LOCAL_SERVER_ENDPOINT=${LOCAL_SERVER_ENDPOINT:-"localhost"}
export LOCAL_SERVER_PORT=${LOCAL_SERVER_PORT:-"8000"}

# 模型檔案位置。不指定的話 BFCL 會去 HF Hub 重抓 tokenizer/config，
# 指定後直接讀本機這份，確保跟 vLLM 正在服務的是同一套檔案。
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"$WORKDIR/models/Qwen3-8B"}

echo "  LOCAL_SERVER_ENDPOINT = $LOCAL_SERVER_ENDPOINT"
echo "  LOCAL_SERVER_PORT     = $LOCAL_SERVER_PORT"
echo "  LOCAL_MODEL_PATH      = $LOCAL_MODEL_PATH"
echo ""

echo "=============================================="
echo "【執行】跑 multi_turn_base"
echo "=============================================="
echo "  ⚠️ 不要加 --partial-eval，正式測試要跑完整批"
echo ""

# --skip-server-setup：用上面那個我們自己開的 vLLM，不要讓 BFCL 另外啟一個。
#   （--backend 只在沒有這個旗標時才會被用到，所以不必指定。
#     出處：base_oss_handler.py:135 的 `if not skip_server_setup:`）
bfcl generate \
  --model "$BFCL_MODEL" \
  --test-category multi_turn_base \
  --num-threads 1 \
  --skip-server-setup \
  --local-model-path "$LOCAL_MODEL_PATH" \
  2>&1 | tee "$WORKDIR/logs/bfcl_generate.log"

echo ""
echo "=============================================="
echo "【評分】"
echo "=============================================="
bfcl evaluate \
  --model "$BFCL_MODEL" \
  --test-category multi_turn_base \
  2>&1 | tee "$WORKDIR/logs/bfcl_evaluate.log"

echo ""
echo "=============================================="
echo "【對答案】"
echo "=============================================="
cat <<'EOF'

  把上面跑出來的分數，跟 leaderboard 的數字比：

  ┌──────────────────────────────────────────────┐
  │  模型              比哪一欄        官方數字   │
  ├──────────────────────────────────────────────┤
  │  Qwen3-8B (FC)     MT_Base         34.5       │
  │  Qwen3-8B (FC)     MultiTurn(平均) 41.75      │
  └──────────────────────────────────────────────┘

  ⚠️ 你只跑了 multi_turn_base 這一個子項目，
     所以要比 MT_Base 那一欄，不是 MultiTurn 平均！

  容許誤差：事先定好 ±5 個百分點

  ✓ 在範圍內  → 環境正確，可以往下做
  ✗ 超出範圍  → 停下來檢查：
       1. bfcl-eval 版本對不對
       2. model ID 有沒有拼對（-FC 後綴？）
       3. tool-call-parser 選對了嗎
       4. BFCL 是不是自己另外開了一個 vLLM（沒用到你的設定）

EOF
