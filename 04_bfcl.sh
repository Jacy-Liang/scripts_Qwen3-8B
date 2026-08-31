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
export VLLM_ENDPOINT=${VLLM_ENDPOINT:-"localhost"}
export VLLM_PORT=${VLLM_PORT:-"8000"}

echo "  VLLM_ENDPOINT = $VLLM_ENDPOINT"
echo "  VLLM_PORT     = $VLLM_PORT"
echo ""
echo "  ⚠️ 這兩個環境變數名稱要跟你的 bfcl-eval 版本對得上。"
echo "     若跑起來發現它自己另外開了一個 vLLM，執行下面這行查正確寫法："
echo "     bfcl generate --help"
echo ""

echo "=============================================="
echo "【執行】跑 multi_turn_base"
echo "=============================================="
echo "  ⚠️ 不要加 --partial-eval，正式測試要跑完整批"
echo ""

bfcl generate \
  --model "$BFCL_MODEL" \
  --test-category multi_turn_base \
  --num-threads 1 \
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
