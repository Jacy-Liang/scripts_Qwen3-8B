#!/usr/bin/env python3
"""
步驟 3：抓記憶體數字（第 4 項待辦的核心）

用法：
    python3 03_capture.py qwen3-8b-fp16

要在 vLLM 已經啟動、而且顯示 "Application startup complete" 之後才跑。
另外開一個終端機視窗執行。
"""
import os, re, sys, json, urllib.request

WORKDIR = os.environ.get("WORKDIR", "/workspace")
TAG  = sys.argv[1] if len(sys.argv) > 1 else "unknown"
PORT = os.environ.get("PORT", "8000")
LOG  = f"{WORKDIR}/logs/serve_{TAG}.log"

print("=" * 60)
print(f"抓取 {TAG} 的記憶體數字")
print("=" * 60)

# ---------- 來源一：啟動 log ----------
# vLLM 啟動時會印一行記憶體分析結果，數字全在裡面。
# 不同版本格式不一樣，所以用關鍵字去找，找到就整行印出來。
print("\n【來源一】啟動 log 裡的記憶體資訊\n")

KEYWORDS = [
    "memory profiling", "weights_memory", "kv cache", "kv_cache",
    "gpu blocks", "maximum concurrency", "graph capturing",
    "marlin", "quantization", "awq", "gptq",
]

found = []
try:
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.lower()
            if any(k in low for k in KEYWORDS):
                found.append(line.rstrip())
except FileNotFoundError:
    print(f"  ✗ 找不到 log：{LOG}")
    print("    確認 02_serve.sh 有跑過，而且 TAG 拼對了")

if found:
    for line in found:
        print("  " + line)
else:
    print("  （沒抓到關鍵字。手動看 log：）")
    print(f"    grep -inE 'memory|kv|marlin|quant' {LOG} | head -40")

# ---------- 來源二：/metrics 端點 ----------
# vLLM 會開一個網頁端點，把即時統計吐出來。
print("\n【來源二】/metrics 即時統計\n")

def get_metrics(port):
    url = f"http://localhost:{port}/metrics"
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode()

WANT = {
    "vllm:cache_config_info":            "快取設定",
    "vllm:gpu_cache_usage_perc":         "KV cache 已用比例",
    "vllm:prefix_cache_queries_total":   "prefix cache 查詢次數",
    "vllm:prefix_cache_hits_total":      "prefix cache 命中次數",
    "vllm:prompt_tokens_total":          "輸入 token 累計",
    "vllm:generation_tokens_total":      "輸出 token 累計",
    "vllm:num_requests_running":         "正在處理的請求數",
}

metrics = {}
try:
    text = get_metrics(PORT)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, label in WANT.items():
            if line.startswith(key):
                print(f"  {label:<22} {line}")
                m = re.search(r"\s([0-9.eE+-]+)$", line)
                if m:
                    metrics[key] = float(m.group(1))
except Exception as e:
    print(f"  ✗ 抓不到：{e}")
    print("    確認 vLLM 還在跑，而且 port 對")

# ---------- 從 config 算 KV cache 上限 ----------
print("\n【推算】這個配置能開幾個 agent\n")

cfg_path = None
for line in found:
    m = re.search(r"(/\S*?)/config\.json", line)
    if m:
        cfg_path = m.group(0)
        break

print("  公式：可開併發數 = KV cache token 數 ÷ 每個 agent 的 context 長度")
print("  KV cache token 數請從上面 log 裡的 'GPU KV cache size' 或")
print("  'Maximum concurrency' 那一行抓。")
print("")
print("  例：KV cache 容納 142,336 tokens，每個 agent 用 16,384")
print("      142,336 ÷ 16,384 ≈ 8.7 → 最多 8 個 agent")

# ---------- 存檔 ----------
out = f"{WORKDIR}/results/memory_{TAG}.json"
os.makedirs(f"{WORKDIR}/results", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"tag": TAG, "log_lines": found, "metrics": metrics},
              f, ensure_ascii=False, indent=2)

print(f"\n✓ 已存到 {out}")
print("\n要填進可用性表的欄位：")
print("  ① 載入成功嗎          ② 權重佔多少 GB")
print("  ③ KV cache 分到多少   ④ 能容納幾個 token")
print("  ⑤ gpu_util 上限       ⑥ 實際用到的 kernel（找 marlin 字樣）")
