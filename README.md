# RunPod 上機清單（RTX 5090 32GB）

## 開 Pod 前先確認三件事

1. **顯卡選 RTX 5090（32GB）** — 只有它能跑 8B 不壓縮的版本
2. **CUDA 版本要 12.8 以上** — 5090 是新架構（sm_120），舊 CUDA 跑不動
3. **一定要掛 Network Volume** — 不然關機後模型全部消失，換卡要重下載一次

> Network Volume 要在建立 Pod 的時候就選，事後補很麻煩。
> 建議掛在 `/workspace`，下面的腳本都預設用這個路徑。

---

## 執行順序

```bash
# 設定工作目錄（如果掛了 Volume）
export WORKDIR=/workspace

# 步驟 0：檢查環境、安裝套件
bash 00_setup.sh

# 步驟 1：下載 Qwen3-8B
bash 01_download.sh

# 步驟 2：啟動 vLLM（這個視窗會一直佔著，不要關）
bash 02_serve.sh /workspace/models/Qwen3-8B qwen3-8b-fp16

# ↓ 另外開一個終端機視窗 ↓

# 步驟 3：抓記憶體數字
python3 03_capture.py qwen3-8b-fp16

# 步驟 4：跑 BFCL 對答案 ← 這是閘門
bash 04_bfcl.sh
```

**步驟 4 沒過就停下來修，不要往下做。**

---

## 過了閘門之後

```bash
# 步驟 5：自己壓一次（驗證手藝）
python3 05_quantize.py Qwen/Qwen3-8B /workspace/models/Qwen3-8B-my-w4a16

# 步驟 6：比對 config（免費，5 分鐘）
python3 06_compare_config.py

# 對上了 → 重複步驟 2-4，這次用壓縮版
bash 02_serve.sh /workspace/models/Qwen3-8B-my-w4a16 qwen3-8b-my-int4
python3 03_capture.py qwen3-8b-my-int4
bash 04_bfcl.sh
```

---

## 常見問題怎麼判斷

### 啟動就 OOM

**先降 `--gpu-memory-utilization`**：

```bash
GPU_UTIL=0.75 bash 02_serve.sh /workspace/models/Qwen3-8B qwen3-8b-fp16
```

還是不行就降 `--max-model-len`：

```bash
GPU_UTIL=0.75 MAX_LEN=8192 bash 02_serve.sh ...
```

**8B FP16 在 5090（32GB）上應該跑得起來**（權重 16GB + KV cache）。
如果連 5090 都爆，代表 `--max-model-len` 設太大。

### 看 log 要往上找

錯誤訊息通常是一連串，**最下面那行是「結果」，往上找才是「原因」**。

例如最下面寫 `RuntimeError: CUDA error`，往上翻可能看到
`No kernel image is available for execution on the device`
→ 這是 CUDA 版本跟顯卡架構不合，要換 RunPod 樣板。

### BFCL 分數差很多

按這個順序查：

1. `bfcl-eval` 版本是不是 `2025.12.17`
2. model ID 拼對了嗎（`Qwen/Qwen3-8B-FC` 有沒有漏掉 `-FC`）
3. 比錯欄位了嗎（跑 `multi_turn_base` 要比 `MT_Base`，不是 `MultiTurn` 平均）
4. BFCL 是不是自己另外開了一個 vLLM（那樣就沒用到你的設定）

第 4 點怎麼確認：跑 BFCL 的時候看你的 vLLM 視窗有沒有出現請求紀錄。
沒有的話代表它連到別的地方去了。

---

## 每一格要記錄的六個數字

| 欄位 | 從哪來 |
|---|---|
| ① 載入成功嗎 | vLLM 有沒有正常啟動 |
| ② 權重佔多少 GB | 啟動 log 的 `weights_memory` |
| ③ KV cache 分到多少 | 啟動 log 的 `GPU KV cache size` |
| ④ 能容納幾個 token | 同上 |
| ⑤ gpu_util 上限 | 從 0.85 往上試，爆掉前的最大值 |
| ⑥ 實際用到的 kernel | 啟動 log 裡找 `marlin` 字樣 |

**第 ⑥ 項特別重要**：壓縮模型跑得快不快完全看 kernel 對不對，
沒對上速度可能差十倍。找到就截圖存檔，論文附錄要用。

---

## 這些腳本我沒辦法幫你實測

我這裡沒有 GPU 也連不到 RunPod，所以：

- **參數名稱可能因 vLLM 版本而異** — 報錯就跑 `vllm serve --help` 查
- **BFCL 的環境變數名稱要自己確認** — 跑 `bfcl generate --help`
- **`06_compare_config.py` 裡那段修正語法我沒驗證過** — 要對照
  llm-compressor 官方文件

遇到報錯把訊息貼給我，我幫你看。
