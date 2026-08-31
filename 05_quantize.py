#!/usr/bin/env python3
"""
步驟 5：自己壓縮模型

用法：
    python3 05_quantize.py Qwen/Qwen3-8B /workspace/models/Qwen3-8B-my-w4a16

⚠️ 這一份的 recipe 抄自 RedHatAI 公開的作法。
   先用它壓 Qwen3-8B，跟官方版對答案，確認手藝沒問題，
   再用同一套去壓 xLAM 和 Nanbeige。
"""
import sys, os, json, time

model_stub = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
save_path  = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/Qwen3-8B-my-w4a16"

print("=" * 60)
print(f"要壓縮的模型 : {model_stub}")
print(f"存到         : {save_path}")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot

# ---------- 載入模型 ----------
# torch_dtype="auto" 很重要：不寫的話可能用 float32 載入，
# 8B × 4 bytes = 32GB，直接爆掉
print("\n[1/5] 載入模型（8B 大概 5-10 分鐘）...")
model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,   # Nanbeige 需要；其他模型加了也無害
)
tokenizer = AutoTokenizer.from_pretrained(model_stub, trust_remote_code=True)

# ---------- 校準資料 ----------
# 這是「壓縮時餵進去觀察模型行為」的那批文字。
# ⚠️ 所有模型都要用同一份，跨模型比較才乾淨。
print("\n[2/5] 載入校準資料...")
NUM_SAMPLES = 1024
MAX_SEQ_LEN = 8192

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))

# 資料集裡是 messages 格式（一輪一輪的對話），不是純文字。
# 要先套 chat template 轉成模型看得懂的樣子。
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# ---------- 壓縮設定 ----------
print("\n[3/5] 設定壓縮方式...")
recipe = GPTQModifier(
    ignore=["lm_head"],      # 最後一層不壓（壓了品質掉很多）
    targets="Linear",        # 只壓線性層
    scheme="W4A16",          # 權重 4 bit、運算 16 bit
    dampening_frac=0.01,     # GPTQ 的穩定參數，抄自 RedHatAI
)

print("  ignore         = ['lm_head']")
print("  targets        = Linear")
print("  scheme         = W4A16")
print("  dampening_frac = 0.01")
print(f"  校準樣本數     = {NUM_SAMPLES}")
print(f"  最長序列       = {MAX_SEQ_LEN}")

# ---------- 開始壓 ----------
print("\n[4/5] 開始壓縮（8B 約 10 分鐘～1 小時，看卡）...")
t0 = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
)

elapsed = time.time() - t0
print(f"  完成，花了 {elapsed/60:.1f} 分鐘")

# ---------- 存檔 ----------
print("\n[5/5] 存檔...")
model.save_pretrained(save_path, save_compressed=True)
tokenizer.save_pretrained(save_path)   # ⚠️ 這行不能少，vLLM 載入需要 tokenizer

# ---------- 檢查 ----------
print("\n" + "=" * 60)
print("檢查結果")
print("=" * 60)

total = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(save_path)
    for f in files
)
print(f"\n  壓縮後大小 : {total/1024**3:.2f} GB")

cfg = json.load(open(os.path.join(save_path, "config.json")))
qc = cfg.get("quantization_config", {})
print(f"\n  quantization_config：")
print(json.dumps(qc, indent=4, ensure_ascii=False))

print("""
=============================================================
下一步：跟 RedHatAI 的版本比對 config

    python3 06_compare_config.py

  要對上的欄位：num_bits=4、group_size=64、symmetric=false
  對不上代表 scheme="W4A16" 的預設值跟他們不同，
  要在 GPTQModifier 裡手動指定 group_size。
=============================================================
""")
