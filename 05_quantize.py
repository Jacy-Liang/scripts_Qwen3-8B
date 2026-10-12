#!/usr/bin/env python3
"""
步驟 5：自己壓縮模型
用法（一定要用量化 venv 的 python，見 07_setup_quant_env.sh）：
    /workspace/venv-quant/bin/python 05_quantize.py <原始模型> <輸出資料夾>
    /workspace/venv-quant/bin/python 05_quantize.py /workspace/models/Qwen3-8B /workspace/models/Qwen3-8B-my-w4a16

⚠️ recipe 照抄 RedHatAI/Qwen3-8B-quantized.w4a16 隨模型發布的 recipe.yaml
   （revision 32527053243a382bc56e941c964fc516a5014d39）：
     GPTQ、int4、group 128、非對稱（symmetric: false）、actorder: weight、
     observer: minmax、ignore lm_head、dampening_frac 0.01
   注意：該模型 README 的範例寫的是 scheme="W4A16"（對稱），跟實際發布的
   recipe.yaml 不一致。以 recipe.yaml 為準，因為它跟 config.json 的結果相符。
   六個模型全部用這同一套配方，跨模型比較才乾淨，也有外部依據。

   驗證目標：同配方下，自壓模型的 BFCL 分數落在公開模型附近（不宣稱逐位元
   重現：RedHatAI 的校準樣本 seed 未公開，llmcompressor 版本也不同）。
"""
import sys, os, json, time

model_stub = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen3-8B"
save_path  = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/Qwen3-8B-my-w4a16"

print("=" * 60)
print(f"要壓縮的模型 : {model_stub}")
print(f"存到         : {save_path}")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier
# llmcompressor 0.14 已移除 llmcompressor.transformers.oneshot，舊路徑會 ImportError
from llmcompressor import oneshot
from llmcompressor.utils.pytorch import infer_sequential_targets

# ---------- 載入模型 ----------
# dtype="auto" 很重要：不寫的話可能用 float32 載入，
# 8B × 4 bytes = 32GB，直接爆掉。（transformers 5 把 torch_dtype 改名為 dtype）
print("\n[1/5] 載入模型（8B 大概 5-10 分鐘）...")
model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,   # Nanbeige 需要；其他模型加了也無害
)
tokenizer = AutoTokenizer.from_pretrained(model_stub, trust_remote_code=True)

# ---------- 校準資料 ----------
# 這是「壓縮時餵進去觀察模型行為」的那批文字。
# ⚠️ 所有模型都要用同一份，跨模型比較才乾淨。
# 資料集、樣本數、最長序列都照 RedHatAI 的 README（Creation 段）。
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
# 每個欄位都明確寫出，不依賴 preset 與預設值（之後 llmcompressor 改預設也不受影響）：
#   - 不用 scheme="W4A16"：ct 0.19 的這個 preset 是對稱量化，跟 RedHatAI 不同
#   - actorder 寫 "weight" 而不是靠 GPTQModifier 的預設 "static"：兩者語意相同
#     （static 是 weight 的別名），但寫進 config.json 的字串會照原樣保留，
#     寫 weight 才能跟 RedHatAI 的 config.json 逐字比對
#   - sequential_targets 不寫在這裡：0.14 已移到 oneshot()，寫在 GPTQModifier
#     會報 extra_forbidden。不指定時由模型的 _no_split_modules 推斷，
#     Qwen3 推斷出來是 Qwen3DecoderLayer，跟 RedHatAI 的 recipe.yaml 相同
print("\n[3/5] 設定壓縮方式...")
WEIGHTS = {
    "num_bits": 4,
    "type": "int",
    "strategy": "group",
    "group_size": 128,
    "symmetric": False,
    "actorder": "weight",
    "observer": "minmax",
}
recipe = GPTQModifier(
    ignore=["lm_head"],      # 最後一層不壓（壓了品質掉很多）
    dampening_frac=0.01,     # GPTQ 的穩定參數
    config_groups={
        # 群組名稱用 group_0，跟 RedHatAI 的 config.json 一致（他們的 recipe.yaml 寫 group0）
        "group_0": {
            "targets": ["Linear"],   # 只壓線性層
            "weights": WEIGHTS,
        },
    },
)

seq_targets = infer_sequential_targets(model)
print("  ignore             = ['lm_head']")
print("  targets            = ['Linear']")
print(f"  weights            = {WEIGHTS}")
print("  dampening_frac     = 0.01")
print(f"  sequential_targets = {seq_targets}（自動推斷）")
print(f"  校準樣本數         = {NUM_SAMPLES}（seed=42 洗牌後取前 {NUM_SAMPLES} 筆）")
print(f"  最長序列           = {MAX_SEQ_LEN}")

# ---------- 開始壓 ----------
print("\n[4/5] 開始壓縮（8B 約 10 分鐘～1 小時，看卡）...")
t0 = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
    # oneshot 預設會再洗一次牌；上面已經用 seed=42 洗過，關掉才能讓
    # 六個模型拿到同一組、同一順序的 1024 筆
    shuffle_calibration_samples=False,
)

elapsed = time.time() - t0
print(f"  完成，花了 {elapsed/60:.1f} 分鐘")

# ---------- 存檔 ----------
# llmcompressor 存檔時會自動在輸出資料夾寫一份 recipe.yaml（實際使用的配方）
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
print(f"\n  壓縮後大小 : {total/1024**3:.2f} GiB")

cfg = json.load(open(os.path.join(save_path, "config.json")))
qc = cfg.get("quantization_config", {})
print(f"\n  quantization_config：")
print(json.dumps(qc, indent=4, ensure_ascii=False))

recipe_file = os.path.join(save_path, "recipe.yaml")
print(f"\n  recipe.yaml：{'有' if os.path.exists(recipe_file) else '⚠️ 沒有，請檢查'}（{recipe_file}）")

print("""
=============================================================
下一步：跟 RedHatAI 的版本比對 config

    MINE=<輸出資料夾> python3 06_compare_config.py

  要對上的欄位（RedHatAI recipe.yaml 第 10–11 行 / config.json 第 29–38 行）：
    num_bits=4、type=int、strategy=group、group_size=128、
    symmetric=false、actorder=weight、observer=minmax，以及 ignore=[lm_head]
=============================================================
""")
