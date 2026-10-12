#!/usr/bin/env python3
"""
步驟 6：比對我壓的 vs RedHatAI 壓的（結構層驗證）

用法：
    python3 06_compare_config.py

這一步免費，不用跑模型，5 分鐘就知道壓對了沒。
一定要在花時間跑 BFCL 之前做。
"""
import json, os
from huggingface_hub import hf_hub_download

MINE   = os.environ.get("MINE", "/workspace/models/Qwen3-8B-my-w4a16")
THEIRS = "RedHatAI/Qwen3-8B-quantized.w4a16"
# 釘住 revision：對方日後更新模型時，比對基準才不會悄悄改變
THEIRS_REV = "32527053243a382bc56e941c964fc516a5014d39"
# 已下載過的話直接讀本機那份（例：/workspace/models/Qwen3-8B-w4a16-redhat）
THEIRS_LOCAL = os.environ.get("THEIRS_LOCAL", "")

print("=" * 60)
print("比對壓縮設定")
print("=" * 60)

# 抓對方的 config
if THEIRS_LOCAL and os.path.exists(os.path.join(THEIRS_LOCAL, "config.json")):
    theirs_path = os.path.join(THEIRS_LOCAL, "config.json")
    print(f"\n讀取本機的 {theirs_path}（請確認它是 revision {THEIRS_REV[:8]}）")
else:
    print(f"\n下載 {THEIRS}@{THEIRS_REV[:8]} 的 config.json ...")
    theirs_path = hf_hub_download(THEIRS, "config.json", revision=THEIRS_REV)
theirs = json.load(open(theirs_path)).get("quantization_config", {})

# 讀自己的
mine_path = os.path.join(MINE, "config.json")
if not os.path.exists(mine_path):
    print(f"✗ 找不到 {mine_path}，先跑 05_quantize.py")
    raise SystemExit(1)
mine = json.load(open(mine_path)).get("quantization_config", {})

def dig(cfg):
    """把巢狀的設定攤平成好比較的樣子"""
    out = {}
    groups = cfg.get("config_groups", {})
    # 依順序編號（group[0]、group[1]...）而不是用名稱比對：群組名稱本身不影響配方，
    # 而 RedHatAI 的 recipe.yaml 寫 group0、config.json 寫 group_0，名稱很容易不一致
    for idx, (_, g) in enumerate(sorted(groups.items())):
        gname = f"group[{idx}]"
        w = g.get("weights", {})
        # actorder / observer / zp_dtype 也要比：對稱與否、actorder 都是配方的一部分
        for k in ("num_bits", "group_size", "symmetric", "strategy", "type",
                  "actorder", "observer", "zp_dtype"):
            if k in w:
                out[f"{gname}.weights.{k}"] = w[k]
        if "targets" in g:
            out[f"{gname}.targets"] = g["targets"]
    for k in ("quant_method", "format", "ignore", "quantization_status"):
        if k in cfg:
            out[k] = cfg[k]
    return out

A, B = dig(theirs), dig(mine)

print("\n" + "=" * 60)
print(f"{'欄位':<32} {'RedHatAI':<20} {'我的':<20} 一致")
print("=" * 60)

same = True
for k in sorted(set(A) | set(B)):
    a, b = A.get(k, "—"), B.get(k, "—")
    ok = (a == b)
    if not ok:
        same = False
    print(f"{k:<32} {str(a):<20} {str(b):<20} {'✓' if ok else '✗'}")

print("=" * 60)
if same:
    print("\n✅ 完全一致，壓縮設定正確，可以往下跑 BFCL")
else:
    print("""
⚠️ 有欄位對不上。

RedHatAI 實際使用的配方（隨模型發布的 recipe.yaml 第 10–11 行）是：
  int4、strategy=group、group_size=128、symmetric=false、
  actorder=weight、observer=minmax，ignore=[lm_head]
（README 的範例寫 scheme="W4A16"，那是對稱量化，跟 recipe.yaml 不一致，以 recipe.yaml 為準）

常見原因：
  - symmetric 不同：用了 scheme="W4A16" preset（ct 0.19 的 preset 是對稱）
  - actorder 顯示 static 對 weight：static 是 weight 的別名，語意相同，
    但字串不同。05_quantize.py 應明確寫 "weight"
  - quantization_status、version 這類欄位不同屬於正常，不影響配方

修法：05_quantize.py 的 GPTQModifier 要用明確的 config_groups：

    recipe = GPTQModifier(
        ignore=["lm_head"],
        dampening_frac=0.01,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4, "type": "int", "strategy": "group",
                    "group_size": 128, "symmetric": False,
                    "actorder": "weight", "observer": "minmax",
                },
            },
        },
    )

（這段已在 llmcompressor 0.14.0 實測：resolve_quantization_config() 解析結果與上面一致）
""")

# 順便比檔案大小
print("\n" + "=" * 60)
print("檔案大小比較")
print("=" * 60)
mine_size = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(MINE) for f in fs
)
mine_st = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(MINE) for f in fs if f.endswith(".safetensors"))
print(f"  我的     : {mine_st/1024**3:.2f} GiB（.safetensors 合計；整個資料夾 {mine_size/1024**3:.2f} GiB，含 tokenizer 等）")
print(f"  RedHatAI : 5.68 GiB（{THEIRS_REV[:8]} 的兩個 .safetensors 合計 6,098,617,040 bytes）")
print(f"  → 差距應該在 1% 以內")
