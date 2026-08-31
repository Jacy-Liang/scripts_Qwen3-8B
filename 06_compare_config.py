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

print("=" * 60)
print("比對壓縮設定")
print("=" * 60)

# 抓對方的 config
print(f"\n下載 {THEIRS} 的 config.json ...")
theirs_path = hf_hub_download(THEIRS, "config.json")
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
    for gname, g in groups.items():
        w = g.get("weights", {})
        for k in ("num_bits", "group_size", "symmetric", "strategy", "type"):
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
⚠️ 有欄位對不上。最常見的是 group_size。

RedHatAI 的說明寫「非對稱 per-group、group size 64」，
但他們公開的 recipe 只寫 scheme="W4A16"，沒指定 group_size。
llm-compressor 的預設值可能是 128。

修法：在 05_quantize.py 的 GPTQModifier 裡改成明確指定，例如：

    from llmcompressor.modifiers.quantization import GPTQModifier
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType
    )

    recipe = GPTQModifier(
        ignore=["lm_head"],
        dampening_frac=0.01,
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=64,          # ← 對齊 RedHatAI
                    symmetric=False,        # ← 非對稱
                ),
            )
        },
    )

⚠️ 上面這段語法我沒有實際跑過驗證，請對照
   llm-compressor 官方文件確認 import 路徑與參數名稱。
""")

# 順便比檔案大小
print("\n" + "=" * 60)
print("檔案大小比較")
print("=" * 60)
mine_size = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(MINE) for f in fs
)
print(f"  我的     : {mine_size/1024**3:.2f} GB")
print(f"  RedHatAI : 到他們的 Files 頁面看 .safetensors 總和")
print(f"  → 差距應該在 1% 以內")
