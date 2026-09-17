"""
token_counts.py — how many tokens does each ChartQA image become?

No model, no GPU. Just Qwen's image processor, which is what decides
the grid. Run from the VLMEvalKit folder (image paths are relative).

    python token_counts.py --csv chartqa_plain.csv
"""

import argparse
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument("--csv", default="chartqa_plain.csv")
p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
p.add_argument("--ratio", type=float, default=0.207, help="VisionZip dominant ratio")
p.add_argument("--floor", type=int, default=64, help="floor to test")
p.add_argument("--out", default="token_counts.csv")
a = p.parse_args()

proc = AutoProcessor.from_pretrained(a.model)
df = pd.read_csv(a.csv)

rows, seen = [], set()
for path in df["image_path"]:
    if path in seen:
        continue
    seen.add(path)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print("skip", path, e)
        continue
    thw = proc.image_processor(images=img, return_tensors="pt")["image_grid_thw"][0]
    tokens = int(thw[1]) * int(thw[2]) // 4      # 2x2 merge
    rows.append({"image": path,
                 "grid": f"{int(thw[1])//2}x{int(thw[2])//2}",
                 "tokens": tokens,
                 "kept_now": int(a.ratio * tokens),
                 "kept_with_floor": max(a.floor, int(a.ratio * tokens))})

out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)

t, k = out["tokens"], out["kept_now"]
print(f"\n{len(out)} unique charts\n")
print(f"tokens per image : min {t.min()}, median {int(t.median())}, max {t.max()}")
print(f"kept now         : min {k.min()}, median {int(k.median())}, max {k.max()}")
for n in (16, 32, 64, 128):
    share = (k < n).mean()
    print(f"  images keeping fewer than {n:>3} tokens: {share:6.1%}  ({(k < n).sum()} charts)")

extra = (out["kept_with_floor"] - out["kept_now"]).sum()
print(f"\nwith a floor of {a.floor}: "
      f"{(out['kept_with_floor'] > out['kept_now']).mean():.1%} of charts get more tokens, "
      f"average budget {out['kept_with_floor'].mean()/t.mean():.1%} vs {k.mean()/t.mean():.1%} now")
print(f"saved {a.out}")

