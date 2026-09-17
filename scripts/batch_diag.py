"""
batch_diag.py — run the coverage comparison over many charts at once.

For each image it compares three selections at the SAME budget:
  visionzip : what your method actually kept   (from the .npy dumps)
  uniform   : every Nth patch, ignores the image
  random    : random patches

Headline number is "ink hit rate": of the patches a method kept, what
share had actual content on them (not blank paper). Higher = better.

Usage:
    python batch_diag.py --csv chartqa_plain.csv --dumpdir /scratch/bh2863/diag \
        --out coverage_summary.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

import token_coverage_diag as tcd   # reuse ink_density / synthetic / measure


def ink_hit_rate(mask, ink, thr):
    """Share of KEPT patches that sit on content rather than blank paper."""
    kept = mask.astype(bool)
    return float((ink[kept] > thr).mean()) if kept.any() else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with an image_path column")
    p.add_argument("--dumpdir", required=True, help="folder holding rowN_kept.npy")
    p.add_argument("--merge", type=int, default=2,
                   help="Qwen merges patches 2x2, so token grid = thw[1:]//merge")
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--ink-thr", type=float, default=0.02,
                   help="a patch counts as 'content' above this ink fraction")
    p.add_argument("--out", default="coverage_summary.csv")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    rows, seen = [], set()

    for i, r in df.iterrows():
        f = os.path.join(a.dumpdir, f"row{i}_kept.npy")
        if not os.path.exists(f):
            continue
        img_path = r["image_path"]
        if img_path in seen:       # ChartQA repeats charts across questions
            continue
        seen.add(img_path)

        gfile = os.path.join(a.dumpdir, f"row{i}_grid.npy")
        if not os.path.exists(gfile):
            print(f"skip row{i}: no grid file")
            continue
        thw = np.load(gfile).ravel()          # [t, h, w] in pre-merge patches
        gh, gw = int(thw[1]) // a.merge, int(thw[2]) // a.merge

        idx = np.load(f).ravel().astype(int)
        if idx.size == 0 or idx.max() >= gh * gw:
            print(f"skip row{i}: index out of range for {gh}x{gw}")
            continue

        vz = np.zeros(gh * gw, dtype=np.uint8)
        vz[idx] = 1
        vz = vz.reshape(gh, gw)
        budget = idx.size / (gh * gw)

        img = Image.open(img_path)
        ink = tcd.ink_density(img, gh, gw)

        uni = tcd.synthetic("uniform", gh, gw, budget)
        rnd = tcd.synthetic("random", gh, gw, budget, seed=i)

        rec = {"image": os.path.basename(img_path), "grid": f"{gh}x{gw}",
               "budget": round(budget, 4)}
        for name, m in (("visionzip", vz), ("uniform", uni), ("random", rnd)):
            rec[f"{name}_ink_hit"] = round(ink_hit_rate(m, ink, a.ink_thr), 4)
            st = tcd.measure(m, ink, margin=a.margin)
            rec[f"{name}_margin_share"] = st["kept_share_in_margin"]
            rec[f"{name}_ink_corr"] = st["ink_kept_correlation"]
        rows.append(rec)

    if not rows:
        print("no dumps matched — check --dumpdir and --csv")
        return

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    print(f"\n{len(out)} unique charts\n")
    print(f"{'metric':<18}{'visionzip':>12}{'uniform':>12}{'random':>12}")
    for label, key in (("ink hit rate", "ink_hit"),
                       ("margin share", "margin_share"),
                       ("ink correlation", "ink_corr")):
        vals = [out[f"{m}_{key}"].astype(float).mean() for m in
                ("visionzip", "uniform", "random")]
        print(f"{label:<18}" + "".join(f"{v:>12.3f}" for v in vals))
    print(f"\nsaved {a.out}")
    print("ink hit rate = share of kept patches that had content on them "
          "(higher is better)")


if __name__ == "__main__":
    main()
