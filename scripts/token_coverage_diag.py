"""
token_coverage_diag.py
----------------------
Diagnostic for chart-specific visual token compression.

Answers one question: when a pruning method keeps K of N visual tokens on a
chart, WHERE do the kept tokens land -- and does it abandon the ink-dense
margins (axis tick labels, legend) where chart answers live?

Two things it produces:
  1. an overlay PNG per image (kept patches bright, dropped patches dimmed)
  2. numbers: ink density per cell vs kept density per cell, and a
     margin-band coverage ratio you can put in a table

No torch, no model. You feed it the kept indices your method already
produces. See DUMPING KEPT INDICES at the bottom.

Usage
-----
    python token_coverage_diag.py \
        --image chart.png \
        --kept kept_idx.npy \
        --grid 34 46 \
        --out out/

    # sanity-check the plumbing with no method at all:
    python token_coverage_diag.py --image chart.png --grid 34 46 \
        --synthetic center --out out/
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------


def token_grid_shape(image_grid_thw, merge_size=2):
    """Qwen2.5-VL: image_grid_thw is in 14px PATCH units, before the 2x2 merge.

    The tokens the LLM actually receives are on a grid merge_size times
    coarser in each direction. Pass the raw [t, h, w] you got from the
    processor and this gives you the token grid you should reshape to.
    """
    t, h, w = image_grid_thw
    return int(h) // merge_size, int(w) // merge_size


def ink_density(image, grid_h, grid_w, bg_percentile=90):
    """Fraction of non-background pixels per token cell.

    Background is estimated as the modal bright value rather than assumed
    white, so dark-themed and tinted charts still work. Returns a
    (grid_h, grid_w) array in [0, 1].
    """
    g = np.asarray(image.convert("L"), dtype=np.float32)
    bg = np.percentile(g, bg_percentile)
    ink = (np.abs(g - bg) > 25).astype(np.float32)  # 25/255 tolerance

    H, W = ink.shape
    # trim to a multiple of the grid so reshape is exact
    H2, W2 = (H // grid_h) * grid_h, (W // grid_w) * grid_w
    ink = ink[:H2, :W2]
    return ink.reshape(grid_h, H2 // grid_h, grid_w, W2 // grid_w).mean(axis=(1, 3))


def pool(a, r):
    """Average-pool a (grid_h, grid_w) map down to roughly r x r cells."""
    gh, gw = a.shape
    ph, pw = max(1, gh // r), max(1, gw // r)
    gh2, gw2 = (gh // ph) * ph, (gw // pw) * pw
    return a[:gh2, :gw2].reshape(gh2 // ph, ph, gw2 // pw, pw).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------


def overlay(image, kept_mask, out_path, dim=0.18, tint=(255, 60, 60)):
    """Kept patches shown normally; dropped patches dimmed and tinted.

    What the model still sees is what stays bright.
    """
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    H, W, _ = arr.shape
    gh, gw = kept_mask.shape

    m = np.asarray(Image.fromarray((kept_mask * 255).astype(np.uint8)).resize(
        (W, H), Image.NEAREST), dtype=np.float32)[..., None] / 255.0

    tint_arr = np.array(tint, dtype=np.float32)
    dropped = arr * dim + tint_arr * (1 - dim) * 0.25
    out = arr * m + dropped * (1 - m)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_path)


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------


def measure(kept_mask, ink, margin=0.15, cells=4):
    """Quantify whether selection tracks ink, and whether margins survive.

    margin_coverage_ratio is the headline number:
        (share of kept tokens in the margin band)
      / (share of total ink in the margin band)

    ~1.0  selection spends on the margins in proportion to their content
    <1.0  margins are UNDER-served -> axis labels and legends are being
          dropped, which is the premise of ink-weighted stratification
    """
    gh, gw = kept_mask.shape
    left = max(1, int(round(gw * margin)))
    bottom = max(1, int(round(gh * margin)))

    band = np.zeros_like(kept_mask, dtype=bool)
    band[:, :left] = True          # y-axis labels
    band[gh - bottom:, :] = True   # x-axis labels

    kept_total = kept_mask.sum()
    ink_total = ink.sum()
    kept_band = kept_mask[band].sum() / kept_total if kept_total else 0.0
    ink_band = ink[band].sum() / ink_total if ink_total else 0.0

    k_cell = pool(kept_mask.astype(np.float32), cells)
    i_cell = pool(ink, cells)

    # cells holding real content that selection almost entirely skipped
    starved = int(((i_cell > i_cell.mean()) & (k_cell < 0.25 * k_cell.mean())).sum())

    kf, inf_ = k_cell.ravel(), i_cell.ravel()
    corr = (float(np.corrcoef(kf, inf_)[0, 1])
            if kf.std() > 1e-8 and inf_.std() > 1e-8 else float("nan"))

    return {
        "tokens_total": int(kept_mask.size),
        "tokens_kept": int(kept_total),
        "budget": round(float(kept_total / kept_mask.size), 4),
        "kept_share_in_margin": round(float(kept_band), 4),
        "ink_share_in_margin": round(float(ink_band), 4),
        "margin_coverage_ratio": (round(float(kept_band / ink_band), 4)
                                  if ink_band > 1e-8 else None),
        "ink_kept_correlation": (None if np.isnan(corr) else round(corr, 4)),
        "starved_cells": starved,
        "cells": cells * cells,
    }


# --------------------------------------------------------------------------
# synthetic selections, for testing the plumbing before wiring your method
# --------------------------------------------------------------------------


def synthetic(kind, gh, gw, budget=0.223, seed=0):
    n = gh * gw
    k = max(1, int(round(n * budget)))
    mask = np.zeros((gh, gw), dtype=np.uint8)
    if kind == "center":
        # crude stand-in for salience-driven global top-k: clusters centrally
        yy, xx = np.mgrid[0:gh, 0:gw]
        d = ((yy - gh / 2) ** 2 + (xx - gw / 2) ** 2).ravel()
        mask.ravel()[np.argsort(d)[:k]] = 1
    elif kind == "random":
        rng = np.random.default_rng(seed)
        mask.ravel()[rng.choice(n, k, replace=False)] = 1
    elif kind == "uniform":
        step = n / k
        mask.ravel()[(np.arange(k) * step).astype(int)] = 1
    else:
        raise ValueError(kind)
    return mask


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--kept", help=".npy or .json list of kept token indices")
    p.add_argument("--grid", nargs=2, type=int, required=True,
                   metavar=("H", "W"), help="TOKEN grid (post 2x2 merge)")
    p.add_argument("--synthetic", choices=["center", "random", "uniform"],
                   help="fake a selection instead of loading --kept")
    p.add_argument("--budget", type=float, default=0.223)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--out", default="out")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    gh, gw = a.grid
    img = Image.open(a.image)

    if a.synthetic:
        mask = synthetic(a.synthetic, gh, gw, a.budget)
        tag = f"synthetic-{a.synthetic}"
    else:
        if not a.kept:
            p.error("pass --kept or --synthetic")
        idx = (np.load(a.kept) if a.kept.endswith(".npy")
               else np.array(json.load(open(a.kept))))
        idx = np.asarray(idx).ravel().astype(int)
        if idx.max() >= gh * gw:
            p.error(f"index {idx.max()} exceeds grid {gh}x{gw}={gh*gw}. "
                    "Wrong grid? Remember Qwen's image_grid_thw is pre-merge — "
                    "divide h and w by 2.")
        mask = np.zeros(gh * gw, dtype=np.uint8)
        mask[idx] = 1
        mask = mask.reshape(gh, gw)
        tag = os.path.splitext(os.path.basename(a.kept))[0]

    ink = ink_density(img, gh, gw)
    stem = os.path.splitext(os.path.basename(a.image))[0]

    overlay(img, mask, os.path.join(a.out, f"{stem}__{tag}__overlay.png"))
    stats = measure(mask, ink, margin=a.margin, cells=a.cells)
    stats.update(image=os.path.basename(a.image), selection=tag,
                 grid=[gh, gw], margin=a.margin)

    with open(os.path.join(a.out, f"{stem}__{tag}__stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    r = stats["margin_coverage_ratio"]
    if r is not None:
        if r < 0.8:
            verdict = "margins UNDER-served — ink-weighted allocation has a target"
        elif r > 1.2:
            verdict = "margins OVER-served relative to their ink"
        else:
            verdict = "margins served in proportion to their ink — premise is weak"
        print(f"\nmargin_coverage_ratio = {r}  ->  {verdict}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# DUMPING KEPT INDICES FROM YOUR EXISTING RUN
# --------------------------------------------------------------------------
# Wherever your VisionZip/HiPrune port picks the surviving tokens, it ends up
# with an index tensor. Save it right there and stop:
#
#     import numpy as np, os
#     if os.environ.get("DUMP_KEPT"):
#         np.save(os.environ["DUMP_KEPT"], keep_idx.detach().cpu().numpy())
#
# Then one image at a time:
#
#     DUMP_KEPT=kept_idx.npy python your_eval.py --limit 1
#
# Two traps:
#   * Qwen's image_grid_thw is in 14px patch units, BEFORE the 2x2 merge.
#     Token grid = (h // 2, w // 2). Pass that to --grid.
#   * If your indices are per-tile (LLaVA-NeXT AnyRes) rather than flat,
#     flatten them first — same bug class as the CDPruner tile-flatten fix.
# --------------------------------------------------------------------------
