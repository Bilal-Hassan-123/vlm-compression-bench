# VLM Token Compression Benchmark

Benchmarking published **visual-token compression** methods for vision-language models (VLMs) on chart question answering.
Compression drops most of the image tokens a VLM reads, to make it faster. This project measures how much accuracy that costs.

Undergraduate research, eBrain Lab, NYU Abu Dhabi (2026). Experiments ran on NVIDIA A100 GPUs on NYUAD's Jubail HPC cluster.

## Results (ChartQA, 2,500 questions per run)

Token budget = share of image tokens kept. Retained = compressed accuracy ÷ baseline accuracy.

| Model | Method | Budget | Baseline | Compressed | Retained |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | VisionZip | 22.3% | 82.84% | 69.28% | **84%** |
| Qwen2.5-VL-3B | HiPrune | 22.3% | 82.84% | 61.36% | 74% |
| Qwen2.5-VL-3B | CDPruner | 22.3% | 82.84% | 47.36% | 57% |
| LLaVA-NeXT-7B | VisionZip | 22.2% | 54.56% | 41.80% | **77%** |
| LLaVA-NeXT-7B | DivPrune¹ | 22.3% | 54.56% | 35.92% | 66% |
| LLaVA-NeXT-7B | CDPruner | 22.3% | 54.56% | 29.16% | 53% |
| InternVL3-8B | CDPruner | 22.3% | 77.04% | 42.72% | 55% |

**Takeaway:** at the same budget, VisionZip keeps 77–84% of baseline accuracy, while CDPruner stays at 53–57% across three different models.

Reference points (LLaVA-1.5-7B, 33% budget) and the ChartQA Pro run are in [`results/summary.csv`](results/summary.csv).

¹ DivPrune was scored with lmms-eval; all other rows with VLMEvalKit.

### Extra experiment: stacking VisionZip + DivPrune
On Qwen2.5-VL-3B at a fixed 22.3% budget, three hybrid settings all scored below stock VisionZip (69.28%): 68.28%, 67.92%, 64.36%.
A control run with the hybrid switched off reproduced 69.28% exactly.

## Methods that didn't make the table
- **VisionZip on InternVL3:** not portable, since the vision encoder doesn't expose the attention weights VisionZip needs.
- **BTP (NeurIPS '25):** released code failed its own smoke test at the authors' default budget, so no number is reported.
- **SparseVLM:** only works with LLaVA-1.5 in practice.

## Repo layout
```
scripts/   inference (infer_*), result packaging (package_*), and dataset export (export_*) scripts
slurm/     SLURM job files used to run them on the cluster
results/   per-question predictions for every run + summary.csv
patches/   the one change made to VLMEvalKit (registering a model name)
envs/      Python package list
```

## Pipeline
1. `export_*.py` converts a benchmark into a plain CSV.
2. `infer_*.py` runs a model (with or without compression) and saves its answers.
3. `package_*.py` converts the answers into VLMEvalKit's format.
4. `eval_*.sbatch` scores them with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

Datasets (ChartQA, ChartQA Pro, PlotQA) are not included. They are downloaded by VLMEvalKit.

## Validation
Before reporting a number, I checked it against the authors' published result where possible
(e.g. DivPrune on MME: 1363 measured vs. 1328 published).
