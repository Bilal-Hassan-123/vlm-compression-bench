import pandas as pd

# Read the already-localized TSV directly — no need to call build_dataset again,
# VLMEvalKit already decoded images to real files on our last check run.
df = pd.read_csv('/scratch/bh2863/fastv_project/LMUData/PlotQA_local.tsv', sep='\t')
print(f"Full dataset: {len(df)} rows")

# Subsample — fixed seed so this is reproducible if we need to rerun
N = 3000
sample = df.sample(n=N, random_state=42).reset_index(drop=True)

sample[['index', 'image_path', 'question', 'answer']].to_csv('plotqa_plain.csv', index=False)
print(f"Saved {len(sample)} rows to plotqa_plain.csv")
