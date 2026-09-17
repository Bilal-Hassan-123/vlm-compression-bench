import pandas as pd
import sys

timestamp_dir = sys.argv[1]

df = pd.read_csv('chartqa_predictions_hiprune_qwen_baseline.csv')
df = df.drop(columns=['image_path'])
out_path = f'outputs/hiprune_baseline_qwen2_5_vl_3b/{timestamp_dir}/hiprune_baseline_qwen2_5_vl_3b_ChartQA_TEST.xlsx'
df.to_excel(out_path, index=False)
print(f"Saved {df.shape} to {out_path}")
