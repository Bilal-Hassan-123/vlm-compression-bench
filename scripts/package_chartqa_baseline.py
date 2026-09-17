import pandas as pd
import sys

timestamp_dir = sys.argv[1]

df = pd.read_csv('chartqa_predictions_baseline.csv')
df = df.drop(columns=['image_path'])
out_path = f'outputs/llava_v1.5_7b_baseline/{timestamp_dir}/llava_v1.5_7b_baseline_ChartQA_TEST.xlsx'
df.to_excel(out_path, index=False)
print(f"Saved {df.shape} to {out_path}")
