import pandas as pd
import sys

timestamp_dir = sys.argv[1]

df = pd.read_csv('chartqapro_predictions_hiprune_qwen.csv')
df = df.drop(columns=['image_path'])
out_path = f'outputs/hiprune_qwen2_5_vl_3b/{timestamp_dir}/hiprune_qwen2_5_vl_3b_ChartQAPro.xlsx'
df.to_excel(out_path, index=False)
print(f"Saved {df.shape} to {out_path}")
