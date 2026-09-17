import pandas as pd
import os

df = pd.read_csv("chartqa_predictions_cdpruner_qwen.csv")

ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
out_dir = f"outputs/cdpruner_qwen2_5_vl_3b/T{ts}"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "cdpruner_qwen2_5_vl_3b_ChartQA_TEST.xlsx")
df.to_excel(out_path, index=False)
print(f"Saved {out_path}")
