import os
import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.smp import decode_base64_to_image_file

d = build_dataset('ChartQAPro')
df = d.data.copy()

os.makedirs('chartqapro_images', exist_ok=True)
rows = []
for _, row in df.iterrows():
    img_path = f"chartqapro_images/{row['index']}.jpg"
    decode_base64_to_image_file(row['image'], img_path)
    rows.append({
        'index': row['index'],
        'image_path': img_path,
        'question': row['question'],
        'answer': row['answer'],
        'question_type': row['question_type'],
        'year': row['year'],
        'paragraph': row['paragraph'],
    })

pd.DataFrame(rows).to_csv('chartqapro_plain.csv', index=False)
print(f"Saved {len(rows)} rows to chartqapro_plain.csv")
