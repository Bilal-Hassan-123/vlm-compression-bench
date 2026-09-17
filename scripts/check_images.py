import pandas as pd
from PIL import Image

df = pd.read_csv("plotqa_plain.csv")
bad_rows = []

for i, row in df.iterrows():
    try:
        img = Image.open(row["image_path"])
        img.verify()  # checks file integrity without fully decoding
    except Exception as e:
        bad_rows.append(row["index"])

print(f"Checked {len(df)} rows, found {len(bad_rows)} broken images")
print(bad_rows[:20])  # preview first 20 bad indices

# Save the cleaned version, dropping broken rows
clean_df = df[~df["index"].isin(bad_rows)].reset_index(drop=True)
clean_df.to_csv("plotqa_plain.csv", index=False)
print(f"Saved cleaned plotqa_plain.csv with {len(clean_df)} rows")
