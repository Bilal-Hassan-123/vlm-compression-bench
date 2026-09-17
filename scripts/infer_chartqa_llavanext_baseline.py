import time
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"

print("Loading model...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = "USER: <image>\n" + str(row["question"]) + " Answer the question using a single word or phrase.\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("ASSISTANT:")[-1].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_llavanext_baseline.csv", index=False)
print("Saved chartqa_predictions_llavanext_baseline.csv")
