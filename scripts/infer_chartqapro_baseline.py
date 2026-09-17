import time
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
revision = "a272c74"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
).to(0)

processor = AutoProcessor.from_pretrained(model_id, revision=revision)

df = pd.read_csv("chartqapro_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = "USER: <image>\n" + str(row["question"]) + " Answer concisely with just the final answer, no explanation.\nASSISTANT:"

    inputs = processor(prompt, image, return_tensors="pt").to(0, torch.float16)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=False,
        )

    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("ASSISTANT:")[-1].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqapro_predictions_baseline.csv", index=False)
print("Saved chartqapro_predictions_baseline.csv")
