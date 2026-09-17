import os
import time
import pandas as pd
import torch
from PIL import Image

# Env vars are module-level constants read at import time in qwen2_5_vl_HiPrune.py
# MUST be set before importing that module.
os.environ["HIPRUNE_QWEN_RETENTION"] = "0.223"  # ~22% tokens kept, 77.7% pruned
os.environ["HIPRUNE_ALPHA"] = "0.1"
os.environ["HIPRUNE_OBJECT_LAYER"] = "16"

import sys
sys.path.insert(0, "/scratch/bh2863/fastv_project/HiPrune/Qwen2_5_VL")
from qwen2_5_vl_HiPrune import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # HiPrune requires eager or flash_attention_2; flash_attn unavailable on this cluster/CUDA combo
)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image_path"]},
                {"type": "text", "text": str(row["question"]) + " Answer the question using a single word or phrase."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=200)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_hiprune_qwen.csv", index=False)
print("Saved chartqa_predictions_hiprune_qwen.csv")
