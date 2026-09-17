import pandas as pd, numpy as np
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
df = pd.read_csv("chartqa_plain.csv")
counts = []
for i, row in df.iterrows():
    messages = [{"role": "user", "content": [{"type": "image", "image": row["image_path"]}, {"type": "text", "text": "x"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    t, h, w = inputs["image_grid_thw"][0]
    counts.append(int((h // 2) * (w // 2) * t))
    if (i + 1) % 500 == 0:
        print(i + 1, "done")

counts = np.array(counts)
print(f"mean={counts.mean():.1f} min={counts.min()} max={counts.max()}")
print(f"implied retain % at fixed 192 budget: {192/counts.mean():.3f}")
