import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

os.environ["CDPRUNER_RETAIN_RATIO"] = "0.223"

import sys
sys.path.insert(0, "/scratch/bh2863/fastv_project/HiPrune/Qwen2_5_VL")
from qwen2_5_vl_CDPruner_ratio import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor, CLIPModel, CLIPProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
clip_path = "openai/clip-vit-large-patch14"

print("Loading Qwen model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained(clip_path).to("cuda").eval()
clip_processor = CLIPProcessor.from_pretrained(clip_path)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

def compute_clip_relevance(image_path, question, llm_grid_h, llm_grid_w):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[question], images=image, return_tensors="pt",
                             padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        patch_embeds = vision_out.last_hidden_state[:, 1:, :]  # drop CLS token
        patch_embeds = clip_model.visual_projection(patch_embeds)
        text_embeds = clip_model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sim = patch_embeds[0] @ text_embeds[0]  # [num_clip_patches]

        side = int(sim.shape[0] ** 0.5)  # 16 for ViT-L/14 @ 224px
        sim_grid = sim.view(1, 1, side, side)
        sim_resized = F.interpolate(sim_grid, size=(llm_grid_h, llm_grid_w),
                                     mode="bilinear", align_corners=False)
        return sim_resized.flatten().to(torch.bfloat16)

predictions = []
start = time.time()

for i, row in df.iterrows():
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": str(row["question"]) + " Answer the question using a single word or phrase."},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to("cuda")

    grid_thw = inputs["image_grid_thw"][0]
    llm_grid_h = int(grid_thw[1].item() // 2)  # spatial_merge_size = 2
    llm_grid_w = int(grid_thw[2].item() // 2)

    clip_relevance = compute_clip_relevance(row["image_path"], row["question"], llm_grid_h, llm_grid_w)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, clip_relevance=clip_relevance, do_sample=False, max_new_tokens=200)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_cdpruner_qwen_matched.csv", index=False)
print("Saved chartqa_predictions_cdpruner_qwen_matched.csv")
