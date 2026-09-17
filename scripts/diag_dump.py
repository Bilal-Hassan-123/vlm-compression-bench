import os, sys, time, pandas as pd, torch
sys.path.insert(0, "/scratch/bh2863/fastv_project/VisionZip/Qwen2_5_VL")
from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
print("Loading Qwen model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

df = pd.read_csv("chartqa_plain.csv").head(40)
print(f"Loaded {len(df)} rows")

predictions, start = [], time.time()
for i, row in df.iterrows():
    messages = [{"role": "user", "content": [
        {"type": "image", "image": row["image_path"]},
        {"type": "text", "text": str(row["question"]) + " Answer the question using a single word or phrase."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    os.environ["DUMP_KEPT"] = f"/scratch/bh2863/diag/row{i}"
    print("IMG:", row["image_path"], "GRID:", inputs["image_grid_thw"].tolist(), flush=True)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=200)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    predictions.append(processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip())

    if (i + 1) % 50 == 0:
        el = time.time() - start
        print(f"{i+1}/{len(df)} done, {el:.1f}s, {el/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("diag_preds.csv", index=False)
print("Saved diag_preds.csv")
